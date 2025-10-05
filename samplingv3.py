import uuid
import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from samplingv2 import generate_batch

def remove_sequence_from_cache(past_key_values , index_to_remove:int):
    if past_key_values is None: return None
    
    if isinstance(past_key_values, tuple):
        past_key_values = list(past_key_values)
    
    if index_to_remove < 0 or index_to_remove >= past_key_values[0][0].size(0):
        return ValueError(f"Invalid index_to_remove {index_to_remove} ")
    
    out = []    
    for k, v in past_key_values:
        indices = list(range(k.size(0)))
        indices.pop(index_to_remove)
        
        k_new = k[indices].contiguous()
        v_new = v[indices].contiguous()
    
        out.append((k_new, v_new))
        
    return tuple(out)
    
def add_sequence_to_cache(past_key_values, new_kv_cache):
    """
    Concatenate new_kv_cache to past_key_values along batch dim.
    Pads the shorter cache to match the longer one before concatenation.
    """
    if past_key_values is None:
        return new_kv_cache

    if len(past_key_values) != len(new_kv_cache):
        raise ValueError("Layer count mismatch")

    # Determine which cache needs padding
    old_seq_len = past_key_values[0][0].shape[2]
    new_seq_len = new_kv_cache[0][0].shape[2]

    out = []
    for (k_old, v_old), (k_new, v_new) in zip(past_key_values, new_kv_cache):
        if old_seq_len < new_seq_len:
            # Case 1: New sequence is longer. Pad the OLD cache to match.
            pad_len = new_seq_len - old_seq_len
            k_pad = torch.zeros(k_old.shape[0], k_old.shape[1], pad_len, k_old.shape[3],
                                dtype=k_old.dtype, device=k_old.device)
            v_pad = torch.zeros(v_old.shape[0], v_old.shape[1], pad_len, v_old.shape[3],
                                dtype=v_old.dtype, device=v_old.device)
            
            k_old_padded = torch.cat([k_pad, k_old], dim=2)
            v_old_padded = torch.cat([v_pad, v_old], dim=2)

            k_cat = torch.cat([k_old_padded, k_new], dim=0).contiguous()
            v_cat = torch.cat([v_old_padded, v_new], dim=0).contiguous()

        else:
            # Case 2: New sequence is shorter or same length. Pad the NEW cache to match.
            pad_len = old_seq_len - new_seq_len
            k_pad = torch.zeros(k_new.shape[0], k_new.shape[1], pad_len, k_new.shape[3],
                                dtype=k_new.dtype, device=k_new.device)
            v_pad = torch.zeros(v_new.shape[0], v_new.shape[1], pad_len, v_new.shape[3],
                                dtype=v_new.dtype, device=v_new.device)
            
            k_new_padded = torch.cat([k_pad, k_new], dim=2)
            v_new_padded = torch.cat([v_pad, v_new], dim=2)

            k_cat = torch.cat([k_old, k_new_padded], dim=0).contiguous()
            v_cat = torch.cat([v_old, v_new_padded], dim=0).contiguous()
        
        out.append((k_cat, v_cat))
            
    return tuple(out)

def analyze_metrics(generator):
    metrics = generator.metrics
    
    total_tokens = sum(seq['tokens_generated'] for seq in generator.completed_sequences)
    total_time = sum(metrics['step_times'])
    
    latencies = list(metrics['request_latencies'].values())
    avg_latency = np.mean(latencies)
    p50 = np.percentile(latencies, 50)
    p90 = np.percentile(latencies, 90)
    p99 = np.percentile(latencies, 99)
    
    avg_batch_size = np.mean(metrics['batch_sizes'])
    
    print(f"Total throughput: {total_tokens / total_time:.2f} tokens/sec")
    print(f"Average latency: {avg_latency:.3f}s")
    print(f"Latency p50/p90/p99: {p50:.3f}s / {p90:.3f}s / {p99:.3f}s")
    print(f"Average batch size: {avg_batch_size:.2f}")
    
class ContinuousBatchGenerator:
    def __init__(self, model, tokenizer, max_batch_size, max_seq_len):
        self.model = model
        self.tokenizer = tokenizer
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.next_request_id = 0
        
        # Active sequences being generated
        self.active_sequences = []  # List of dicts with metadata
        self.past_key_values = None
        
        # Request queue
        self.pending_requests = []
        self.completed_sequences = [] # Store finished sequences
        
        self.decode_steps = 0
        
        # Benchmarking metrics
        self.metrics = {
            'step_times': [],
            'batch_sizes': [],
            'request_latencies': {},  # request_id -> (submit_time, complete_time)
            'request_submit_times': {},
        }
    
    def add_request(self, prompt, max_new_tokens, sampling_fn):
        """Add a new generation request to the queue."""
        request = {
            "request_id": self.next_request_id,  # unique string
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
            "sampling_fn": sampling_fn,
           
            "input_ids": self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.model.device),
            "tokens_generated": 0,
            "finished": False,
        }
        req_id = request["request_id"]
        self.next_request_id +=1
        self.pending_requests.append(request)
        self.metrics['request_submit_times'][req_id] = time.perf_counter()
        return req_id 
    
    def step(self):
        """
        One decode step for all active sequences.
        - Generate next token for each active sequence
        - Remove finished sequences
        - Add new sequences from queue if space available
        """
        device = self.model.device
        step_start = time.perf_counter()
        
        # Promote pending -> active  
        while len(self.active_sequences) < self.max_batch_size and self.pending_requests:
            req = self.pending_requests.pop(0)
            ids = req["input_ids"] 
            
            # prefill for this single sequence
            with torch.no_grad():
                out = self.model(ids, use_cache = True, return_dict = True)
            
            # merge kv into global cache
            new_kv = out.past_key_values
            # Find the max sequence length in the current batch to pad the new sequence

            self.past_key_values = add_sequence_to_cache(self.past_key_values, new_kv)            
            self.active_sequences.append(req)
            
        if not self.active_sequences:
            return 
        
        last_tokens = torch.cat([seq["input_ids"][:, -1:] for seq in self.active_sequences], dim=0)
        
        with torch.no_grad():
            out = self.model(last_tokens, 
                             past_key_values=self.past_key_values,
                             use_cache=True,
                             return_dict=True)
            logits = out.logits[:, -1, :]
            self.past_key_values = out.past_key_values
            
            for i , seq in enumerate(self.active_sequences):
                nxt = seq["sampling_fn"](logits[i:i+1, :])
                seq["input_ids"] = torch.cat([seq["input_ids"], nxt], dim=1)   
                seq["tokens_generated"] += 1
                if nxt.item() == self.tokenizer.eos_token_id or \
                    seq["tokens_generated"] >= seq["max_new_tokens"]:
                        seq["finished"] = True  
                        
            self.decode_steps += 1
            
            to_remove = [i for i, s in enumerate(self.active_sequences) if s["finished"]]
            
            for i in to_remove:
                self.completed_sequences.append(self.active_sequences[i])
                
            for idx in to_remove:
                seq = self.active_sequences[idx]
                req_id = seq['request_id']
                submit_time = self.metrics['request_submit_times'][req_id]
                complete_time = time.perf_counter()
                self.metrics['request_latencies'][req_id] = complete_time - submit_time
            
            for idx in reversed(to_remove):
                self.active_sequences.pop(idx)
                self.past_key_values = remove_sequence_from_cache(self.past_key_values, idx)
                
            # Track completed request latencies
           
                
            step_time = time.perf_counter() - step_start
            self.metrics['step_times'].append(step_time)
            self.metrics['batch_sizes'].append(len(self.active_sequences))
                
        # print(f"Step {self.decode_steps}: Active={len(self.active_sequences)}, Pending={len(self.pending_requests)}")

            
    
    def run_until_complete(self):
        """Keep stepping until all requests are done."""
        while self.pending_requests or self.active_sequences:
            self.step()
            
            
tok = AutoTokenizer.from_pretrained("gpt2")
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained("gpt2").eval().to("mps")

# --- Key Change 1: Smaller batch size than total requests ---
# generator = ContinuousBatchGenerator(
#     model=model,
#     tokenizer=tok,
#     max_batch_size=2,  # Batch size is 2
#     max_seq_len=100
# )

# # --- Key Change 2: Requests with different lengths ---
# requests_to_add = [
#     {"prompt": "The best thing about AI is", "max_new_tokens": 10},  # Short
#     {"prompt": "In a world where technology has advanced", "max_new_tokens": 50}, # Long
#     {"prompt": "Python is a great language because", "max_new_tokens": 10}, # Short
#     {"prompt": "The history of machine learning began", "max_new_tokens": 50}, # Long
# ]

greedy = lambda logits: logits.argmax(dim=-1, keepdim=True)

# print("--- Starting Generation ---")
# for req in requests_to_add:
#     req_id = generator.add_request(
#         prompt=req["prompt"],
#         max_new_tokens=req["max_new_tokens"],
#         sampling_fn=greedy
#     )
#     print(f"Added request {req_id}: '{req['prompt']}' (max {req['max_new_tokens']} tokens)")

# start = time.perf_counter()
# generator.run_until_complete()
# elapsed = time.perf_counter() - start

# print("\n--- Generation Complete ---")
# # Print results in the order they were added
# print("\n--- Results ---")
# for original_req in requests_to_add:
#     # Find the corresponding completed sequence
#     for seq in generator.completed_sequences:
#         if seq["prompt"] == original_req["prompt"]:
#             print(f"Prompt: '{seq['prompt']}'")
#             print(f"Generated: '{tok.decode(seq['input_ids'][0], skip_special_tokens=True)}'")
#             print(f"Tokens generated: {seq['tokens_generated']}")
#             print("---")


# print(f"\nTotal time: {elapsed:.2f}s")
# print(f"Total decode steps: {generator.decode_steps}")

# generator = ContinuousBatchGenerator(model, tok, max_batch_size=4, max_seq_len=100)
# for i in range(8):
#     generator.add_request(f"Prompt {i}", max_new_tokens=20, sampling_fn=greedy)

# print("============== Short =============")
# start = time.perf_counter()
# generator.run_until_complete()
# v3_time_s = time.perf_counter() - start
# analyze_metrics(generator)
# print(f"Total wall time: {v3_time_s:.6f}s\n")

# generator = ContinuousBatchGenerator(model, tok, max_batch_size=4, max_seq_len=100)
# # Alternate short and long
# for i in range(4):
#     generator.add_request(f"Short {i}", max_new_tokens=10, sampling_fn=greedy)
#     generator.add_request(f"Long {i}", max_new_tokens=50, sampling_fn=greedy)

# print("============== Short & Long =============")
# start = time.perf_counter()
# generator.run_until_complete()
# v3_time_sl = time.perf_counter() - start
# analyze_metrics(generator)
# print(f"Total wall time: {v3_time_sl:.6f}s\n")

# # Test naive batching (Version 2)
# print("=== Naive Batching (Version 2) ===")
# prompts = []
# max_tokens = []
# for i in range(4):
#     prompts.append(f"Short {i}")
#     max_tokens.append(10)
#     prompts.append(f"Long {i}")
#     max_tokens.append(50)

# enc = tok(prompts, return_tensors="pt", padding=True).to("mps")
# orig_lens = [len(tok.encode(p)) for p in prompts]

# start = time.perf_counter()
# result = generate_batch(model, enc.input_ids, max(max_tokens), greedy, tok.eos_token_id, tok.eos_token_id, orig_lens)
# v2_time = time.perf_counter() - start

# print(f"Total throughput: {result['tokens_per_sec']:.2f} tokens/sec")
# print(f"Total wall time: {v2_time:.2f}s")

# print(f"\nSpeedup SL: {v2_time / v3_time_sl:.2f}x")

# Version 3: Mixed workload
generator = ContinuousBatchGenerator(model, tok, max_batch_size=4, max_seq_len=100)
short_ids = []
long_ids = []
for i in range(4):
    short_ids.append(generator.add_request(f"Short {i}", max_new_tokens=10, sampling_fn=greedy))
    long_ids.append(generator.add_request(f"Long {i}", max_new_tokens=50, sampling_fn=greedy))

generator.run_until_complete()

# Compare short vs long latencies
short_latencies = [generator.metrics['request_latencies'][id] for id in short_ids]
long_latencies = [generator.metrics['request_latencies'][id] for id in long_ids]

print(f"Short request latencies: {np.mean(short_latencies):.2f}s")
print(f"Long request latencies: {np.mean(long_latencies):.2f}s")