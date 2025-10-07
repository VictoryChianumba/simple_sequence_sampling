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
        # In your __init__
        # self.model = torch.compile(model, mode="reduce-overhead")
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
        
        # Pre-allocated KV cache
        self.max_cache_len = max_seq_len                                      
        self.kv_cache_buffer = None  # Lazily initialize when we know layer config
        self.active_slots = [False] * max_batch_size  # Track which slots are occupied
        self.slot_to_request = [None] * max_batch_size  # Map slot index to request metadata
        
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
    
    def _initialize_cache_buffer(self, sample_cache):
        """Initialize pre-allocated cache based on model architecture."""
        if self.kv_cache_buffer is not None:
            return  # Already initialized
        
        num_layers = len(sample_cache)
        # Get dimensions from first layer's key tensor
        _, num_heads, _, head_dim = sample_cache[0][0].shape
        device = sample_cache[0][0].device
        dtype = sample_cache[0][0].dtype
        
        # Allocate buffer: list of (key_buffer, value_buffer) per layer
        self.kv_cache_buffer = []
        for _ in range(num_layers):
            k_buffer = torch.zeros(
                self.max_batch_size, num_heads, self.max_cache_len, head_dim,
                dtype=dtype, device=device
            )
            v_buffer = torch.zeros(
                self.max_batch_size, num_heads, self.max_cache_len, head_dim,
                dtype=dtype, device=device
            )
            self.kv_cache_buffer.append((k_buffer, v_buffer))
    
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
            # We find an empty slot
            try:
                empty_slot = self.active_slots.index(False)     
            except ValueError:
                break
            
            req = self.pending_requests.pop(0)
            ids = req["input_ids"] 
            
            # prefill for this single sequence
            with torch.no_grad():
                out = self.model(ids, use_cache = True, return_dict = True)
            
            # Initialise KV_cache_buffer when None
            if self.kv_cache_buffer is None:
                self._initialize_cache_buffer(out.past_key_values)
                
            seq_len = out.past_key_values[0][0].shape[2]
 
            # Copy the new KV cache to the allocated slot
            for layer_idx, (k, v) in enumerate(out.past_key_values):
                # Copy to the appropriate slot in our pre-allocated buffer
                self.kv_cache_buffer[layer_idx][0][empty_slot, :, :seq_len, :] = k
                self.kv_cache_buffer[layer_idx][1][empty_slot, :, :seq_len, :] = v
            
            # Mark slot as occupied and store request metadata
            self.active_slots[empty_slot] = True
            self.slot_to_request[empty_slot] = req
            req["cache_slot"] = empty_slot  # Store slot index in request
            req["current_seq_len"] = seq_len  # Track current sequence length
            
            self.active_sequences.append(req)
        
        if not self.active_sequences:
            return

        # Extract active slots from buffer to pass to model
        active_slot_indices = [seq['cache_slot'] for seq in self.active_sequences]
        max_seq_len_in_batch = max(seq['current_seq_len'] for seq in self.active_sequences)

        # Build cache for active sequences only
        active_cache = []
        for layer_idx in range(len(self.kv_cache_buffer)):
            k_buffer, v_buffer = self.kv_cache_buffer[layer_idx]
            # Extract only active slots, up to max_seq_len_in_batch
            k_active = k_buffer[active_slot_indices, :, :max_seq_len_in_batch, :].contiguous()
            v_active = v_buffer[active_slot_indices, :, :max_seq_len_in_batch, :].contiguous()
            active_cache.append((k_active, v_active))

        # Gather last tokens
        last_tokens = torch.cat([seq["input_ids"][:, -1:] for seq in self.active_sequences], dim=0)

        with torch.no_grad():
            out = self.model(last_tokens, past_key_values=tuple(active_cache), use_cache=True, return_dict=True)
            logits = out.logits[:, -1, :]
            
            # Write updated cache back to buffer
            for i, seq in enumerate(self.active_sequences):
                slot = seq['cache_slot']
                seq_len = seq['current_seq_len']
                
                # Sample next token
                nxt = seq["sampling_fn"](logits[i:i+1, :])
                seq["input_ids"] = torch.cat([seq["input_ids"], nxt], dim=1)
                seq["tokens_generated"] += 1
                
                # Write updated KV for this sequence back to buffer
                for layer_idx in range(len(self.kv_cache_buffer)):
                    # Extract just the new token's KV
                    new_k = out.past_key_values[layer_idx][0][i:i+1, :, -1:, :]
                    new_v = out.past_key_values[layer_idx][1][i:i+1, :, -1:, :]
                    
                    # Update the cache at the right position
                    self.kv_cache_buffer[layer_idx][0][slot, :, seq_len:seq_len+1, :] = new_k
                    self.kv_cache_buffer[layer_idx][1][slot, :, seq_len:seq_len+1, :] = new_v
                
                seq['current_seq_len'] = seq_len + 1
                
                # Check if finished
                if nxt.item() == self.tokenizer.eos_token_id or seq["tokens_generated"] >= seq["max_new_tokens"]:
                    seq["finished"] = True 
                        
            self.decode_steps += 1

            # Remove finished sequences
            to_remove = [i for i, s in enumerate(self.active_sequences) if s["finished"]]

            for idx in to_remove:
                seq = self.active_sequences[idx]
                self.completed_sequences.append(seq)
                
                # Track latency
                req_id = seq['request_id']
                submit_time = self.metrics['request_submit_times'][req_id]
                complete_time = time.perf_counter()
                self.metrics['request_latencies'][req_id] = complete_time - submit_time
                
                # Free the slot
                slot = seq['cache_slot']
                self.active_slots[slot] = False
                self.slot_to_request[slot] = None

            # Remove from active list (in reverse to maintain indices)
            for idx in reversed(to_remove):
                self.active_sequences.pop(idx)

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
model = AutoModelForCausalLM.from_pretrained("gpt2", dtype=torch.float16).eval().to("mps")

greedy = lambda logits: logits.argmax(dim=-1, keepdim=True)
