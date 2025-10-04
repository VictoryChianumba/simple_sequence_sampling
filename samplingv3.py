import uuid
import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer


def remove_sequence_from_cache(past_key_values, index_to_remove: int):
    if past_key_values is None:
        return None

    # Ensure we are working with a list for modification
    if isinstance(past_key_values, tuple):
        past_key_values = list(past_key_values)

    if index_to_remove < 0 or index_to_remove >= past_key_values[0][0].size(0):
        raise ValueError(f"Invalid index_to_remove {index_to_remove}")

    out = []
    for k, v in past_key_values:
        # Create a new tensor by selecting all indices except the one to remove
        k_new = torch.cat([k[:index_to_remove], k[index_to_remove + 1:]], dim=0).contiguous()
        v_new = torch.cat([v[:index_to_remove], v[index_to_remove + 1:]], dim=0).contiguous()
        out.append((k_new, v_new))

    # CRITICAL FIX: Return a tuple to match the model's expected format
    return tuple(out)


def add_sequence_to_cache(past_key_values, new_kv_cache, pad_to_length=None):
    """
    Concatenate new_kv_cache to past_key_values along batch dim.
    If pad_to_length is given, pad new sequence's *seq* dim (dim=2) to that length
    so every sequence in the batch has identical seq_len before the batch concat.
    """
    if past_key_values is None:
        return new_kv_cache

    # Ensure we are working with lists for modification
    if isinstance(past_key_values, tuple):
        past_key_values = list(past_key_values)
    if isinstance(new_kv_cache, tuple):
        new_kv_cache = list(new_kv_cache)

    if len(past_key_values) != len(new_kv_cache):
        raise ValueError("Layer count mismatch")

    out = []
    for (k_old, v_old), (k_new, v_new) in zip(past_key_values, new_kv_cache):
        # ---- optional padding along seq dim ----
        if pad_to_length is not None:
            current_len = k_new.shape[2]
            pad_len = pad_to_length - current_len
            if pad_len > 0:
                k_pad = torch.zeros(1, k_new.shape[1], pad_len, k_new.shape[3],
                                    dtype=k_new.dtype, device=k_new.device)
                v_pad = torch.zeros(1, v_new.shape[1], pad_len, v_new.shape[3],
                                    dtype=v_new.dtype, device=v_new.device)
                k_new = torch.cat([k_pad, k_new], dim=2)
                v_new = torch.cat([v_pad, v_new], dim=2)

        # ---- concat along batch dim ----
        k_cat = torch.cat([k_old, k_new], dim=0).contiguous()
        v_cat = torch.cat([v_old, v_new], dim=0).contiguous()
        out.append((k_cat, v_cat))

    # CRITICAL FIX: Return a tuple to match the model's expected format
    return tuple(out)


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
        self.completed_sequences = [] # NEW: Store finished sequences

        self.decode_steps = 0

    def add_request(self, prompt, max_new_tokens, sampling_fn):
        """Add a new generation request to the queue."""
        request = {
            "request_id": self.next_request_id,
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
            "sampling_fn": sampling_fn,
            "input_ids": self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.model.device),
            "tokens_generated": 0,
            "finished": False,
        }
        self.next_request_id += 1
        self.pending_requests.append(request)
        return request["request_id"]

    def step(self):
        """One decode step for all active sequences."""
        # Promote pending -> active (prefill new ones)
        while len(self.active_sequences) < self.max_batch_size and self.pending_requests:
            req = self.pending_requests.pop(0)
            ids = req["input_ids"]

            with torch.no_grad():
                out = self.model(ids, use_cache=True, return_dict=True)

            new_kv = out.past_key_values
            # Find the max sequence length in the current batch to pad the new sequence
            pad_length = self.past_key_values[0][0].shape[2] if self.past_key_values else None
            self.past_key_values = add_sequence_to_cache(self.past_key_values, new_kv, pad_to_length=pad_length)
            self.active_sequences.append(req)

        if not self.active_sequences:
            return

        last_tokens = torch.cat([seq["input_ids"][:, -1:] for seq in self.active_sequences], dim=0)

        with torch.no_grad():
            out = self.model(
                last_tokens,
                past_key_values=self.past_key_values,
                use_cache=True,
                return_dict=True
            )
            logits = out.logits[:, -1, :]
            self.past_key_values = out.past_key_values # This is a tuple

            for i, seq in enumerate(self.active_sequences):
                nxt = seq["sampling_fn"](logits[i:i+1, :])
                seq["input_ids"] = torch.cat([seq["input_ids"], nxt], dim=1)
                seq["tokens_generated"] += 1
                if nxt.item() == self.tokenizer.eos_token_id or \
                   seq["tokens_generated"] >= seq["max_new_tokens"]:
                    seq["finished"] = True

            self.decode_steps += 1

            # Remove finished sequences
            to_remove_indices = [i for i, s in enumerate(self.active_sequences) if s["finished"]]
            # Store completed sequences before removing them
            for i in to_remove_indices:
                self.completed_sequences.append(self.active_sequences[i])

            # Remove from active list and KV cache in reverse order
            for idx in reversed(to_remove_indices):
                self.active_sequences.pop(idx)
                self.past_key_values = remove_sequence_from_cache(self.past_key_values, idx)

    def run_until_complete(self):
        """Keep stepping until all requests are done."""
        while self.pending_requests or self.active_sequences:
            self.step()


# --- Main execution ---
tok = AutoTokenizer.from_pretrained("gpt2")
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained("gpt2").eval().to("mps")

generator = ContinuousBatchGenerator(
    model=model,
    tokenizer=tok,
    max_batch_size=4,
    max_seq_len=100
)

prompts = [
    "The quick brown fox",
    "Hello world",
    "Machine learning is",
    "Python programming"
]

greedy = lambda logits: logits.argmax(dim=-1, keepdim=True)

for prompt in prompts:
    generator.add_request(prompt, max_new_tokens=20, sampling_fn=greedy)

start = time.perf_counter()
generator.run_until_complete()
elapsed = time.perf_counter() - start

# Print results from the completed_sequences list
for seq in generator.completed_sequences:
    print(f"Prompt: {seq['prompt']}")
    print(f"Generated: {tok.decode(seq['input_ids'][0], skip_special_tokens=True)}")
    print("---")

print(f"Total time: {elapsed:.2f}s")
print(f"Total decode steps: {generator.decode_steps}")