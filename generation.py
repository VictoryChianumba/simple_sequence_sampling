import cProfile
import pstats
from samplingv3 import ContinuousBatchGenerator
import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np

tok = AutoTokenizer.from_pretrained("gpt2")
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained("gpt2", dtype=torch.float16).eval().to("mps")

# --- Smaller batch size than total requests ---
generator = ContinuousBatchGenerator(
    model=model,
    tokenizer=tok,
    max_batch_size=2,  # Batch size is 2
    max_seq_len=100
)

# # ---  Requests with different lengths ---
requests_to_add = [
    {"prompt": "The best thing about AI is", "max_new_tokens": 10},  # Short
    {"prompt": "In a world where technology has advanced", "max_new_tokens": 50}, # Long
    {"prompt": "Python is a great language because", "max_new_tokens": 10}, # Short
    {"prompt": "The history of machine learning began", "max_new_tokens": 50}, # Long
]

greedy = lambda logits: logits.argmax(dim=-1, keepdim=True)

print("--- Starting Generation ---")
for req in requests_to_add:
    req_id = generator.add_request(
        prompt=req["prompt"],
        max_new_tokens=req["max_new_tokens"],
        sampling_fn=greedy
    )
    print(f"Added request {req_id}: '{req['prompt']}' (max {req['max_new_tokens']} tokens)")

start = time.perf_counter()
generator.run_until_complete()
elapsed = time.perf_counter() - start

print("\n--- Generation Complete ---")
# Print results in the order they were added
print("\n--- Results ---")
for original_req in requests_to_add:
    # Find the corresponding completed sequence
    for seq in generator.completed_sequences:
        if seq["prompt"] == original_req["prompt"]:
            # Reconstruct full sequence from prompt + generated tokens
            # full_ids = torch.cat([seq["input_ids"], torch.tensor([seq["generated_tokens"]], device='mps')], dim=1)
            # decoded = tok.decode(full_ids[0], skip_special_tokens=True)
            print(f"Prompt: '{seq['prompt']}'")
            print(f"Generated: '{tok.decode(seq['input_ids'][0], skip_special_tokens=True)}'")
            print(f"Tokens generated: {seq['tokens_generated']}")
            print("---")


print(f"\nTotal time: {elapsed:.2f}s")
print(f"Total decode steps: {generator.decode_steps}")