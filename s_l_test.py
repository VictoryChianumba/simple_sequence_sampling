import cProfile
import pstats
from samplingv3 import ContinuousBatchGenerator, analyze_metrics
from samplingv2 import generate_batch
import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np

tok = AutoTokenizer.from_pretrained("gpt2")
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained("gpt2", dtype=torch.float16).eval().to("mps")

greedy = lambda logits: logits.argmax(dim=-1, keepdim=True)

generator = ContinuousBatchGenerator(model, tok, max_batch_size=4, max_seq_len=100)
for i in range(8):
    generator.add_request(f"Prompt {i}", max_new_tokens=20, sampling_fn=greedy)

print("============== Short =============")
start = time.perf_counter()
generator.run_until_complete()
v3_time_s = time.perf_counter() - start
analyze_metrics(generator)
print(f"Total wall time: {v3_time_s:.6f}s\n")

generator = ContinuousBatchGenerator(model, tok, max_batch_size=4, max_seq_len=100)
# Alternate short and long
for i in range(4):
    generator.add_request(f"Short {i}", max_new_tokens=10, sampling_fn=greedy)
    generator.add_request(f"Long {i}", max_new_tokens=50, sampling_fn=greedy)

print("============== Short & Long =============")
start = time.perf_counter()
generator.run_until_complete()
v3_time_sl = time.perf_counter() - start
analyze_metrics(generator)
print(f"Total wall time: {v3_time_sl:.6f}s\n")

# Test naive batching (Version 2)
print("=== Naive Batching (Version 2) ===")
prompts = []
max_tokens = []
for i in range(4):
    prompts.append(f"Short {i}")
    max_tokens.append(10)
    prompts.append(f"Long {i}")
    max_tokens.append(50)

enc = tok(prompts, return_tensors="pt", padding=True).to("mps")
orig_lens = [len(tok.encode(p)) for p in prompts]

start = time.perf_counter()
result = generate_batch(model, enc.input_ids, max(max_tokens), greedy, tok.eos_token_id, tok.eos_token_id, orig_lens)
v2_time = time.perf_counter() - start

print(f"Total throughput: {result['tokens_per_sec']:.2f} tokens/sec")
print(f"Total wall time: {v2_time:.2f}s")

print(f"\nSpeedup SL: {v2_time / v3_time_sl:.2f}x")

