import cProfile
import pstats
from samplingv3 import ContinuousBatchGenerator
import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np

profiler = cProfile.Profile()
profiler.enable()

tok = AutoTokenizer.from_pretrained("gpt2")
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained("gpt2", dtype=torch.float16).eval().to("mps")
greedy = lambda logits: logits.argmax(dim=-1, keepdim=True)


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

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)  # Top 20 functions by time