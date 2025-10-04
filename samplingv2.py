import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Dict, List
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
from test import sample_temperature, sample_topk, sample_topp


@torch.no_grad()
def generate_batch(model, 
                   input_ids: torch.Tensor, 
                   max_new_tokens: int, 
                   sampling_fn, 
                   eos_token_id: int, 
                   pad_token_id: int= None,
                   orig_lens = None
                   ) -> Dict[str, List[torch.Tensor]]:
    
    device = input_ids.device
    batch_size, seq_length = input_ids.shape
    
    if pad_token_id is None:
        pad_token_id = eos_token_id
    
    torch.mps.synchronize() 
    start = time.perf_counter()
    outputs = model(input_ids, use_cache=True, return_dict=True)
    print(outputs.logits.shape)
    prefill_time = time.perf_counter() - start    
    
    past_key_values = outputs.past_key_values
    next_token = outputs.logits[:, -1,:].argmax(dim=-1, keepdim=True)
    generated = torch.cat([input_ids, next_token], dim=-1)

    finished = torch.zeros(batch_size, dtype=torch.bool, device = device)
    
    #decode start
    
    decode_start = time.perf_counter()
    for _ in range(max_new_tokens):
        
        out = model(generated[:, -1:], past_key_values, use_cache=True, return_dict = True)   
        logits = out.logits[:, -1,  :] #tensor of (1, vocab_size)   
        past_key_values = out.past_key_values# update the cache
        
        next_token = sampling_fn(logits)
        next_token = next_token.masked_fill(finished.unsqueeze(1), pad_token_id)
        generated = torch.cat([generated, next_token], dim=-1)
    
        just_finished = (next_token.squeeze(1) == eos_token_id) & (~finished)
        finished |=just_finished
        
        if finished.all():
            break 
    decode_time = time.perf_counter() - decode_start
    torch.mps.synchronize() 
    final_lens = [generated[i].shape[0] for i in range(batch_size)]
    num_generated = [final_lens[i] - orig_lens[i] for i in range(batch_size)]
    total_gen = sum(num_generated)
    return {
        "sequences": [generated[i] for i in range(batch_size)],
        'prefill_time': prefill_time,
        'decode_time': decode_time,
        'num_generated': num_generated,
        'tokens_per_sec': total_gen / decode_time
    }

tok = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2").eval().to("mps")
tok.pad_token = tok.eos_token          # reuse EOS as pad
tok.padding_side = "left"              # crucial

BATCH_SIZES = [1, 2, 4, 8, 16]
MAX_NEW = 30
records = []

device = "mps" if torch.backends.mps.is_available() else "cpu"
model = model.to(device)
PROMPTS = ["Hello", "Good morning, world!", "Hi"]
greedy = lambda logits: logits.argmax(dim=-1, keepdim=True)
for b in BATCH_SIZES:
    prompts = (PROMPTS * ((b // len(PROMPTS)) + 1))[:b]

    enc = tok(prompts, return_tensors="pt", padding=True).to(device)

    orig_lens = [len(tok.encode(p)) for p in prompts]  
    
    _ = generate_batch(model, enc.input_ids, 5, lambda l: sample_temperature(l, 0.7),
                       tok.eos_token_id, tok.eos_token_id, orig_lens)
    # measured run
    torch.mps.empty_cache()
    if hasattr(torch.mps, 'reset_peak_memory_stats'):
        torch.mps.reset_peak_memory_stats()
        
    out = generate_batch(model, enc.input_ids, MAX_NEW, lambda l: sample_temperature(l, 0.7),
                        tok.eos_token_id, tok.eos_token_id, orig_lens)

    # metrics
    # 
    total_gen = sum(out["num_generated"])
    total_time = out["prefill_time"] + out["decode_time"]
    total_tok_per_sec = total_gen / total_time
    per_seq_tok_per_sec = total_tok_per_sec / b
    
    records.append({
        "batch_size": b,
        "total_tokens/sec": total_tok_per_sec,
        "per_seq_tokens/sec": per_seq_tok_per_sec,
        "prefill_ms": out["prefill_time"] * 1000,
        "decode_ms": out["decode_time"] * 1000
    })



# results = generate_batch(
#     model,
#     enc.input_ids,
#     max_new_tokens=30,
#     sampling_fn=lambda l: sample_temperature(l, temperature=0.7),
#     eos_token_id=tok.eos_token_id,
#     pad_token_id=tok.eos_token_id,    
#     orig_lens=orig_lens
# )

    

# for seq in results["sequences"]:
#     print(tok.decode(seq, skip_special_tokens=True))
    
# print(results["tokens_per_sec"], "tokens/sec")


df = pd.DataFrame(records)
print(df.round(2))
