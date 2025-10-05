import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import numpy as np


@torch.no_grad()
def generate(model, input_ids, max_new_tokens, sampling_fn,eos_token_id):
    """
    Args:
        model (_type_): From HuggingFace. We use GPT2 as we are on an M1 chip
        input_ids (_type_): _description_
        max_new_tokens (_type_): _description_
        sampling_fn (_type_): _description_
    """
    
    start = time.perf_counter()
    outputs = model(input_ids, use_cache=True, return_dict=True)
    print(outputs.logits.shape)
    prefill_time = time.perf_counter() - start
    
    past_key_values = outputs.past_key_values
    next_token = outputs.logits[:, -1,:].argmax(dim=-1, keepdim=True)
    generated = torch.cat([input_ids, next_token], dim=-1)
    
    #decode start
    decode_start = time.perf_counter()
    for _ in range(max_new_tokens-1):
        out = model(generated[:,-1:], past_key_values = past_key_values, use_cache = True, return_dict = True)        
        logits = out.logits[:, -1,  :] #tensor of (1, vocab_size)   
        past_key_values = out.past_key_values# update the cache
        
        next_token = sampling_fn(logits)
        generated = torch.cat([generated, next_token], dim=-1)


        if next_token.item()== eos_token_id:
            break
    decode_time = time.perf_counter() - decode_start
    num_generated = generated.shape[1] - input_ids.shape[1]
    return {
    'generated_ids': generated,
    'prefill_time': prefill_time,
    'decode_time': decode_time,
    'num_generated': num_generated,
    'tokens_per_sec': num_generated / decode_time
}

def sample_temperature(logits, temperature=1.0):
    if temperature == 0.0:
        return logits.argmax(dim=-1, keepdim=True)
    logits_t = logits / temperature
    probs =  F.softmax(logits_t, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def sample_topk(logits, k=50):
    # Keep only top-k logits, set rest to -inf, then sample
    
    vocab_size = logits.size(-1)
    k = min(k, vocab_size)
    
    outputs, idxs = torch.topk(logits, k, dim=-1)
    masked = torch.full_like(logits, float('-inf'))
    masked.scatter_(dim=-1, index=idxs, src=outputs)

    probs = F.softmax(masked, dim=-1)
    token = torch.multinomial(probs, num_samples = 1)
    return token

def sample_topp(logits, p=0.9):
    # Keep smallest set of tokens whose cumulative probability >= p
    # Set rest to -inf, then sample
    sorted_logits, sorted_ids = torch.sort(logits, dim=-1, descending=True)
    probs = F.softmax(sorted_logits, dim=-1)
    cumul = torch.cumsum(probs, dim=-1)
    mask = (cumul - probs) >= p 
    mask[..., 0] = False
    sorted_logits[mask] = float('-inf')
    
    masked_l = torch.empty_like(logits)
    masked_l.scatter_(dim=-1, index=sorted_ids, src=sorted_logits)
    
    probs = F.softmax(masked_l, dim=-1)
    token = torch.multinomial(probs, num_samples = 1)
    return token 
       
 
 
# tok = AutoTokenizer.from_pretrained("gpt2")
# model = AutoModelForCausalLM.from_pretrained("gpt2").eval().to("mps")
# prompt = "The quick brown fox"
# input_ids = tok(prompt, return_tensors="pt").input_ids.to("mps")

# greedy = lambda logits: sample_topp(logits)
# generated = generate(model, input_ids, max_new_tokens=20, sampling_fn = greedy, eos_token_id=tok.eos_token_id)
# print(tok.decode(generated['generated_ids'][0], skip_special_tokens=True))
# print("tokens per second: ", generated['tokens_per_sec'])