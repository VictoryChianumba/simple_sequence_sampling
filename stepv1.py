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
        