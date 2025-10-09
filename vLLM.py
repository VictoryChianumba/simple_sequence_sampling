import torch  

class BlockManager:
    def __init__(self, num_blocks, block_size, num_layers, num_heads, head_dim, device, dtype):
        """
        Manages physical memory blocks for KV cache.
        
        Args:
            num_blocks: Total number of blocks in memory pool
            block_size: Number of tokens per block
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            head_dim: Dimension per head
        """
        self.num_blocks = num_blocks
        self.block_size = block_size
        
        # Physical memory: separate for keys and values, per layer
        # Shape: [num_layers, num_blocks, block_size, num_heads, head_dim]
        self.key_blocks = torch.zeros(
            num_layers, num_blocks, block_size, num_heads, head_dim,
            device=device, dtype=dtype
        )
        self.value_blocks = torch.zeros(
            num_layers, num_blocks, block_size, num_heads, head_dim,
            device=device, dtype=dtype
        )
        
        # Free block management
        self.free_blocks = list(range(num_blocks))
        
        # Block tables: maps sequence_id -> list of block indices
        self.block_tables = {}
    
    def allocate_blocks(self, sequence_id, num_blocks):
        """
        Allocate blocks for a sequence.
        
        Args:
            sequence_id: Unique identifier for the sequence
            num_blocks: Number of blocks to allocate
            
        Returns:
            List of allocated block indices, or None if not enough blocks available
        """
        if len(self.free_blocks) < num_blocks:
            return None
        
        # Allocate blocks from the free list
        allocated_blocks = self.free_blocks[:num_blocks]
        self.free_blocks = self.free_blocks[num_blocks:]
        
        # Store the block table for this sequence
        self.block_tables[sequence_id] = allocated_blocks
        
        return allocated_blocks
    
    def free_blocks(self, sequence_id):
        """
        Free all blocks for a sequence.
        
        Args:
            sequence_id: Unique identifier for the sequence. 
        """
        if sequence_id not in self.block_tables:
            return
        
        # Return blocks to the free list
        allocated_blocks = self.block_tables[sequence_id]
        self.free_blocks.extend(allocated_blocks)
        
        # Remove the block table entry
        del self.block_tables[sequence_id]
    
    def get_num_free_blocks(self):
        """
        Check available memory.
        
        Returns:
            Number of free blocks
        """
        return len(self.free_blocks)
    
    def write_kv_cache(self, sequence_id, layer_idx, keys, values):
        """
        Write KV cache for a sequence into its allocated blocks.
        
        Args:
            sequence_id: Sequence identifier
            layer_idx: Transformer layer index
            keys: Keys tensor [1, num_heads, seq_len, head_dim]
            values: Values tensor [1, num_heads, seq_len, head_dim]
        """
        if sequence_id not in self.block_tables:
            raise ValueError(f"No blocks allocated for sequence {sequence_id}")
        
        block_indices = self.block_tables[sequence_id]
        seq_len = keys.shape[2]
        
        # Remove batch dimension and transpose to [seq_len, num_heads, head_dim]
        keys = keys.squeeze(0).transpose(0, 1)  # [seq_len, num_heads, head_dim]
        values = values.squeeze(0).transpose(0, 1)  # [seq_len, num_heads, head_dim]
        
        # Write keys and values to blocks
        for block_idx, physical_block in enumerate(block_indices):
            start_pos = block_idx * self.block_size
            end_pos = min(start_pos + self.block_size, seq_len)
            
            if start_pos >= seq_len:
                break  # No more tokens to write
            
            # Extract the chunk for this block
            keys_chunk = keys[start_pos:end_pos]  # [tokens_in_block, num_heads, head_dim]
            values_chunk = values[start_pos:end_pos]  # [tokens_in_block, num_heads, head_dim]
            
            # Write to physical storage
            self.key_blocks[layer_idx, physical_block, :len(keys_chunk)] = keys_chunk
            self.value_blocks[layer_idx, physical_block, :len(values_chunk)] = values_chunk

    def read_kv_cache(self, sequence_id, layer_idx):
        """
        Read KV cache for a sequence from its blocks.
        
        Args:
            sequence_id: Sequence identifier
            layer_idx: Transformer layer index
            
        Returns:
            keys: [1, num_heads, total_seq_len, head_dim]
            values: [1, num_heads, total_seq_len, head_dim]
        """
        if sequence_id not in self.block_tables:
            raise ValueError(f"No blocks allocated for sequence {sequence_id}")
        
        block_indices = self.block_tables[sequence_id]
        
        # Collect all chunks
        keys_chunks = []
        values_chunks = []
        
        for physical_block in block_indices:
            # Read from physical storage
            keys_chunk = self.key_blocks[layer_idx, physical_block]  # [block_size, num_heads, head_dim]
            values_chunk = self.value_blocks[layer_idx, physical_block]  # [block_size, num_heads, head_dim]
            
            keys_chunks.append(keys_chunk)
            values_chunks.append(values_chunk)
        
        # Concatenate all chunks
        keys = torch.cat(keys_chunks, dim=0)  # [total_seq_len, num_heads, head_dim]
        values = torch.cat(values_chunks, dim=0)  # [total_seq_len, num_heads, head_dim]
        
        # Transpose back to [num_heads, seq_len, head_dim] and add batch dimension
        keys = keys.transpose(0, 1).unsqueeze(0)  # [1, num_heads, seq_len, head_dim]
        values = values.transpose(0, 1).unsqueeze(0)  # [1, num_heads, seq_len, head_dim]
        
        return keys, values