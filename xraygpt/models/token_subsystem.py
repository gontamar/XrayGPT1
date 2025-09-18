import torch
from transformers import LlamaTokenizer

class TokenSubsystem:
    def __init__(self, llama_model_path):
        # Use external tokenization logic from XrayGPT inference pipeline
        self.tokenizer = LlamaTokenizer.from_pretrained(llama_model_path, use_fast=False)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Expose tokenizer attributes for compatibility with existing code
        self.pad_token = self.tokenizer.pad_token
        self.eos_token = self.tokenizer.eos_token
        self.bos_token = self.tokenizer.bos_token
        self.pad_token_id = self.tokenizer.pad_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        self.bos_token_id = self.tokenizer.bos_token_id
    
    @property
    def padding_side(self):
        """Get padding side from underlying tokenizer"""
        return self.tokenizer.padding_side
    
    @padding_side.setter
    def padding_side(self, value):
        """Set padding side on underlying tokenizer"""
        self.tokenizer.padding_side = value
    
    def __call__(self, *args, **kwargs):
        """Make TokenSubsystem callable like the original tokenizer"""
        return self.tokenizer(*args, **kwargs)
    
    def decode(self, *args, **kwargs):
        """Expose decode method for compatibility"""
        return self.tokenizer.decode(*args, **kwargs)
    
    def encode(self, *args, **kwargs):
        """Expose encode method for compatibility"""
        return self.tokenizer.encode(*args, **kwargs)
    
    def text_to_tokens(self, text, add_bos=True, max_length=128):
        """
        Convert list of text strings into token IDs + attention masks.
        Uses external tokenization logic from XrayGPT inference pipeline.
        """
        tokens = self.tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=max_length,
            add_special_tokens=False
        )
        if add_bos:
            bos = torch.full(
                (len(text), 1),
                self.tokenizer.bos_token_id,
                dtype=tokens.input_ids.dtype
            )
            tokens.input_ids = torch.cat([bos, tokens.input_ids], dim=1)
            tokens.attention_mask = torch.cat(
                [torch.ones_like(bos), tokens.attention_mask], dim=1
             )
        return tokens           
