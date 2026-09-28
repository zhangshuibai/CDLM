# -*- coding: utf-8 -*-
"""
Diffusion Transformer (DiT) for Sudoku
A lightweight 28.6M parameter model for discrete diffusion language modeling
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional


@dataclass
class DiTConfig:
    """Configuration for Diffusion Transformer."""
    vocab_size: int = 10  # 0: MASK, 1-9: digits (no EOL)
    seq_length: int = 81  # 81 cells (9×9 grid, no EOL tokens)
    hidden_dim: int = 512
    num_layers: int = 12
    num_heads: int = 8
    mlp_ratio: int = 4
    dropout: float = 0.1
    use_layer_norm: bool = True
    
    def __post_init__(self):
        assert self.hidden_dim % self.num_heads == 0, "hidden_dim must be divisible by num_heads"


class SinusoidalPositionEmbedding(nn.Module):
    """Sinusoidal position embeddings (not used, kept for reference)."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, positions):
        device = positions.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = positions[:, :, None] * emb[None, None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention with dropout."""
    def __init__(self, config: DiTConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_dim // config.num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(config.hidden_dim, 3 * config.hidden_dim, bias=True)
        self.proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x, attention_mask=None):
        B, N, C = x.shape
        
        # Compute Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        if attention_mask is not None:
            # attention_mask: (B, N) -> (B, 1, 1, N)
            attn_mask = attention_mask[:, None, None, :].expand(B, self.num_heads, N, N)
            attn = attn.masked_fill(~attn_mask, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Output
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.dropout(x)
        
        return x


class MLP(nn.Module):
    """Feed-forward network."""
    def __init__(self, config: DiTConfig):
        super().__init__()
        hidden_features = int(config.hidden_dim * config.mlp_ratio)
        self.fc1 = nn.Linear(config.hidden_dim, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer block with pre-norm."""
    def __init__(self, config: DiTConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_dim, eps=1e-6)
        self.attn = MultiHeadSelfAttention(config)
        self.norm2 = nn.LayerNorm(config.hidden_dim, eps=1e-6)
        self.mlp = MLP(config)
    
    def forward(self, x, attention_mask=None):
        # Pre-norm
        x = x + self.attn(self.norm1(x), attention_mask)
        x = x + self.mlp(self.norm2(x))
        return x


class DiffusionTransformer(nn.Module):
    """
    Diffusion Transformer for discrete tokens.
    Total parameters: ~28.6M
    """
    def __init__(self, config: DiTConfig):
        super().__init__()
        self.config = config
        
        # Embeddings
        self.token_embed = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, config.seq_length, config.hidden_dim))
        self.dropout = nn.Dropout(config.dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_layers)
        ])
        
        # Output
        self.norm = nn.LayerNorm(config.hidden_dim, eps=1e-6)
        self.head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Special init for output layer
        nn.init.normal_(self.head.weight, std=0.02)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(self, input_ids, attention_mask=None, is_causal=False):
        """
        Args:
            input_ids: (B, seq_len) token IDs
            attention_mask: (B, seq_len) attention mask (1=attend, 0=ignore)
            is_causal: whether to use causal attention (for compatibility, not used in diffusion)
        
        Returns:
            logits: (B, seq_len, vocab_size)
        """
        B, N = input_ids.shape
        
        # Token + position embeddings
        x = self.token_embed(input_ids)
        x = x + self.pos_embed[:, :N, :]
        x = self.dropout(x)
        
        # Transformer blocks (bidirectional attention for diffusion)
        for block in self.blocks:
            x = block(x, attention_mask)
        
        # Output
        x = self.norm(x)
        logits = self.head(x)
        
        return type('ModelOutput', (), {'logits': logits})()
    
    def count_parameters(self):
        """Count total and trainable parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'total': total, 'trainable': trainable}


def create_sudoku_dit(vocab_size=10, seq_length=81):
    """Create a DiT for Sudoku."""
    config = DiTConfig(
        vocab_size=vocab_size,
        seq_length=seq_length,
        hidden_dim=512,
        num_layers=12,
        num_heads=8,
        mlp_ratio=4,
        dropout=0.1,
    )
    model = DiffusionTransformer(config)
    params = model.count_parameters()
    print(f"Created DiT with {params['total']:,} parameters ({params['total']/1e6:.1f}M)")
    return model


if __name__ == '__main__':
    # Test model creation
    model = create_sudoku_dit()
    
    # Test forward pass
    batch_size = 4
    seq_length = 81
    input_ids = torch.randint(0, 10, (batch_size, seq_length))
    
    output = model(input_ids)
    print(f"Input shape: {input_ids.shape}")
    print(f"Output logits shape: {output.logits.shape}")
    
    # Count parameters by layer
    print("\nParameter breakdown:")
    total = 0
    for name, param in model.named_parameters():
        num_params = param.numel()
        total += num_params
        if 'blocks' not in name or 'blocks.0' in name:
            print(f"  {name}: {num_params:,}")
    print(f"Total: {total:,} ({total/1e6:.1f}M)")

