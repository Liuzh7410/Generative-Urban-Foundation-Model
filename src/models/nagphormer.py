"""
NAGphormer model adapted from the reference implementation.

Expected input shape:

    [batch_size, hops + 1, input_dim]

Each row is one node's Hop2Token sequence:

    [x_v^0, x_v^1, ..., x_v^K]

where x_v^k is the k-hop aggregated feature vector for node v.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def init_params(module: nn.Module, n_layers: int) -> None:
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(n_layers))
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)


class NAGphormer(nn.Module):
    def __init__(
        self,
        hops: int,
        n_class: int,
        input_dim: int,
        n_layers: int = 6,
        num_heads: int = 8,
        hidden_dim: int = 64,
        dropout_rate: float = 0.0,
        attention_dropout_rate: float = 0.1,
    ):
        super().__init__()

        self.seq_len = hops + 1
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.ffn_dim = 2 * hidden_dim
        self.num_heads = num_heads
        self.n_layers = n_layers
        self.n_class = n_class

        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [
                EncoderLayer(
                    hidden_size=hidden_dim,
                    ffn_size=self.ffn_dim,
                    dropout_rate=dropout_rate,
                    attention_dropout_rate=attention_dropout_rate,
                    num_heads=num_heads,
                )
                for _ in range(n_layers)
            ]
        )
        self.final_ln = nn.LayerNorm(hidden_dim)

        self.out_proj = nn.Linear(hidden_dim, hidden_dim // 2)
        self.readout_attention = nn.Linear(2 * hidden_dim, 1)
        self.classifier = nn.Linear(hidden_dim // 2, n_class)

        self.apply(lambda module: init_params(module, n_layers=n_layers))

    def forward(self, hop_tokens: torch.Tensor) -> torch.Tensor:
        if hop_tokens.dim() != 3:
            raise ValueError(
                "NAGphormer expects [batch_size, hops + 1, input_dim], "
                f"got shape {tuple(hop_tokens.shape)}"
            )
        if hop_tokens.size(1) != self.seq_len:
            raise ValueError(
                f"Expected sequence length {self.seq_len}, got {hop_tokens.size(1)}"
            )

        hidden = self.input_projection(hop_tokens)
        for layer in self.layers:
            hidden = layer(hidden)
        hidden = self.final_ln(hidden)

        node_token = hidden[:, :1, :]
        neighbor_tokens = hidden[:, 1:, :]

        target = node_token.repeat(1, self.seq_len - 1, 1)
        hop_weights = self.readout_attention(torch.cat((target, neighbor_tokens), dim=2))
        hop_weights = F.softmax(hop_weights, dim=1)

        neighbor_summary = torch.sum(neighbor_tokens * hop_weights, dim=1, keepdim=True)
        output = (node_token + neighbor_summary).squeeze(1)
        logits = self.classifier(torch.relu(self.out_proj(output)))
        return torch.log_softmax(logits, dim=1)


class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size: int, ffn_size: int, dropout_rate: float):
        super().__init__()
        self.layer1 = nn.Linear(hidden_size, ffn_size)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(ffn_size, hidden_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.layer2(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size: int, attention_dropout_rate: float, num_heads: int):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")

        self.num_heads = num_heads
        self.att_size = hidden_size // num_heads
        self.scale = self.att_size**-0.5

        self.linear_q = nn.Linear(hidden_size, num_heads * self.att_size)
        self.linear_k = nn.Linear(hidden_size, num_heads * self.att_size)
        self.linear_v = nn.Linear(hidden_size, num_heads * self.att_size)
        self.att_dropout = nn.Dropout(attention_dropout_rate)
        self.output_layer = nn.Linear(num_heads * self.att_size, hidden_size)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        original_size = q.size()
        batch_size = q.size(0)

        q = self.linear_q(q).view(batch_size, -1, self.num_heads, self.att_size)
        k = self.linear_k(k).view(batch_size, -1, self.num_heads, self.att_size)
        v = self.linear_v(v).view(batch_size, -1, self.num_heads, self.att_size)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2).transpose(2, 3)
        v = v.transpose(1, 2)

        scores = torch.matmul(q * self.scale, k)
        if attn_bias is not None:
            scores = scores + attn_bias

        attention = torch.softmax(scores, dim=3)
        attention = self.att_dropout(attention)
        x = attention.matmul(v)

        x = x.transpose(1, 2).contiguous()
        x = x.view(batch_size, -1, self.num_heads * self.att_size)
        x = self.output_layer(x)

        assert x.size() == original_size
        return x


class EncoderLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        ffn_size: int,
        dropout_rate: float,
        attention_dropout_rate: float,
        num_heads: int,
    ):
        super().__init__()
        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.self_attention = MultiHeadAttention(
            hidden_size=hidden_size,
            attention_dropout_rate=attention_dropout_rate,
            num_heads=num_heads,
        )
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor, attn_bias: torch.Tensor | None = None) -> torch.Tensor:
        y = self.self_attention_norm(x)
        y = self.self_attention(y, y, y, attn_bias)
        y = self.self_attention_dropout(y)
        x = x + y

        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y
        return x
