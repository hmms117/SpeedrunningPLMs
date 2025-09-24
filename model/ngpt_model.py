import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from dataclasses import dataclass
from transformers import EsmTokenizer

from model.utils import norm


class NormLinear(nn.Linear):
    """Linear layer with weight normalization on the input dimension."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = F.normalize(self.weight, dim=1)
        return F.linear(x, weight, self.bias)


class Residual(nn.Module):
    """Simple residual wrapper with learnable scale."""
    def __init__(self, module: nn.Module, scale: float = 1.0):
        super().__init__()
        self.module = module
        self.scale = nn.Parameter(torch.tensor(scale, dtype=torch.float))

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return x + self.scale * self.module(x, *args, **kwargs)


def l2norm(t: torch.Tensor, dim: int = -1, eps: float = 1e-5) -> torch.Tensor:
    return F.normalize(t, dim=dim, eps=eps)


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int, dim_head: int, causal: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.dim_head = dim_head
        inner = dim_head * num_heads

        self.to_q = NormLinear(dim, inner, bias=False)
        self.to_k = NormLinear(dim, inner, bias=False)
        self.to_v = NormLinear(dim, inner, bias=False)
        self.to_out = NormLinear(inner, dim, bias=False)
        self.causal = causal

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, n, _ = x.shape
        q = self.to_q(x).view(b, n, self.num_heads, self.dim_head).transpose(1, 2)
        k = self.to_k(x).view(b, n, self.num_heads, self.dim_head).transpose(1, 2)
        v = self.to_v(x).view(b, n, self.num_heads, self.dim_head).transpose(1, 2)

        q = l2norm(q)
        k = l2norm(k)

        attn = torch.einsum('bhid,bhjd->bhij', q, k) / (self.dim_head ** 0.5)
        if mask is not None:
            mask = mask[:, None, None, :]
            attn = attn.masked_fill(~mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = out.transpose(1, 2).contiguous().view(b, n, self.num_heads * self.dim_head)
        return self.to_out(out)


class FeedForward(nn.Module):
    def __init__(self, dim: int, expansion: int = 4):
        super().__init__()
        hidden = dim * expansion
        self.net = nn.Sequential(
            NormLinear(dim, hidden, bias=False),
            nn.GELU(),
            NormLinear(hidden, dim, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, dim_head: int):
        super().__init__()
        self.attn = Residual(Attention(dim, num_heads, dim_head))
        self.ff = Residual(FeedForward(dim))

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.attn(x, mask)
        x = self.ff(x)
        return x


def subsequent_mask(n: int) -> torch.Tensor:
    return torch.tril(torch.ones(n, n, dtype=torch.bool))


@dataclass
class NGPTConfig:
    vocab_size: int = 33
    hidden_size: int = 512
    num_layers: int = 8
    num_heads: int = 8
    dim_head: int = 64
    num_register_tokens: int = 4


class NGPT(nn.Module):
    def __init__(self, config: NGPTConfig):
        super().__init__()
        self.config = config
        self.tokenizer = EsmTokenizer.from_pretrained('facebook/esm2_t6_8M_UR50D')
        self.register_tokens = (
            nn.Parameter(torch.zeros(1, config.num_register_tokens, config.hidden_size))
            if config.num_register_tokens > 0 else None
        )

        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.blocks = nn.ModuleList([
            Block(config.hidden_size, config.num_heads, config.dim_head)
            for _ in range(config.num_layers)
        ])
        self.ln = nn.LayerNorm(config.hidden_size)
        self.to_logits = NormLinear(config.hidden_size, config.vocab_size, bias=False)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids: torch.Tensor, return_loss: bool = False):
        b, n = input_ids.shape
        x = self.embed(input_ids)
        if self.register_tokens is not None:
            reg = self.register_tokens.expand(b, -1, -1)
            x = torch.cat((reg, x), dim=1)
            mask = subsequent_mask(n + self.config.num_register_tokens).to(x.device)
        else:
            mask = subsequent_mask(n).to(x.device)
        for block in self.blocks:
            x = block(x, mask)
        x = self.ln(x)
        logits = self.to_logits(x)
        if self.register_tokens is not None:
            logits = logits[:, self.config.num_register_tokens:]
        if not return_loss:
            return logits
        labels = input_ids
        loss = self.loss_fn(logits.view(-1, self.config.vocab_size), labels.view(-1))
        return loss
