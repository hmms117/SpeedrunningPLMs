import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional

from transformers import EsmTokenizer, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import ModelOutput
from torch.nn.attention.flex_attention import create_block_mask

from model.attention import SelfAttention, MultiHeadPAttention
from model.utils import norm, MLP
from masker import ProteinMasker


@dataclass
class PoetConfig(PretrainedConfig):
    """Configuration class for the PoET decoder."""

    hidden_size: int = 512
    num_attention_heads: int = 8
    num_hidden_layers: int = 12
    num_att_tokens: int = 512
    vocab_size: int = 33
    expansion_ratio: float = 2.0
    dropout: float = 0.1
    soft_logit_cap: float = 16.0
    sliding_window_size: int = 2048
    p_attention: bool = False
    tie_embeddings: bool = False
    mlm_probability: float = 0.5


class LMHead(nn.Module):
    """Simple language modeling head used for both AR and MLM."""

    def __init__(self, hidden_size: int, vocab_size: int, soft_logit_cap: float = 30.0):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.decoder = nn.Linear(hidden_size, vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(vocab_size))
        self.soft_logit_cap = soft_logit_cap
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dense(norm(x))
        x = self.act(x)
        x = self.decoder(x) + self.bias
        return self.soft_logit_cap * torch.tanh(x / self.soft_logit_cap)


class TransformerBlock(nn.Module):
    def __init__(self, config: PoetConfig):
        super().__init__()
        if config.p_attention:
            self.attn = MultiHeadPAttention(
                config.hidden_size,
                config.num_attention_heads,
                config.num_att_tokens,
                config.sliding_window_size,
            )
        else:
            self.attn = SelfAttention(config.hidden_size, config.num_attention_heads)
        self.mlp = MLP(config.hidden_size, config.expansion_ratio)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(norm(x), attention_mask)
        x = x + self.mlp(norm(x))
        return x


class Transformer(nn.Module):
    def __init__(self, config: PoetConfig):
        super().__init__()
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_hidden_layers)])

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, attention_mask)
        return x


class PoetDecoder(PreTrainedModel):
    config_class = PoetConfig

    def __init__(self, config: PoetConfig):
        super().__init__(config)
        self.config = config
        self.tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
        self.masker = ProteinMasker(self.tokenizer, mask_rate=0.15)

        self.cls_token_id = self.tokenizer.cls_token_id
        self.mask_token_id = self.tokenizer.mask_token_id

        self.vocab_size = config.vocab_size
        self.n_heads = config.num_attention_heads
        self.sliding_window_size = config.sliding_window_size

        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.transformer = Transformer(config)
        self.lm_head = LMHead(config.hidden_size, config.vocab_size, config.soft_logit_cap)
        if config.tie_embeddings:
            self.lm_head.decoder.weight = self.embedding.weight

        self.ce = nn.CrossEntropyLoss()

    # ------------------------------------------------------------------
    # Attention masks
    # ------------------------------------------------------------------
    def _build_attention_mask(self, input_ids: torch.Tensor, causal: bool) -> torch.Tensor:
        docs = (input_ids == self.cls_token_id).cumsum(0)

        def mask_fn(b, h, q_idx, kv_idx):
            if causal:
                within_window = (q_idx - kv_idx) >= 0 and (q_idx - kv_idx) < self.sliding_window_size
            else:
                within_window = abs(q_idx - kv_idx) < self.sliding_window_size
            same_doc = docs[q_idx] == docs[kv_idx]
            return within_window and bool(same_doc)

        L = len(input_ids)
        mask = create_block_mask(
            mask_mod=mask_fn,
            B=1,
            H=self.n_heads,
            Q_LEN=L,
            KV_LEN=L,
            device=input_ids.device,
        )
        return mask

    def get_last_hidden_state(self, input_ids: torch.Tensor, causal: bool = False) -> torch.Tensor:
        attn_mask = self._build_attention_mask(input_ids, causal)
        x = self.embedding(input_ids)
        x = self.transformer(x, attn_mask)
        return x

    # ------------------------------------------------------------------
    # Autoregressive forward
    # ------------------------------------------------------------------
    def forward_ar(self, input_ids: torch.Tensor) -> ModelOutput:
        hidden = self.get_last_hidden_state(input_ids[:-1], causal=True)
        logits = self.lm_head(hidden)
        labels = input_ids[1:]
        loss = self.ce(logits.view(-1, self.vocab_size), labels.view(-1))
        return ModelOutput(loss=loss, logits=(logits, labels), last_hidden_state=hidden)

    # ------------------------------------------------------------------
    # Masked Language Modeling forward
    # ------------------------------------------------------------------
    def forward_mlm(self, input_ids: torch.Tensor) -> ModelOutput:
        masked_ids, labels = self.masker(input_ids.unsqueeze(0))
        masked_ids = masked_ids.squeeze(0)
        labels = labels.squeeze(0)
        hidden = self.get_last_hidden_state(masked_ids, causal=False)
        logits = self.lm_head(hidden)
        mask = labels != -100
        if mask.any():
            token_loss = self.ce(logits[mask].view(-1, self.vocab_size), labels[mask].view(-1))
        else:
            token_loss = torch.tensor(0.0, device=logits.device)
        return ModelOutput(loss=token_loss, logits=(logits, labels), last_hidden_state=hidden)

    # ------------------------------------------------------------------
    # Combined forward following PoET style mixture of AR and MLM
    # ------------------------------------------------------------------
    def forward(self, input_ids: torch.Tensor, task: Optional[str] = None) -> ModelOutput:
        """Run either AR or MLM objective.

        Args:
            input_ids: token ids including [CLS]/[EOS] delimiters.
            task: optional string 'ar' or 'mlm'. If None a random task is chosen.
        """
        if task is None:
            task = "mlm" if torch.rand(1).item() < self.config.mlm_probability else "ar"
        if task == "mlm":
            return self.forward_mlm(input_ids)
        else:
            return self.forward_ar(input_ids)

