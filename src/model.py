from torch import Tensor
import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin
from dataclasses import dataclass


@dataclass
class TransformerConfig:
    n_layer: int
    n_head: int
    n_kv_head: int
    hidden_dim: int
    intermediate_dim: int
    dropout: float
    vocab_size: int
    max_seq_len: int


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        x = x / torch.sqrt((x ** 2).mean(-1, keepdim=True) + self.eps)

        return self.scale * x
    

class CausalSelfAttention(nn.Module):
    def __init__(self, config: TransformerConfig):
        """Causal Self-Attention with support of
        Grouped-Query Attention and ALiBi for positional encoding
        """
        super().__init__()
        self.config = config
        assert self.config.hidden_dim % self.config.n_head == 0
        assert self.config.n_head % self.config.n_kv_head == 0
        self.head_dim = self.config.hidden_dim // self.config.n_head
        self.scale = self.head_dim**-0.5
        self.q_per_kv = self.config.n_head // self.config.n_kv_head

        # Init projection layers
        self.q_proj = nn.Linear(self.config.hidden_dim, self.config.hidden_dim, bias=False)
        self.kv_proj = nn.Linear(self.config.hidden_dim, 2 * self.config.hidden_dim // self.q_per_kv, bias=False)
        self.out_proj = nn.Linear(self.config.hidden_dim, self.config.hidden_dim, bias=False)

        self.attn_dropout = nn.Dropout(self.config.dropout)

        self.register_buffer("causal_mask", self._create_causal_mask(self.config.max_seq_len))
        self.register_buffer("alibi", self._build_alibi_bias(self.config.n_head))

    def _build_alibi_bias(self, num_heads: int) -> Tensor:
        bias = torch.arange(num_heads) + 1
        bias = 2 ** ((-8 / num_heads) * bias)

        return bias[None, :, None, None]

    def _create_causal_mask(self, max_seq_len: int) -> Tensor:
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len))

        return mask[None, None, ...]

    def forward(self, x: Tensor, attention_mask: Tensor = None) -> Tensor:
        B, L, H = x.shape
        q_proj = self.q_proj(x).reshape(B, L, self.config.n_kv_head, self.q_per_kv, self.head_dim) # B * L * nkv_head * q_per_kv, h
        kv_proj = self.kv_proj(x).reshape(B, L, self.config.n_kv_head, 2, self.head_dim) # B * L * nkv_head * 2 * h
        k_proj = kv_proj[:, :, :, 0, :] # B * L * nkv_head * h
        v_proj = kv_proj[:, :, :, 1, :] # B * L * nkv_head * h



        qk = self.scale *  (q_proj.permute(0, 2, 3, 1, 4) @ k_proj.permute(0, 2, 3, 1)[:, :, None, ...]) # B * nkv_head * q_per_kv * L * L
        qk = qk.reshape(B, self.config.n_head, L, L) # B * n_head * L * L


        alibi = self.alibi * torch.tril((torch.arange(L)[None, :] - torch.arange(L)[:, None])).to(qk.device)
        qk = qk + alibi
        qk = qk.masked_fill(self.causal_mask[:, :, :L, :L] == 0, -float('inf'))

        if attention_mask is not None:
            qk = qk.masked_fill(attention_mask[:, None, None, :] == 0, -float('inf'))

        score = self.attn_dropout(torch.softmax(qk, dim=-1)).reshape(B, self.config.n_kv_head, self.q_per_kv, L, L)

        result = (score @ v_proj.permute(0, 2, 1, 3)[:, :, None, ...]) # B * self.config.n_kv_head *  self.q_per_kv * L * h
        result = result.reshape(B, self.config.n_head, L, self.head_dim) # B * n_head * L * head_dim]
        result = self.out_proj(result.permute(0, 2, 1, 3).reshape(B, L, H)) # B * L * H

        return result


class SwiGLU(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        # Init up- and down- projection layers
        self.fc1 = nn.Linear(config.hidden_dim, config.intermediate_dim)
        self.fc2 = nn.Linear(config.intermediate_dim, config.hidden_dim)
        self.silu = nn.SiLU()

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.fc2(self.silu(x) * x)

        return x


class Block(nn.Module):
    def __init__(self, config: TransformerConfig):
        """Base Transformer Block
        - Causal Self-Attention and SwiGLU as main elements
        - Pre-normalization via RMSNorm
        - Regularization with dropouts before residuals
        """
        super().__init__()
        self.ln_1 = RMSNorm(config.hidden_dim)
        self.res_dropout_1 = nn.Dropout(config.dropout)
        self.attn = CausalSelfAttention(config) #torch.nn.MultiheadAttention(embed_dim=config.hidden_dim, num_heads=config.n_head, batch_first=True) #CausalSelfAttention(config)

        self.ln_2 = RMSNorm(config.hidden_dim)
        self.res_dropout_2 = nn.Dropout(config.dropout)
        self.mlp = SwiGLU(config)

    def forward(self, x: Tensor, attention_mask: Tensor = None) -> Tensor:
        causal_mask = torch.tril(torch.ones(x.shape[1], x.shape[1])).to(x.device)
        x = self.ln_1(x)
        x = x + self.res_dropout_1(self.attn(x))
        x = x + self.res_dropout_2(self.mlp(self.ln_2(x)))

        return x

class TransformerForCausalLM(nn.Module, PyTorchModelHubMixin):
    def __init__(self, config: TransformerConfig):
        """Transformer model for Language Modeling"""
        super().__init__()
        self.vocab_size = config.vocab_size
        self.max_seq_len = config.max_seq_len
        self.n_layer = config.n_layer
        self.n_head = config.n_head
        self.hidden_dim = config.hidden_dim
        self.dropout = config.dropout

        self.token_emb = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.emb_dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_final = RMSNorm(config.hidden_dim)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

        self.apply(self._init_weights)

        n_params = sum(p.numel() for p in self.parameters())
        print(f"Number of parameters: {n_params / 1e6:.2f}M")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, RMSNorm):
            torch.nn.init.ones_(module.scale)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None) -> Tensor:
        emb = self.token_emb(input_ids)
        x = self.emb_dropout(emb)
        for layer in self.layers:
            x = layer(x, attention_mask)

        x = self.lm_head(self.ln_final(x))

        return x

    @torch.inference_mode()
    def generate(
        self, idx: Tensor, max_new_tokens, eos_token_id, temperature=1.0, do_sample=False, top_k=None
    ) -> Tensor:
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.shape[1] <= self.max_seq_len else idx[:, -self.max_seq_len :]
            logits = self(idx_cond)

            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                mask = logits < torch.topk(logits, k=top_k, dim=-1).values[:, -1][:, None]
                logits[mask] = -float("inf")


            probs = torch.softmax(logits, dim=-1)
            if do_sample:
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                idx_next = torch.argmax(probs, dim=-1, keepdim=True)

            idx = torch.cat((idx, idx_next), dim=1)
            if idx_next == eos_token_id:
                break
        return idx
