"""
Conditional autoregressive transformer prior over VQ-VAE discrete tokens.
GPT-style model conditioned on class, action, direction tokens.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.tril(torch.ones(1, 1, 512, 512)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        y = attn @ v
        y = y.transpose(1, 2).contiguous().reshape(B, T, C)
        y = self.proj(y)
        return self.dropout(y)


class MLP(nn.Module):
    def __init__(self, d_model: int, expansion: int = 4, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, expansion * d_model)
        self.fc2 = nn.Linear(expansion * d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(F.gelu(self.fc1(x))))


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class SpriteTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int = 256,
        condition_vocab_size: int = 64,
        d_model: int = 256,
        n_layers: int = 8,
        n_heads: int = 4,
        max_seq_len: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_len, d_model))

        # Condition embeddings (class, action, direction)
        self.class_embedding = nn.Embedding(condition_vocab_size, d_model)
        self.action_embedding = nn.Embedding(condition_vocab_size, d_model)
        self.direction_embedding = nn.Embedding(condition_vocab_size, d_model)

        # Condition projection
        self.cond_proj = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.ReLU(),
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, dropout) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)

        # Output head
        self.head = nn.Linear(d_model, vocab_size)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.zeros_(module.bias)
                nn.init.ones_(module.weight)

    def forward(
        self,
        token_indices: torch.Tensor,
        class_ids: torch.Tensor,
        action_ids: torch.Tensor,
        direction_ids: torch.Tensor,
    ) -> torch.Tensor:
        B, T = token_indices.shape

        # Token embeddings + position
        tok_emb = self.token_embedding(token_indices)
        pos_emb = self.pos_embedding[:, :T, :]
        x = tok_emb + pos_emb

        # Condition embeddings (prepended as prefix)
        c_emb = self.cond_proj(
            torch.cat([
                self.class_embedding(class_ids),
                self.action_embedding(action_ids),
                self.direction_embedding(direction_ids),
            ], dim=-1)
        ).unsqueeze(1)  # (B, 1, d_model)

        x = torch.cat([c_emb, x[:, :-1, :]], dim=1)

        # Transformer
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)

        logits = self.head(x)
        return logits

    def generate(
        self,
        class_ids: torch.Tensor,
        action_ids: torch.Tensor,
        direction_ids: torch.Tensor,
        max_tokens: int = 64,
        temperature: float = 1.0,
        top_k: int = 40,
        top_p: float = 0.9,
    ) -> torch.Tensor:
        self.eval()
        B = class_ids.shape[0]
        tokens = torch.zeros(B, 0, dtype=torch.long, device=next(self.parameters()).device)

        with torch.no_grad():
            for _ in range(max_tokens):
                logits = self.forward(tokens, class_ids, action_ids, direction_ids)
                next_logits = logits[:, -1, :] / temperature

                # Top-k filtering
                if top_k > 0:
                    top_k_vals, _ = torch.topk(next_logits, top_k, dim=-1)
                    next_logits[next_logits < top_k_vals[:, -1:]] = float("-inf")

                # Top-p filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_logits, descending=True, dim=-1)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_logits[cumulative_probs > top_p] = float("-inf")
                    sorted_logits[:, 0] = next_logits.gather(1, sorted_indices[:, 0:1])
                    next_logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)

                probs = F.softmax(next_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1)
                tokens = torch.cat([tokens, next_tokens], dim=1)

        return tokens
