import torch
import pytest

from models.transformer.model import (
    SpriteTransformer, CausalSelfAttention, MLP, TransformerBlock
)


class TestCausalSelfAttention:
    def test_output_shape(self):
        attn = CausalSelfAttention(d_model=256, n_heads=4)
        x = torch.randn(2, 10, 256)
        out = attn(x)
        assert out.shape == x.shape

    def test_causal_mask(self):
        attn = CausalSelfAttention(d_model=64, n_heads=2)
        x = torch.randn(1, 16, 64)
        out = attn(x)
        assert out.shape == x.shape

    def test_mask_stops_lookahead(self):
        d_model = 32
        n_heads = 2
        attn = CausalSelfAttention(d_model, n_heads)
        x = torch.randn(1, 5, d_model)
        qkv = attn.qkv(x).reshape(1, 5, 3, n_heads, d_model // n_heads)
        q, k, v = qkv.unbind(2)
        q, k = q.transpose(1, 2), k.transpose(1, 2)
        attn_matrix = (q @ k.transpose(-2, -1)) * ((d_model // n_heads) ** -0.5)
        masked = attn_matrix.masked_fill(attn.mask[:, :, :5, :5] == 0, float("-inf"))
        assert torch.isinf(masked[0, 0, 0, 1:]).all()
        assert not torch.isinf(masked[0, 0, 0, 0])


class TestMLP:
    def test_output_shape(self):
        mlp = MLP(d_model=256)
        x = torch.randn(2, 10, 256)
        out = mlp(x)
        assert out.shape == x.shape

    def test_different_expansion(self):
        mlp = MLP(d_model=128, expansion=8)
        x = torch.randn(2, 5, 128)
        out = mlp(x)
        assert out.shape == x.shape


class TestTransformerBlock:
    def test_output_shape(self):
        block = TransformerBlock(d_model=256, n_heads=4)
        x = torch.randn(2, 10, 256)
        out = block(x)
        assert out.shape == x.shape

    def test_does_not_change_batch_or_seq_len(self):
        block = TransformerBlock(d_model=128, n_heads=2)
        x = torch.randn(4, 15, 128)
        out = block(x)
        assert out.shape == x.shape


class TestSpriteTransformer:
    @pytest.fixture
    def model(self):
        return SpriteTransformer(
            vocab_size=256,
            condition_vocab_size=64,
            d_model=64,
            n_layers=2,
            n_heads=2,
            max_seq_len=32,
        )

    def test_forward_shapes(self, model):
        B, T = 2, 16
        token_indices = torch.randint(0, 256, (B, T))
        class_ids = torch.randint(0, 10, (B,))
        action_ids = torch.randint(0, 10, (B,))
        direction_ids = torch.randint(0, 8, (B,))
        logits = model(token_indices, class_ids, action_ids, direction_ids)
        assert logits.shape == (B, T, 256)

    def test_condition_embedding_shapes(self, model):
        B, T = 2, 16
        token_indices = torch.randint(0, 256, (B, T))
        class_ids = torch.tensor([1, 2])
        action_ids = torch.tensor([3, 4])
        direction_ids = torch.tensor([5, 6])
        logits = model(token_indices, class_ids, action_ids, direction_ids)
        assert logits.shape == (B, T, 256)

    def test_generate_output_shape(self, model):
        B = 2
        class_ids = torch.zeros(B, dtype=torch.long)
        action_ids = torch.zeros(B, dtype=torch.long)
        direction_ids = torch.zeros(B, dtype=torch.long)
        tokens = model.generate(
            class_ids, action_ids, direction_ids,
            max_tokens=8, temperature=1.0, top_k=0, top_p=1.0,
        )
        assert tokens.shape == (B, 8)
        assert tokens.dtype == torch.long

    def test_generate_without_top_k(self, model):
        B = 1
        class_ids = torch.zeros(B, dtype=torch.long)
        action_ids = torch.zeros(B, dtype=torch.long)
        direction_ids = torch.zeros(B, dtype=torch.long)
        tokens = model.generate(
            class_ids, action_ids, direction_ids,
            max_tokens=4, temperature=0.5, top_k=0, top_p=1.0,
        )
        assert tokens.shape == (B, 4)

    def test_gradient_flow(self, model):
        B, T = 2, 16
        token_indices = torch.randint(0, 256, (B, T))
        class_ids = torch.randint(0, 10, (B,))
        action_ids = torch.randint(0, 10, (B,))
        direction_ids = torch.randint(0, 8, (B,))
        logits = model(token_indices, class_ids, action_ids, direction_ids)
        loss = torch.nn.functional.cross_entropy(logits.view(-1, 256), token_indices.view(-1))
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Parameter {name} has no gradient"

    def test_overfit_single_batch(self, model):
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        B, T = 2, 16
        token_indices = torch.randint(0, 256, (B, T))
        class_ids = torch.randint(0, 10, (B,))
        action_ids = torch.randint(0, 10, (B,))
        direction_ids = torch.randint(0, 8, (B,))

        logits = model(token_indices, class_ids, action_ids, direction_ids)
        initial_loss = torch.nn.functional.cross_entropy(
            logits.view(-1, 256), token_indices.view(-1)
        ).item()

        for _ in range(50):
            optimizer.zero_grad()
            logits = model(token_indices, class_ids, action_ids, direction_ids)
            loss = torch.nn.functional.cross_entropy(logits.view(-1, 256), token_indices.view(-1))
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            logits = model(token_indices, class_ids, action_ids, direction_ids)
            final_loss = torch.nn.functional.cross_entropy(
                logits.view(-1, 256), token_indices.view(-1)
            ).item()
        assert final_loss < initial_loss, f"Loss did not decrease: {initial_loss:.4f} -> {final_loss:.4f}"

    def test_different_conditions_produce_different_logits(self, model):
        B, T = 2, 16
        token_indices = torch.randint(0, 256, (B, T))

        class_ids_1 = torch.zeros(B, dtype=torch.long)
        class_ids_2 = torch.ones(B, dtype=torch.long)
        action_ids = torch.zeros(B, dtype=torch.long)
        direction_ids = torch.zeros(B, dtype=torch.long)

        logits_1 = model(token_indices, class_ids_1, action_ids, direction_ids)
        logits_2 = model(token_indices, class_ids_2, action_ids, direction_ids)
        assert not torch.allclose(logits_1, logits_2)
