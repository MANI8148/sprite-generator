import torch
import pytest

from models.vqvae.model import (
    VQVAE, Encoder, Decoder, VectorQuantizer, ResidualBlock
)


class TestResidualBlock:
    def test_output_shape(self):
        block = ResidualBlock(64)
        x = torch.randn(2, 64, 8, 8)
        out = block(x)
        assert out.shape == x.shape

    def test_residual_connection(self):
        block = ResidualBlock(32)
        x = torch.randn(1, 32, 16, 16)
        out = block(x)
        assert out.shape == x.shape


class TestEncoder:
    def test_output_shape(self):
        encoder = Encoder(in_channels=4, hidden_dim=128, latent_dim=64)
        x = torch.randn(2, 4, 32, 32)
        out = encoder(x)
        assert out.shape == (2, 64, 8, 8)

    def test_different_batch_size(self):
        encoder = Encoder()
        x = torch.randn(4, 4, 32, 32)
        out = encoder(x)
        assert out.shape == (4, 64, 8, 8)


class TestDecoder:
    def test_output_shape(self):
        decoder = Decoder(out_channels=4, hidden_dim=128, latent_dim=64)
        z = torch.randn(2, 64, 8, 8)
        out = decoder(z)
        assert out.shape == (2, 4, 32, 32)

    def test_different_batch_size(self):
        decoder = Decoder()
        z = torch.randn(4, 64, 8, 8)
        out = decoder(z)
        assert out.shape == (4, 4, 32, 32)


class TestVectorQuantizer:
    def test_output_shapes(self):
        quantizer = VectorQuantizer(num_embeddings=256, embedding_dim=64)
        z = torch.randn(2, 64, 8, 8)
        quantized, vq_loss, indices = quantizer(z)
        assert quantized.shape == z.shape
        assert isinstance(vq_loss, torch.Tensor) and vq_loss.ndim == 0
        assert indices.shape == (2 * 8 * 8,)

    def test_codebook_entry(self):
        quantizer = VectorQuantizer(num_embeddings=256, embedding_dim=64)
        indices = torch.randint(0, 256, (16,))
        entries = quantizer.get_codebook_entry(indices)
        assert entries.shape == (16, 64)

    def test_indices_in_range(self):
        quantizer = VectorQuantizer(num_embeddings=256, embedding_dim=64)
        z = torch.randn(1, 64, 8, 8)
        _, _, indices = quantizer(z)
        assert indices.min() >= 0
        assert indices.max() < 256

    def test_commitment_cost(self):
        quantizer = VectorQuantizer(num_embeddings=128, embedding_dim=32, commitment_cost=0.5)
        z = torch.randn(1, 32, 4, 4)
        _, vq_loss, _ = quantizer(z)
        assert vq_loss.item() >= 0

    def test_ema_update_does_not_crash(self):
        quantizer = VectorQuantizer(num_embeddings=32, embedding_dim=16)
        z = torch.randn(1, 16, 4, 4)
        _, _, indices = quantizer(z)
        quantizer.ema_update(z, indices)
        assert quantizer.ema_count.shape[0] == 32
        assert quantizer.ema_sum.shape[1] == 32

    def test_ema_update_changes_codebook(self):
        quantizer = VectorQuantizer(num_embeddings=32, embedding_dim=16, decay=0.5)
        z = torch.randn(2, 16, 4, 4)
        _, _, indices = quantizer(z)
        old_weight = quantizer.embedding.weight.data.clone()
        quantizer.ema_update(z, indices)
        assert not torch.equal(quantizer.embedding.weight.data, old_weight), "EMA did not update codebook"

    def test_ema_training_mode_only_updates(self):
        quantizer = VectorQuantizer(num_embeddings=32, embedding_dim=16)
        z = torch.randn(1, 16, 4, 4)
        _, _, indices = quantizer(z)
        old_count = quantizer.ema_count.clone()
        old_sum = quantizer.ema_sum.clone()
        quantizer.eval()
        quantizer.ema_update(z, indices)
        assert torch.equal(quantizer.ema_count, old_count), "EMA should not update in eval mode"
        assert torch.equal(quantizer.ema_sum, old_sum), "EMA should not update in eval mode"

    def test_perplexity_in_range(self):
        quantizer = VectorQuantizer(num_embeddings=256, embedding_dim=64)
        indices = torch.randint(0, 256, (2, 64))
        ppl = quantizer.perplexity(indices)
        assert 1.0 <= ppl <= 256.0

    def test_perplexity_one_for_single_code(self):
        quantizer = VectorQuantizer(num_embeddings=256, embedding_dim=64)
        indices = torch.zeros(2, 64, dtype=torch.long)
        ppl = quantizer.perplexity(indices)
        assert ppl.item() == pytest.approx(1.0, abs=1e-4)

    def test_reset_dead_codes_reinitializes_unused(self):
        quantizer = VectorQuantizer(num_embeddings=8, embedding_dim=4)
        z = torch.randn(1, 4, 4, 4)
        _, _, indices = quantizer(z)
        old_weight = quantizer.embedding.weight.data.clone()
        n_reset = quantizer.reset_dead_codes(z, indices, threshold=0.0)
        assert n_reset > 0
        assert not torch.equal(quantizer.embedding.weight.data, old_weight)

    def test_reset_dead_codes_no_dead_all_used(self):
        quantizer = VectorQuantizer(num_embeddings=8, embedding_dim=4)
        z = torch.randn(32, 4, 4, 4)
        _, _, indices = quantizer(z)
        for _ in range(50):
            z = torch.randn(32, 4, 4, 4)
            _, _, indices = quantizer(z)
            usage = torch.bincount(indices.view(-1), minlength=8)
            if (usage > 0).sum().item() == 8:
                break
        usage = torch.bincount(indices.view(-1), minlength=8)
        assert (usage > 0).sum().item() == 8
        old_weight = quantizer.embedding.weight.data.clone()
        n_reset = quantizer.reset_dead_codes(z, indices, threshold=0.0)
        assert n_reset == 0
        assert torch.equal(quantizer.embedding.weight.data, old_weight)

    def test_reset_dead_codes_with_ema_buffers(self):
        quantizer = VectorQuantizer(num_embeddings=8, embedding_dim=4)
        quantizer.train()
        z = torch.randn(1, 4, 4, 4)
        _, _, indices = quantizer(z)
        quantizer.ema_update(z, indices)
        n_reset = quantizer.reset_dead_codes(z, indices, threshold=0.0)
        dead_counts = (quantizer.ema_count == 0).sum().item()
        assert dead_counts >= n_reset


class TestVQVAE:
    def test_forward_shapes(self):
        model = VQVAE()
        x = torch.randn(2, 4, 32, 32)
        out = model(x)
        assert out["recon"].shape == x.shape
        assert list(out["indices"].shape) == [2, 64]
        assert isinstance(out["loss"], torch.Tensor)
        assert isinstance(out["recon_loss"], torch.Tensor)
        assert isinstance(out["vq_loss"], torch.Tensor)

    def test_loss_non_negative(self):
        model = VQVAE()
        x = torch.randn(2, 4, 32, 32)
        out = model(x)
        assert out["loss"].item() >= 0
        assert out["recon_loss"].item() >= 0
        assert out["vq_loss"].item() >= 0

    def test_encode_to_indices(self):
        model = VQVAE()
        x = torch.randn(2, 4, 32, 32)
        indices = model.encode_to_indices(x)
        assert indices.shape == (2, 64)
        assert indices.dtype == torch.long

    def test_decode_from_indices(self):
        model = VQVAE()
        indices = torch.randint(0, 256, (2, 64))
        recon = model.decode_from_indices(indices, (64, 8, 8))
        assert recon.shape == (2, 4, 32, 32)

    def test_encode_decode_cycle(self):
        model = VQVAE()
        x = torch.randn(2, 4, 32, 32)
        indices = model.encode_to_indices(x)
        recon = model.decode_from_indices(indices, (64, 8, 8))
        assert recon.shape == x.shape

    def test_ema_update_on_vqvae(self):
        model = VQVAE(in_channels=4, hidden_dim=32, latent_dim=16, num_embeddings=32)
        x = torch.randn(2, 4, 32, 32)
        model.train()
        old_weight = model.quantizer.embedding.weight.data.clone()
        model.ema_update(x)
        assert not torch.equal(model.quantizer.embedding.weight.data, old_weight)

    def test_perplexity_on_vqvae(self):
        model = VQVAE()
        x = torch.randn(2, 4, 32, 32)
        ppl = model.perplexity(x)
        assert 1.0 <= ppl <= model.quantizer.num_embeddings

    def test_reset_dead_codes_on_vqvae(self):
        model = VQVAE(in_channels=4, hidden_dim=32, latent_dim=16, num_embeddings=32)
        x = torch.randn(1, 4, 32, 32)
        old_weight = model.quantizer.embedding.weight.data.clone()
        n_reset = model.reset_dead_codes(x, threshold=0.0)
        total = model.quantizer.num_embeddings
        assert 0 < n_reset <= total
        assert not torch.equal(model.quantizer.embedding.weight.data, old_weight)

    def test_gradient_flow(self):
        model = VQVAE()
        x = torch.randn(2, 4, 32, 32)
        out = model(x)
        loss = out["loss"]
        loss.backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f"Parameter {name} has no gradient"

    @pytest.fixture
    def small_model(self):
        return VQVAE(in_channels=4, hidden_dim=32, latent_dim=16, num_embeddings=32)

    def test_small_model_shapes(self, small_model):
        x = torch.randn(2, 4, 32, 32)
        out = small_model(x)
        assert out["recon"].shape == x.shape

    def test_overfit_constant_input(self):
        model = VQVAE(in_channels=4, hidden_dim=64, latent_dim=32, num_embeddings=64)
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
        x = torch.ones(4, 4, 32, 32)
        initial_loss = model(x)["loss"].item()
        for _ in range(100):
            optimizer.zero_grad()
            loss = model(x)["loss"]
            loss.backward()
            optimizer.step()
        final_loss = model(x)["loss"].item()
        assert final_loss < initial_loss, f"Loss did not decrease: {initial_loss:.4f} -> {final_loss:.4f}"
