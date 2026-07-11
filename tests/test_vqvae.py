import torch
import pytest

from models.vqvae.model import (
    ImprovedVQVAE, VQVAE, Encoder, Decoder, VectorQuantizerEMA, VectorQuantizer, ResBlock,
    SelfAttention2d, PatchDiscriminator,
)


class TestResBlock:
    def test_output_shape(self):
        block = ResBlock(64)
        x = torch.randn(2, 64, 8, 8)
        out = block(x)
        assert out.shape == x.shape


class TestEncoder:
    def test_output_shape(self):
        encoder = Encoder(in_channels=4, hidden_dim=128, latent_dim=64)
        x = torch.randn(2, 4, 32, 32)
        z, skips = encoder(x)
        assert z.shape == (2, 64, 8, 8)
        assert len(skips) == 3

    def test_different_batch_size(self):
        encoder = Encoder()
        x = torch.randn(4, 4, 32, 32)
        z, skips = encoder(x)
        assert z.shape == (4, 96, 8, 8)


class TestDecoder:
    def test_output_shape(self):
        decoder = Decoder(out_channels=4, hidden_dim=128, latent_dim=64)
        z = torch.randn(2, 64, 8, 8)
        out = decoder(z, None, None, None)
        assert out.shape == (2, 4, 32, 32)


class TestVectorQuantizerEMA:
    def test_output_shapes(self):
        quantizer = VectorQuantizerEMA(num_embeddings=256, embedding_dim=64)
        z = torch.randn(2, 64, 8, 8)
        quantized, vq_loss, indices = quantizer(z)
        assert quantized.shape == z.shape
        assert isinstance(vq_loss, torch.Tensor) and vq_loss.ndim == 0
        assert indices.shape == (2 * 8 * 8,)

    def test_codebook_entry(self):
        quantizer = VectorQuantizerEMA(num_embeddings=256, embedding_dim=64)
        indices = torch.randint(0, 256, (16,))
        entries = quantizer.get_codebook_entry(indices)
        assert entries.shape == (16, 64)

    def test_indices_in_range(self):
        quantizer = VectorQuantizerEMA(num_embeddings=256, embedding_dim=64)
        z = torch.randn(1, 64, 8, 8)
        _, _, indices = quantizer(z)
        assert indices.min() >= 0
        assert indices.max() < 256

    def test_reset_dead_codes(self):
        quantizer = VectorQuantizerEMA(num_embeddings=64, embedding_dim=32)
        z = torch.randn(2, 32, 4, 4)
        quantizer(z)
        quantizer.reset_dead_codes(z)
        assert quantizer.ema_cluster_size.sum() > 0


class TestPatchDiscriminator:
    def test_output_shape(self):
        disc = PatchDiscriminator(in_channels=4, ch=32, n_layers=2)
        x = torch.randn(2, 4, 32, 32)
        out = disc(x)
        assert len(out.shape) == 4
        assert out.size(1) == 1

    def test_ema_update_does_not_crash(self):
        quantizer = VectorQuantizer(num_embeddings=32, embedding_dim=16)
        z = torch.randn(1, 16, 4, 4)
        _, _, indices = quantizer(z)
        quantizer.ema_update(z, indices)
        assert quantizer.ema_cluster_size.shape[0] == 32
        assert quantizer.ema_embedding.shape[1] == 16

    def test_ema_update_changes_codebook(self):
        quantizer = VectorQuantizer(num_embeddings=32, embedding_dim=16, decay=0.5)
        z = torch.randn(2, 16, 4, 4)
        _, _, indices = quantizer(z)
        old_weight = quantizer.embedding.data.clone()
        quantizer.ema_update(z, indices)
        assert not torch.equal(quantizer.embedding.data, old_weight), "EMA did not update codebook"

    def test_ema_training_mode_only_updates(self):
        quantizer = VectorQuantizer(num_embeddings=32, embedding_dim=16)
        z = torch.randn(1, 16, 4, 4)
        _, _, indices = quantizer(z)
        old_count = quantizer.ema_cluster_size.clone()
        old_sum = quantizer.ema_embedding.clone()
        quantizer.eval()
        quantizer.ema_update(z, indices)
        assert torch.equal(quantizer.ema_cluster_size, old_count), "EMA should not update in eval mode"
        assert torch.equal(quantizer.ema_embedding, old_sum), "EMA should not update in eval mode"

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
        quantizer = VectorQuantizer(num_embeddings=16, embedding_dim=8)
        z = torch.randn(1, 8, 2, 2)
        _, _, indices = quantizer(z)
        old_weight = quantizer.embedding.data.clone()
        n_reset = quantizer.reset_dead_codes(z, indices, threshold=0.0)
        assert n_reset > 0
        assert not torch.equal(quantizer.embedding.data, old_weight)

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
        old_weight = quantizer.embedding.data.clone()
        n_reset = quantizer.reset_dead_codes(z, indices, threshold=0.0)
        assert n_reset == 0
        assert torch.equal(quantizer.embedding.data, old_weight)

    def test_reset_dead_codes_with_ema_buffers(self):
        quantizer = VectorQuantizer(num_embeddings=16, embedding_dim=8)
        quantizer.train()
        z = torch.randn(1, 8, 2, 2)
        _, _, indices = quantizer(z)
        quantizer.ema_update(z, indices)
        usage = torch.bincount(indices.view(-1), minlength=16)
        dead_before = (usage == 0).sum().item()
        n_reset = quantizer.reset_dead_codes(z, indices, threshold=0.0)
        assert n_reset > 0
        assert n_reset <= dead_before


class TestVQVAE:
    def test_forward_shapes(self):
        model = ImprovedVQVAE(hidden_dim=64, latent_dim=32, num_embeddings=64)
        x = torch.randn(2, 4, 32, 32)
        out = model(x)
        assert out["recon"].shape == x.shape
        assert list(out["indices"].shape) == [2, 64]
        assert isinstance(out["loss"], torch.Tensor)
        assert isinstance(out["recon_loss"], torch.Tensor)
        assert isinstance(out["vq_loss"], torch.Tensor)

    def test_loss_non_negative(self):
        model = ImprovedVQVAE(hidden_dim=64, latent_dim=32, num_embeddings=64)
        x = torch.randn(2, 4, 32, 32)
        out = model(x)
        assert out["loss"].item() >= 0
        assert out["recon_loss"].item() >= 0
        assert out["vq_loss"].item() >= 0

    def test_encode_to_indices(self):
        model = ImprovedVQVAE(hidden_dim=64, latent_dim=32, num_embeddings=64)
        x = torch.randn(2, 4, 32, 32)
        indices = model.encode_to_indices(x)
        assert indices.shape == (2, 64)
        assert indices.dtype == torch.long

    def test_decode_from_indices(self):
        model = ImprovedVQVAE(hidden_dim=64, latent_dim=32, num_embeddings=64)
        indices = torch.randint(0, 64, (2, 64))
        recon = model.decode_from_indices(indices, (32, 8, 8))
        assert recon.shape == (2, 4, 32, 32)

    def test_encode_decode_cycle(self):
        model = ImprovedVQVAE(hidden_dim=64, latent_dim=32, num_embeddings=64)
        x = torch.randn(2, 4, 32, 32)
        indices = model.encode_to_indices(x)
        recon = model.decode_from_indices(indices, (32, 8, 8))
        assert recon.shape == x.shape

    def test_ema_update_on_vqvae(self):
        model = VQVAE(in_channels=4, hidden_dim=32, latent_dim=16, num_embeddings=32)
        x = torch.randn(2, 4, 32, 32)
        model.train()
        old_weight = model.quantizer.embedding.data.clone()
        model.ema_update(x)
        assert not torch.equal(model.quantizer.embedding.data, old_weight)

    def test_perplexity_on_vqvae(self):
        model = VQVAE()
        x = torch.randn(2, 4, 32, 32)
        ppl = model.perplexity(x)
        assert 1.0 <= ppl <= model.quantizer.num_embeddings

    def test_reset_dead_codes_on_vqvae(self):
        model = VQVAE(in_channels=4, hidden_dim=32, latent_dim=16, num_embeddings=32)
        x = torch.randn(1, 4, 32, 32)
        old_weight = model.quantizer.embedding.data.clone()
        n_reset = model.reset_dead_codes(x, threshold=0.0)
        total = model.quantizer.num_embeddings
        assert 0 < n_reset <= total
        assert not torch.equal(model.quantizer.embedding.data, old_weight)

    def test_gradient_flow(self):
        model = ImprovedVQVAE(hidden_dim=64, latent_dim=32, num_embeddings=64)
        x = torch.randn(2, 4, 32, 32)
        out = model(x)
        loss = out["loss"]
        loss.backward()
        used_params = []
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                used_params.append(name)
        core = [n for n in used_params if 'skip_conv' not in n and 'res2' in n]
        assert len(core) > 0, "No gradients in core decoder params"
        grad_count = sum(1 for _, p in model.named_parameters() if p.requires_grad and p.grad is not None)
        total_trainable = sum(1 for _, p in model.named_parameters() if p.requires_grad)
        assert grad_count >= total_trainable * 0.7, f"Only {grad_count}/{total_trainable} params have gradients"

    @pytest.fixture
    def small_model(self):
        return ImprovedVQVAE(in_channels=4, hidden_dim=32, latent_dim=16, num_embeddings=32)

    def test_small_model_shapes(self, small_model):
        x = torch.randn(2, 4, 32, 32)
        out = small_model(x)
        assert out["recon"].shape == x.shape

    def test_overfit_constant_input(self):
        model = ImprovedVQVAE(hidden_dim=32, latent_dim=16, num_embeddings=32)
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
        x = torch.randn(4, 4, 32, 32)
        initial_loss = model(x)["loss"].item()
        for _ in range(100):
            optimizer.zero_grad()
            loss = model(x)["loss"]
            loss.backward()
            optimizer.step()
        final_loss = model(x)["loss"].item()
        assert final_loss < initial_loss, f"Loss did not decrease: {initial_loss:.4f} -> {final_loss:.4f}"

    def test_discriminator_attached(self):
        model = ImprovedVQVAE()
        assert hasattr(model, "discriminator")
        x = torch.randn(2, 4, 32, 32)
        d_out = model.discriminator(x)
        assert d_out.shape[-1] < 32

    def test_perceptual_loss_attached(self):
        model = ImprovedVQVAE()
        assert hasattr(model, "perceptual_loss")

    def test_ema_init_and_update(self):
        model = ImprovedVQVAE(hidden_dim=64, latent_dim=32, num_embeddings=64)
        model.init_ema()
        model.update_ema()
        assert model.ema_encoder is not None

    def test_full_loss_computation(self):
        model = ImprovedVQVAE(hidden_dim=64, latent_dim=32, num_embeddings=64)
        x = torch.randn(2, 4, 32, 32)
        out = model(x)
        palette = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        extra = model.compute_full_loss(x, out["recon"], out["indices"], palette)
        assert isinstance(extra, torch.Tensor)
        assert extra.ndim == 0
