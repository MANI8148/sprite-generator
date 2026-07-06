import torch
import torch.nn.functional as F
import pytest

from models.lora.model import LoRALinear, LoRAConv2d, SpriteLoRAWrapper


class TestLoRALinear:
    def test_output_shape(self):
        layer = LoRALinear(64, 128, rank=4)
        x = torch.randn(2, 64)
        out = layer(x)
        assert out.shape == (2, 128)

    def test_lora_adaptation_changes_output(self):
        layer = LoRALinear(32, 32, rank=4, alpha=10.0)
        x = torch.randn(1, 32)
        out_before = layer(x)
        layer.lora_b.data = torch.randn(4, 32) * 0.5
        out_after = layer(x)
        assert not torch.allclose(out_before, out_after)

    def test_merge_weights(self):
        layer = LoRALinear(16, 16, rank=2, alpha=2.0)
        x = torch.randn(1, 16)
        out_before_merge = layer(x)
        layer.merge_weights()
        out_after_merge = layer(x)
        assert torch.allclose(out_before_merge, out_after_merge, atol=1e-5)

    def test_merge_sets_lora_to_zero(self):
        layer = LoRALinear(16, 16, rank=2)
        layer.merge_weights()
        assert torch.allclose(layer.lora_a, torch.zeros_like(layer.lora_a))
        assert torch.allclose(layer.lora_b, torch.zeros_like(layer.lora_b))

    def test_different_batch_sizes(self):
        layer = LoRALinear(32, 64, rank=4)
        x1 = torch.randn(1, 32)
        out1 = layer(x1)
        assert out1.shape == (1, 64)
        x2 = torch.randn(8, 32)
        out2 = layer(x2)
        assert out2.shape == (8, 64)

    def test_base_weights_frozen(self):
        layer = LoRALinear(32, 64, rank=4)
        assert not layer.linear.weight.requires_grad
        assert not layer.linear.bias.requires_grad
        assert layer.lora_a.requires_grad
        assert layer.lora_b.requires_grad


class TestLoRAConv2d:
    def test_output_shape(self):
        layer = LoRAConv2d(4, 64, 3, rank=4, padding=1)
        x = torch.randn(2, 4, 32, 32)
        out = layer(x)
        assert out.shape == (2, 64, 32, 32)

    def test_stride_changes_resolution(self):
        layer = LoRAConv2d(64, 128, 4, rank=4, stride=2, padding=1)
        x = torch.randn(2, 64, 16, 16)
        out = layer(x)
        assert out.shape == (2, 128, 8, 8)

    def test_lora_adaptation_changes_output(self):
        layer = LoRAConv2d(4, 4, 3, rank=4, alpha=10.0, padding=1)
        x = torch.randn(1, 4, 8, 8)
        out_before = layer(x)
        layer.lora_b.data = torch.randn(4, 4) * 0.5
        out_after = layer(x)
        assert not torch.allclose(out_before, out_after)

    def test_merge_weights(self):
        layer = LoRAConv2d(4, 8, 3, rank=2, alpha=2.0, padding=1)
        x = torch.randn(1, 4, 8, 8)
        out_before_merge = layer(x)
        layer.merge_weights()
        out_after_merge = layer(x)
        assert torch.allclose(out_before_merge, out_after_merge, atol=1e-5)


class TestSpriteLoRAWrapper:
    @pytest.fixture
    def model(self):
        return SpriteLoRAWrapper(rank=4, alpha=1.0)

    def test_forward_shapes(self, model):
        x = torch.randn(2, 4, 32, 32)
        out = model(x)
        assert out.shape == x.shape

    def test_different_batch_size(self, model):
        x = torch.randn(4, 4, 32, 32)
        out = model(x)
        assert out.shape == x.shape

    def test_gradient_flow(self, model):
        x = torch.randn(2, 4, 32, 32)
        out = model(x)
        loss = F.mse_loss(out, x)
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Parameter {name} has no gradient"

    def test_trainable_parameters_count(self):
        model = SpriteLoRAWrapper(rank=4)
        count = model.trainable_parameters()
        assert count > 0

    def test_lora_parameters_list(self):
        model = SpriteLoRAWrapper(rank=4)
        lora_params = model.lora_parameters()
        for p in lora_params:
            assert p.requires_grad

    def test_overfit_constant_input(self):
        model = SpriteLoRAWrapper(rank=8, alpha=4.0)
        optimizer = torch.optim.Adam(model.lora_parameters(), lr=1e-2)
        x = torch.ones(2, 4, 32, 32)
        initial_loss = F.mse_loss(model(x), x).item()
        for _ in range(100):
            optimizer.zero_grad()
            loss = F.mse_loss(model(x), x)
            loss.backward()
            optimizer.step()
        final_loss = F.mse_loss(model(x), x).item()
        assert final_loss < initial_loss, f"Loss did not decrease: {initial_loss:.4f} -> {final_loss:.4f}"

    def test_base_conv_weights_frozen(self):
        model = SpriteLoRAWrapper(rank=4)
        for name, param in model.named_parameters():
            if "lora" not in name:
                assert not param.requires_grad, f"Parameter {name} should be frozen"

    def test_merge_all_weights(self):
        model = SpriteLoRAWrapper(rank=4, alpha=2.0)
        x = torch.randn(1, 4, 32, 32)
        out_before = model(x)
        for module in model.modules():
            if isinstance(module, (LoRALinear, LoRAConv2d)):
                module.merge_weights()
        out_after = model(x)
        assert torch.allclose(out_before, out_after, atol=1e-4)
