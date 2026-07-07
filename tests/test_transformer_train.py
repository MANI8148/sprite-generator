"""Tests for Transformer training loop components (roadmap item #3)."""
import torch
import math
from pathlib import Path

from models.transformer.model import SpriteTransformer


class TestCheckpointCycle:
    def test_save_and_load(self, tmp_path):
        model = SpriteTransformer(
            vocab_size=64, condition_vocab_size=64,
            d_model=32, n_layers=2, n_heads=2, max_seq_len=32,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        ckpt_path = tmp_path / "transformer_test.pt"
        torch.save({
            "epoch": 5,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "loss": 0.5,
        }, ckpt_path)

        model2 = SpriteTransformer(
            vocab_size=64, condition_vocab_size=64,
            d_model=32, n_layers=2, n_heads=2, max_seq_len=32,
        )
        optimizer2 = torch.optim.Adam(model2.parameters(), lr=1e-3)
        checkpoint = torch.load(ckpt_path)
        model2.load_state_dict(checkpoint["model_state"])
        optimizer2.load_state_dict(checkpoint["optimizer_state"])

        for p1, p2 in zip(model.parameters(), model2.parameters()):
            assert torch.equal(p1, p2), "Weights differ after save/load"

    def test_checkpoint_contains_metadata(self, tmp_path):
        model = SpriteTransformer(
            vocab_size=64, condition_vocab_size=64,
            d_model=32, n_layers=2, n_heads=2, max_seq_len=32,
        )
        ckpt_path = tmp_path / "transformer_meta.pt"
        data = {
            "epoch": 10,
            "model_state": model.state_dict(),
            "optimizer_state": torch.optim.Adam(model.parameters(), lr=1e-3).state_dict(),
            "loss": 0.25,
            "config": {
                "d_model": 32,
                "n_layers": 2,
                "n_heads": 2,
            },
        }
        torch.save(data, ckpt_path)
        loaded = torch.load(ckpt_path)
        assert loaded["epoch"] == 10
        assert loaded["loss"] == 0.25
        assert loaded["config"]["d_model"] == 32
        assert loaded["config"]["n_layers"] == 2


class TestTrainingSteps:
    @staticmethod
    def make_model():
        return SpriteTransformer(
            vocab_size=64, condition_vocab_size=64,
            d_model=32, n_layers=2, n_heads=2, max_seq_len=32,
        )

    def test_training_step_updates_weights(self):
        model = self.make_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        B, T = 2, 16
        token_indices = torch.randint(0, 64, (B, T))
        class_ids = torch.randint(0, 10, (B,))
        action_ids = torch.randint(0, 10, (B,))
        direction_ids = torch.randint(0, 8, (B,))

        initial_params = [p.clone() for p in model.parameters()]

        logits = model(token_indices, class_ids, action_ids, direction_ids)
        loss = torch.nn.functional.cross_entropy(logits.view(-1, 64), token_indices.view(-1))
        loss.backward()
        optimizer.step()

        changed = any(
            not torch.equal(p1, p2)
            for p1, p2 in zip(initial_params, model.parameters())
        )
        assert changed, "No parameters changed after training step"

    def test_training_loss_decreases(self):
        model = self.make_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        B, T = 2, 16
        token_indices = torch.randint(0, 64, (B, T))
        class_ids = torch.randint(0, 10, (B,))
        action_ids = torch.randint(0, 10, (B,))
        direction_ids = torch.randint(0, 8, (B,))

        logits = model(token_indices, class_ids, action_ids, direction_ids)
        initial_loss = torch.nn.functional.cross_entropy(
            logits.view(-1, 64), token_indices.view(-1)
        ).item()

        for _ in range(50):
            optimizer.zero_grad()
            logits = model(token_indices, class_ids, action_ids, direction_ids)
            loss = torch.nn.functional.cross_entropy(logits.view(-1, 64), token_indices.view(-1))
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            logits = model(token_indices, class_ids, action_ids, direction_ids)
            final_loss = torch.nn.functional.cross_entropy(
                logits.view(-1, 64), token_indices.view(-1)
            ).item()
        assert final_loss < initial_loss, (
            f"Loss did not decrease: {initial_loss:.4f} -> {final_loss:.4f}"
        )

    def test_training_with_synthetic_dataset(self):
        class SyntheticTokenDataset(torch.utils.data.Dataset):
            def __init__(self, size=16, seq_len=16):
                self.size = size
                self.seq_len = seq_len

            def __len__(self):
                return self.size

            def __getitem__(self, idx):
                tokens = torch.randint(0, 64, (self.seq_len,))
                class_id = torch.randint(0, 10, ())
                action_id = torch.randint(0, 10, ())
                direction_id = torch.randint(0, 8, ())
                return tokens, class_id, action_id, direction_id

        model = self.make_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        dataset = SyntheticTokenDataset(size=16, seq_len=16)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=4,
            collate_fn=lambda batch: (
                torch.stack([b[0] for b in batch]),
                torch.stack([b[1] for b in batch]),
                torch.stack([b[2] for b in batch]),
                torch.stack([b[3] for b in batch]),
            ),
        )

        model.train()
        for tokens, class_ids, action_ids, direction_ids in dataloader:
            optimizer.zero_grad()
            logits = model(tokens, class_ids, action_ids, direction_ids)
            loss = torch.nn.functional.cross_entropy(logits.view(-1, 64), tokens.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    def test_gradient_clipping_does_not_crash(self):
        model = self.make_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        B, T = 2, 16
        token_indices = torch.randint(0, 64, (B, T))
        class_ids = torch.randint(0, 10, (B,))
        action_ids = torch.randint(0, 10, (B,))
        direction_ids = torch.randint(0, 8, (B,))

        logits = model(token_indices, class_ids, action_ids, direction_ids)
        loss = torch.nn.functional.cross_entropy(logits.view(-1, 64), token_indices.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    def test_training_with_lr_scheduler(self):
        model = self.make_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

        B, T = 2, 16
        token_indices = torch.randint(0, 64, (B, T))
        class_ids = torch.randint(0, 10, (B,))
        action_ids = torch.randint(0, 10, (B,))
        direction_ids = torch.randint(0, 8, (B,))

        lrs = []
        for epoch in range(8):
            optimizer.zero_grad()
            logits = model(token_indices, class_ids, action_ids, direction_ids)
            loss = torch.nn.functional.cross_entropy(logits.view(-1, 64), token_indices.view(-1))
            loss.backward()
            optimizer.step()
            scheduler.step()
            lrs.append(optimizer.param_groups[0]["lr"])

        # StepLR: LR halves every 3 steps after init epoch 0
        assert lrs[0] == 0.001
        assert lrs[1] == 0.001
        assert lrs[2] == 0.0005
        assert lrs[5] == 0.00025

    def test_training_with_checkpoint_and_resume(self, tmp_path):
        model = self.make_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        B, T = 2, 16
        token_indices = torch.randint(0, 64, (B, T))
        class_ids = torch.randint(0, 10, (B,))
        action_ids = torch.randint(0, 10, (B,))
        direction_ids = torch.randint(0, 8, (B,))

        for _ in range(3):
            optimizer.zero_grad()
            logits = model(token_indices, class_ids, action_ids, direction_ids)
            loss = torch.nn.functional.cross_entropy(logits.view(-1, 64), token_indices.view(-1))
            loss.backward()
            optimizer.step()

        ckpt_path = tmp_path / "resume_test.pt"
        torch.save({
            "epoch": 3,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "loss": 0.5,
        }, ckpt_path)

        model2 = self.make_model()
        optimizer2 = torch.optim.Adam(model2.parameters(), lr=1e-3)
        checkpoint = torch.load(ckpt_path)
        model2.load_state_dict(checkpoint["model_state"])
        optimizer2.load_state_dict(checkpoint["optimizer_state"])

        model.eval()
        model2.eval()
        logits_orig = model(token_indices, class_ids, action_ids, direction_ids)
        logits_loaded = model2(token_indices, class_ids, action_ids, direction_ids)
        assert torch.allclose(logits_orig, logits_loaded), (
            "Loaded model should produce same output as saved model"
        )

    def test_checkpoint_with_scheduler_state(self, tmp_path):
        model = self.make_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        ckpt_path = tmp_path / "scheduler_ckpt.pt"
        torch.save({
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "epoch": 5,
            "config": {"d_model": 32, "n_layers": 2, "n_heads": 2},
        }, ckpt_path)
        loaded = torch.load(ckpt_path)
        assert "scheduler_state" in loaded
        assert loaded["config"]["d_model"] == 32

    def test_training_with_different_batch_sizes(self):
        model = self.make_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        for B in [1, 4, 8]:
            T = 16
            token_indices = torch.randint(0, 64, (B, T))
            class_ids = torch.randint(0, 10, (B,))
            action_ids = torch.randint(0, 10, (B,))
            direction_ids = torch.randint(0, 8, (B,))

            optimizer.zero_grad()
            logits = model(token_indices, class_ids, action_ids, direction_ids)
            loss = torch.nn.functional.cross_entropy(logits.view(-1, 64), token_indices.view(-1))
            loss.backward()
            optimizer.step()
            assert logits.shape == (B, T, 64)
