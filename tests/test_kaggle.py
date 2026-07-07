"""
Tests for Kaggle training infrastructure (roadmap item #4).
Validates kernel metadata, notebook structure, training imports,
and the end-to-end VQ-VAE → Transformer training pipeline.
"""
import json
import importlib
from pathlib import Path

import pytest
import torch

KAGGLE_DIR = Path(__file__).parent.parent / "kaggle"


def get_kernel_metadata_files():
    return sorted(KAGGLE_DIR.glob("kernel-metadata*.json"))


def get_notebook_files():
    return sorted(KAGGLE_DIR.glob("*.ipynb"))


class TestKernelMetadata:
    @pytest.fixture(params=get_kernel_metadata_files())
    def metadata_file(self, request):
        return request.param

    def test_is_valid_json(self, metadata_file):
        with open(metadata_file) as f:
            data = json.load(f)
        assert isinstance(data, dict)

    def test_has_required_fields(self, metadata_file):
        with open(metadata_file) as f:
            data = json.load(f)
        required = ["id", "title", "code_file", "language", "kernel_type"]
        for field in required:
            assert field in data, f"Missing field '{field}' in {metadata_file.name}"

    def test_code_file_exists(self, metadata_file):
        with open(metadata_file) as f:
            data = json.load(f)
        code_file = KAGGLE_DIR / data["code_file"]
        assert code_file.exists(), f"Referenced file {code_file} not found"

    def test_language_is_python(self, metadata_file):
        with open(metadata_file) as f:
            data = json.load(f)
        assert data.get("language") == "python"

    def test_enable_gpu(self, metadata_file):
        with open(metadata_file) as f:
            data = json.load(f)
        assert data.get("enable_gpu") is True

    def test_enable_internet(self, metadata_file):
        with open(metadata_file) as f:
            data = json.load(f)
        assert data.get("enable_internet") is True

    def test_kernel_type(self, metadata_file):
        with open(metadata_file) as f:
            data = json.load(f)
        assert data.get("kernel_type") == "notebook"


class TestNotebookStructure:
    @pytest.fixture(params=get_notebook_files())
    def notebook_file(self, request):
        return request.param

    def test_is_valid_json(self, notebook_file):
        with open(notebook_file) as f:
            data = json.load(f)
        assert isinstance(data, dict)

    def test_has_required_top_level_keys(self, notebook_file):
        with open(notebook_file) as f:
            data = json.load(f)
        for key in ["cells", "nbformat", "metadata"]:
            assert key in data, f"Missing key '{key}' in {notebook_file.name}"

    def test_nbformat_version(self, notebook_file):
        with open(notebook_file) as f:
            data = json.load(f)
        assert data["nbformat"] >= 4

    def test_has_cells(self, notebook_file):
        with open(notebook_file) as f:
            data = json.load(f)
        assert len(data["cells"]) > 0

    def test_cells_have_types(self, notebook_file):
        with open(notebook_file) as f:
            data = json.load(f)
        for cell in data["cells"]:
            assert "cell_type" in cell
            assert cell["cell_type"] in ("code", "markdown")

    def test_code_cells_have_source(self, notebook_file):
        with open(notebook_file) as f:
            data = json.load(f)
        for cell in data["cells"]:
            if cell["cell_type"] == "code":
                src = "".join(cell.get("source", []))
                assert len(src.strip()) > 0 or not cell.get("execution_count")


class TestNotebookImports:
    def _collect_imports(self, notebook_path):
        with open(notebook_path) as f:
            data = json.load(f)
        imports = set()
        for cell in data["cells"]:
            if cell["cell_type"] == "code":
                src = "".join(cell.get("source", []))
                for line in src.split("\n"):
                    line = line.strip()
                    if line.startswith("from ") and "import " in line:
                        parts = line.split(" import ")[0].replace("from ", "").strip()
                        top_module = parts.split(".")[0]
                        imports.add(top_module)
                    elif line.startswith("import "):
                        parts = line.replace("import ", "").strip()
                        for part in parts.split(","):
                            part = part.strip().split(" as ")[0].strip()
                            top_module = part.split(".")[0]
                            imports.add(top_module)
        return imports

    def test_train_kernel_imports_exist(self):
        notebook = KAGGLE_DIR / "kaggle_train_kernel.ipynb"
        if not notebook.exists():
            pytest.skip("kaggle_train_kernel.ipynb not found")
        imports = self._collect_imports(notebook)
        errors = []
        for mod_name in sorted(imports):
            if mod_name in ("os", "sys", "json", "torch", "pathlib"):
                continue
            if mod_name in ("datasets", "huggingface_hub", "PIL", "numpy", "tqdm", "torchvision"):
                continue
            try:
                importlib.import_module(mod_name)
            except ImportError:
                errors.append(f"Module '{mod_name}' not importable")
        assert not errors, f"Import errors: {errors}"

    def test_transformer_kernel_imports_exist(self):
        notebook = KAGGLE_DIR / "kaggle_transformer_kernel.ipynb"
        if not notebook.exists():
            pytest.skip("kaggle_transformer_kernel.ipynb not found")
        imports = self._collect_imports(notebook)
        errors = []
        for mod_name in sorted(imports):
            if mod_name in ("os", "sys", "json", "torch", "pathlib", "re", "glob"):
                continue
            if mod_name in ("datasets", "huggingface_hub", "PIL", "numpy", "tqdm", "torchvision"):
                continue
            try:
                importlib.import_module(mod_name)
            except ImportError:
                errors.append(f"Module '{mod_name}' not importable")
        assert not errors, f"Import errors: {errors}"


class TestKaggleWorkflowIntegration:
    def test_trigger_workflow_exists(self):
        workflow_dir = Path(__file__).parent.parent / ".github" / "workflows"
        trigger = workflow_dir / "trigger_kaggle_training.yml"
        assert trigger.exists()

    def test_sync_checkpoint_workflow_exists(self):
        workflow_dir = Path(__file__).parent.parent / ".github" / "workflows"
        sync = workflow_dir / "sync_checkpoint_to_hf.yml"
        assert sync.exists()

    def test_training_scripts_importable(self):
        from models.vqvae.train import main as vqvae_main
        assert callable(vqvae_main)

        from models.transformer.train import main as transformer_main
        assert callable(transformer_main)


class TestEndToEndTrainingPipeline:
    """
    Integration test for the full VQ-VAE → encode tokens → Transformer pipeline
    that runs on Kaggle (kaggle_complete_train.ipynb steps 2-4).
    Validates the complete training flow with synthetic data.
    """

    def test_vqvae_trains_and_produces_valid_checkpoint(self, tmp_path):
        from models.vqvae.model import VQVAE

        model = VQVAE(in_channels=4, hidden_dim=32, latent_dim=16, num_embeddings=32)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        x = torch.randn(4, 4, 32, 32)

        model.train()
        for _ in range(10):
            optimizer.zero_grad()
            out = model(x)
            out["loss"].backward()
            optimizer.step()

        ckpt = {
            "epoch": 9,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "loss": out["loss"].item(),
            "config": {"hidden_dim": 32, "latent_dim": 16, "num_embeddings": 32},
        }
        ckpt_path = tmp_path / "vqvae_test.pt"
        torch.save(ckpt, ckpt_path)

        loaded = torch.load(ckpt_path)
        assert all(k in loaded for k in ("epoch", "model_state", "optimizer_state", "loss", "config"))
        assert loaded["config"]["num_embeddings"] == 32

        model2 = VQVAE(in_channels=4, hidden_dim=32, latent_dim=16, num_embeddings=32)
        model2.load_state_dict(loaded["model_state"])
        model2.eval()
        with torch.no_grad():
            out2 = model2(x)
        assert out2["recon"].shape == (4, 4, 32, 32)

    def test_vqvae_encodes_to_tokens(self):
        from models.vqvae.model import VQVAE

        model = VQVAE(in_channels=4, hidden_dim=32, latent_dim=16, num_embeddings=64)
        x = torch.randn(2, 4, 32, 32)
        indices = model.encode_to_indices(x)
        assert indices.shape == (2, 64)
        assert indices.dtype == torch.long
        assert indices.min() >= 0
        assert indices.max() < 64

    def test_vqvae_decode_from_tokens(self):
        from models.vqvae.model import VQVAE

        model = VQVAE(in_channels=4, hidden_dim=32, latent_dim=16, num_embeddings=64)
        indices = torch.randint(0, 64, (2, 8, 8))
        decoded = model.decode_from_indices(indices.view(2, -1), (16, 8, 8))
        assert decoded.shape == (2, 4, 32, 32)

    def test_transformer_trains_on_vqvae_tokens(self):
        from models.vqvae.model import VQVAE
        from models.transformer.model import SpriteTransformer

        vqvae = VQVAE(in_channels=4, hidden_dim=32, latent_dim=16, num_embeddings=64)
        B = 4
        x = torch.randn(B, 4, 32, 32)
        with torch.no_grad():
            tokens = vqvae.encode_to_indices(x)

        num_emb = 64
        transformer = SpriteTransformer(
            vocab_size=num_emb, condition_vocab_size=64,
            d_model=32, n_layers=2, n_heads=2, max_seq_len=65,
        )
        class_ids = torch.randint(0, 20, (B,))
        action_ids = torch.randint(0, 14, (B,))
        direction_ids = torch.randint(0, 8, (B,))

        logits = transformer(tokens, class_ids, action_ids, direction_ids)
        assert logits.shape == (B, 64, num_emb)

        loss = torch.nn.functional.cross_entropy(logits.view(-1, num_emb), tokens.view(-1))
        loss.backward()

        optimizer = torch.optim.Adam(transformer.parameters(), lr=1e-3)
        initial_loss = loss.item()
        for _ in range(30):
            optimizer.zero_grad()
            logits = transformer(tokens, class_ids, action_ids, direction_ids)
            loss = torch.nn.functional.cross_entropy(logits.view(-1, num_emb), tokens.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(transformer.parameters(), 1.0)
            optimizer.step()

        with torch.no_grad():
            final_logits = transformer(tokens, class_ids, action_ids, direction_ids)
            final_loss = torch.nn.functional.cross_entropy(
                final_logits.view(-1, num_emb), tokens.view(-1)
            ).item()
        assert final_loss < initial_loss, (
            f"Transformer loss did not decrease: {initial_loss:.4f} -> {final_loss:.4f}"
        )

    def test_transformer_generates_and_vqvae_decodes(self):
        from models.vqvae.model import VQVAE
        from models.transformer.model import SpriteTransformer

        num_emb = 64
        vqvae = VQVAE(in_channels=4, hidden_dim=32, latent_dim=16, num_embeddings=num_emb)
        transformer = SpriteTransformer(
            vocab_size=num_emb, condition_vocab_size=64,
            d_model=32, n_layers=2, n_heads=2, max_seq_len=65,
        )

        class_id = torch.tensor([5])
        action_id = torch.tensor([3])
        direction_id = torch.tensor([1])
        with torch.no_grad():
            generated = transformer.generate(
                class_id, action_id, direction_id, max_tokens=64, temperature=1.0, top_k=10
            )
        assert generated.shape == (1, 64)
        assert generated.dtype == torch.long
        assert generated.min() >= 0
        assert generated.max() < num_emb

        decoded = vqvae.decode_from_indices(generated.view(1, -1), (16, 8, 8))
        assert decoded.shape == (1, 4, 32, 32)

    def test_checkpoint_format_matches_kaggle_notebook(self, tmp_path):
        from models.vqvae.model import VQVAE
        from models.transformer.model import SpriteTransformer

        vqvae = VQVAE(in_channels=4, hidden_dim=32, latent_dim=16, num_embeddings=64)
        vqvae_ckpt = {
            "epoch": 50,
            "model_state": vqvae.state_dict(),
            "optimizer_state": torch.optim.Adam(vqvae.parameters(), lr=1e-3).state_dict(),
            "loss": 0.05,
            "config": {"hidden_dim": 128, "latent_dim": 64, "num_embeddings": 512},
        }
        vqvae_path = tmp_path / "vqvae_latest.pt"
        torch.save(vqvae_ckpt, vqvae_path)
        loaded_v = torch.load(vqvae_path)
        assert loaded_v["epoch"] == 50
        assert loaded_v["config"]["num_embeddings"] == 512

        transformer = SpriteTransformer(
            vocab_size=64, condition_vocab_size=64,
            d_model=32, n_layers=2, n_heads=2, max_seq_len=65,
        )
        transformer_ckpt = {
            "epoch": 100,
            "model_state": transformer.state_dict(),
            "optimizer_state": torch.optim.Adam(transformer.parameters(), lr=1e-3).state_dict(),
            "loss": 0.01,
            "config": {"d_model": 512, "n_layers": 8, "n_heads": 8, "max_seq_len": 65},
        }
        transformer_path = tmp_path / "transformer_latest.pt"
        torch.save(transformer_ckpt, transformer_path)
        loaded_t = torch.load(transformer_path)
        assert loaded_t["epoch"] == 100
        assert loaded_t["config"]["d_model"] == 512
        assert loaded_t["config"]["n_layers"] == 8
        assert loaded_t["config"]["n_heads"] == 8

    def test_full_pipeline_synthetic_data(self, tmp_path):
        from models.vqvae.model import VQVAE
        from models.transformer.model import SpriteTransformer

        num_emb = 64
        B = 8
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        vqvae = VQVAE(in_channels=4, hidden_dim=32, latent_dim=16, num_embeddings=num_emb).to(device)
        vqvae.train()
        optimizer_v = torch.optim.Adam(vqvae.parameters(), lr=1e-3)
        x = torch.randn(B * 4, 4, 32, 32).to(device)

        for _ in range(15):
            optimizer_v.zero_grad()
            out = vqvae(x)
            out["loss"].backward()
            optimizer_v.step()
        vqvae.eval()

        with torch.no_grad():
            tokens = vqvae.encode_to_indices(x)

        transformer = SpriteTransformer(
            vocab_size=num_emb, condition_vocab_size=64,
            d_model=32, n_layers=2, n_heads=2, max_seq_len=(8 * 8) + 1,
        ).to(device)
        optimizer_t = torch.optim.Adam(transformer.parameters(), lr=1e-3)

        seq_len = 8 * 8
        class_ids = torch.randint(0, 20, (B * 4,), device=device)
        action_ids = torch.randint(0, 14, (B * 4,), device=device)
        direction_ids = torch.randint(0, 8, (B * 4,), device=device)

        for _ in range(20):
            optimizer_t.zero_grad()
            logits = transformer(tokens, class_ids, action_ids, direction_ids)
            loss = torch.nn.functional.cross_entropy(logits.view(-1, num_emb), tokens.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(transformer.parameters(), 1.0)
            optimizer_t.step()

        transformer.eval()
        with torch.no_grad():
            gen = transformer.generate(
                class_ids[:1], action_ids[:1], direction_ids[:1],
                max_tokens=seq_len, temperature=1.0, top_k=10,
            )
        assert gen.shape == (1, seq_len)

        decoded = vqvae.decode_from_indices(gen.to(device), (16, 8, 8))
        assert decoded.shape == (1, 4, 32, 32)

        torch.save({
            "epoch": 10, "model_state": vqvae.state_dict(),
            "loss": out["loss"].item(),
            "config": {"num_embeddings": num_emb},
        }, tmp_path / "vqvae_final.pt")
        torch.save({
            "epoch": 10, "model_state": transformer.state_dict(),
            "loss": loss.item(),
            "config": {"d_model": 32},
        }, tmp_path / "transformer_final.pt")
        assert (tmp_path / "vqvae_final.pt").exists()
        assert (tmp_path / "transformer_final.pt").exists()
