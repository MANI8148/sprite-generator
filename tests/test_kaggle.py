"""
Tests for Kaggle training infrastructure (roadmap item #4).
Validates kernel metadata, notebook structure, and training imports.
"""
import json
import importlib
from pathlib import Path

import pytest

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
