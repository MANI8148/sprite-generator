"""Tests for Hugging Face Spaces deployment configuration (roadmap item #6)."""
import importlib.util
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).parent.parent
GRADIO_APP = ROOT / "gradio_app"
README = GRADIO_APP / "README.md"
APP_PY = GRADIO_APP / "app.py"


def _import_gradio_app():
    spec = importlib.util.spec_from_file_location("gradio_app_module", APP_PY)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


class TestSpacesReadme:
    def test_readme_exists(self):
        assert README.exists(), "gradio_app/README.md must exist for HF Spaces deployment"

    def test_readme_has_yaml_frontmatter(self):
        text = README.read_text()
        assert text.startswith("---"), "README.md must start with YAML frontmatter"
        assert "---" in text[3:], "README.md must have closing YAML frontmatter"

    def test_readme_has_title(self):
        text = README.read_text()
        assert "title:" in text.split("---")[1]

    def test_readme_has_sdk_gradio(self):
        text = README.read_text()
        assert "sdk: gradio" in text.split("---")[1]

    def test_readme_has_app_file(self):
        text = README.read_text()
        assert "app_file:" in text.split("---")[1]

    def test_readme_valid_yaml(self):
        parts = README.read_text().split("---")
        metadata = yaml.safe_load(parts[1])
        assert isinstance(metadata, dict)
        assert "title" in metadata
        assert metadata["sdk"] == "gradio"


class TestSpacesApp:
    def test_app_py_exists(self):
        assert APP_PY.exists(), "gradio_app/app.py must exist"

    def test_app_is_importable(self):
        mod = _import_gradio_app()
        assert hasattr(mod, "demo"), "app.py must export 'demo' object"

    def test_app_has_css(self):
        mod = _import_gradio_app()
        assert hasattr(mod, "css"), "app.py must define 'css' variable"

    def test_app_has_pipeline(self):
        mod = _import_gradio_app()
        assert hasattr(mod, "pipeline"), "app.py must define 'pipeline' variable"


class TestSpacesConfig:
    def test_gradio_app_has_requirements(self):
        req = GRADIO_APP / "requirements.txt"
        assert req.exists(), "gradio_app/requirements.txt must exist for HF Spaces"

    def test_requirements_include_gradio(self):
        text = (GRADIO_APP / "requirements.txt").read_text()
        assert "gradio" in text, "requirements.txt must include gradio"

    def test_requirements_include_torch(self):
        text = (GRADIO_APP / "requirements.txt").read_text()
        assert "torch" in text, "requirements.txt must include torch"


class TestDemoFallback:
    def test_demo_launch_method(self):
        mod = _import_gradio_app()
        assert hasattr(mod.demo, "launch"), "demo object must have launch method"

    def test_demo_queue_method(self):
        mod = _import_gradio_app()
        assert hasattr(mod.demo, "queue"), "demo object must have queue method for Spaces"

    def test_demo_title(self):
        mod = _import_gradio_app()
        assert hasattr(mod.demo, "title")

    def test_load_model_exists(self):
        mod = _import_gradio_app()
        assert hasattr(mod, "load_model"), "app.py must define load_model function"

    def test_generate_exists(self):
        mod = _import_gradio_app()
        assert hasattr(mod, "generate"), "app.py must define generate function"
