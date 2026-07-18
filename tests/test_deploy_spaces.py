"""Tests for the Hugging Face Spaces deployment script (roadmap item #7)."""
from pathlib import Path

SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"
DEPLOY_SCRIPT = SCRIPTS_DIR / "deploy_spaces.py"
WORKFLOW = Path(__file__).parent.parent / ".github" / "workflows" / "deploy_demo.yml"


class TestDeployScript:
    def test_script_exists(self):
        assert DEPLOY_SCRIPT.exists(), "scripts/deploy_spaces.py must exist"

    def test_script_is_python(self):
        assert DEPLOY_SCRIPT.suffix == ".py"

    def test_script_is_valid_python(self):
        import py_compile
        py_compile.compile(str(DEPLOY_SCRIPT), doraise=True)

    def test_collect_files_returns_gradio_app_files(self):
        from scripts.deploy_spaces import collect_files
        files = collect_files([
            Path(__file__).parent.parent / "gradio_app",
        ])
        paths = [repo for _local, repo in files]
        assert any(p.endswith("app.py") for p in paths)
        assert any(p.endswith("README.md") for p in paths)
        assert any(p.endswith("requirements.txt") for p in paths)

    def test_collect_files_returns_backend_files(self):
        from scripts.deploy_spaces import collect_files
        files = collect_files([
            Path(__file__).parent.parent / "backend",
        ])
        paths = [repo for _local, repo in files]
        assert any(p.endswith(".py") for p in paths)
        assert any("backend/modules" in p for p in paths)

    def test_collect_files_missing_dir_does_not_crash(self):
        from scripts.deploy_spaces import collect_files
        files = collect_files([
            Path("/nonexistent/path"),
        ])
        assert files == []

    def test_deploy_dry_run_outputs_files(self, capsys):
        from scripts.deploy_spaces import deploy
        rc = deploy(
            space_repo="test/repo",
            hf_token="fake-token",
            source_dirs=[Path(__file__).parent.parent / "gradio_app"],
            dry_run=True,
        )
        captured = capsys.readouterr()
        assert rc == 0
        assert "[DRY RUN]" in captured.out
        assert "app.py" in captured.out

    def test_deploy_requires_token(self, capsys):
        from scripts.deploy_spaces import deploy
        rc = deploy(
            space_repo="test/repo",
            hf_token="",
            source_dirs=[Path(__file__).parent.parent / "gradio_app"],
        )
        captured = capsys.readouterr()
        assert rc == 1

    def test_script_has_main(self):
        from scripts.deploy_spaces import main
        assert callable(main)

    def test_collect_files_flatten_gradio_app(self):
        from scripts.deploy_spaces import collect_files
        files = collect_files(
            [Path(__file__).parent.parent / "gradio_app"],
            flatten_dirs={"gradio_app"},
        )
        paths = [repo for _local, repo in files]
        assert any(p == "app.py" for p in paths), "gradio_app/app.py should flatten to app.py"
        assert any(p == "README.md" for p in paths), "gradio_app/README.md should flatten to README.md"
        assert all(not p.startswith("gradio_app/") for p in paths), "No paths should start with gradio_app/"

    def test_collect_files_keeps_backend_paths(self):
        from scripts.deploy_spaces import collect_files
        files = collect_files(
            [Path(__file__).parent.parent / "backend"],
            flatten_dirs={"gradio_app"},
        )
        paths = [repo for _local, repo in files]
        assert any(p.startswith("backend/") for p in paths), "Backend paths should keep backend/ prefix"


class TestDeployWorkflow:
    def test_workflow_exists(self):
        assert WORKFLOW.exists(), "deploy_demo.yml workflow must exist"

    def test_triggers_on_push_main(self):
        text = WORKFLOW.read_text()
        assert "push:" in text
        assert "main" in text

    def test_triggers_on_demo_paths(self):
        text = WORKFLOW.read_text()
        assert "demo/" in text or "'demo/**'" in text or "demo/**" in text

    def test_triggers_on_models_paths(self):
        text = WORKFLOW.read_text()
        assert "models/" in text or "'models/**'" in text or "models/**" in text

    def test_uses_hf_api_upload(self):
        text = WORKFLOW.read_text()
        assert "upload_file" in text and "HfApi" in text

    def test_uses_hf_token_secret(self):
        text = WORKFLOW.read_text()
        assert "HF_TOKEN" in text

    def test_uses_actions_checkout(self):
        text = WORKFLOW.read_text()
        assert "actions/checkout@v4" in text

    def test_installs_huggingface_hub(self):
        text = WORKFLOW.read_text()
        assert "huggingface_hub" in text

    def test_has_workflow_dispatch(self):
        text = WORKFLOW.read_text()
        assert "workflow_dispatch" in text

    def test_sets_hf_space_repo(self):
        text = WORKFLOW.read_text()
        script_text = DEPLOY_SCRIPT.read_text()
        assert (
            "sprite-generator-demo" in script_text
            or "HF_SPACE_REPO" in text
        )
