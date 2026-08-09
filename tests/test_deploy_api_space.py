"""Tests for the FastAPI model+API Hugging Face Spaces deployment
(roadmap Phase 1: "Deploy the model + API — start with Hugging Face Spaces")."""
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).parent.parent
API_SPACE_DIR = REPO_ROOT / "spaces" / "api"
DOCKERFILE = API_SPACE_DIR / "Dockerfile"
README = API_SPACE_DIR / "README.md"
DEPLOYMENT_DOC = REPO_ROOT / "DEPLOYMENT.md"
DEPLOY_SCRIPT = REPO_ROOT / "scripts" / "deploy_spaces.py"


class TestApiSpaceConfig:
    def test_dockerfile_exists(self):
        assert DOCKERFILE.exists(), "spaces/api/Dockerfile must exist for the Docker SDK Space"

    def test_readme_exists(self):
        assert README.exists(), "spaces/api/README.md must exist for HF Spaces deployment"

    def test_readme_has_yaml_frontmatter(self):
        text = README.read_text()
        assert text.startswith("---"), "README.md must start with YAML frontmatter"
        assert "---" in text[3:], "README.md must have closing YAML frontmatter"

    def test_readme_valid_yaml(self):
        parts = README.read_text().split("---")
        metadata = yaml.safe_load(parts[1])
        assert isinstance(metadata, dict)
        assert "title" in metadata
        assert metadata["sdk"] == "docker", "API Space must use the Docker SDK"

    def test_dockerfile_runs_uvicorn(self):
        text = DOCKERFILE.read_text()
        assert "uvicorn" in text, "Dockerfile must start uvicorn for the FastAPI app"
        assert "backend.main:app" in text, "Dockerfile must run backend.main:app"

    def test_dockerfile_exposes_spaces_port(self):
        text = DOCKERFILE.read_text()
        assert "7860" in text, "HF Spaces expects the app to listen on port 7860"

    def test_dockerfile_copies_backend(self):
        text = DOCKERFILE.read_text()
        assert "COPY backend/" in text, "Dockerfile must copy the backend package"

    def test_dockerfile_installs_requirements(self):
        text = DOCKERFILE.read_text()
        assert "requirements.txt" in text, "Dockerfile must install requirements.txt"

    def test_readme_documents_endpoints(self):
        text = README.read_text()
        assert "/generate" in text, "README must document the /generate endpoint"
        assert "/health" in text, "README must document the /health endpoint"


class TestDeployApiSpaceScript:
    def test_deploy_script_exists(self):
        assert DEPLOY_SCRIPT.exists()

    def test_script_exports_api_functions(self):
        from scripts.deploy_spaces import collect_api_space_files, deploy_api_space
        assert callable(collect_api_space_files)
        assert callable(deploy_api_space)

    def test_collect_includes_dockerfile_at_root(self):
        from scripts.deploy_spaces import collect_api_space_files
        repo_paths = [repo for _local, repo in collect_api_space_files()]
        assert "Dockerfile" in repo_paths, "Dockerfile must be flattened to the Space root"

    def test_collect_includes_readme_at_root(self):
        from scripts.deploy_spaces import collect_api_space_files
        repo_paths = [repo for _local, repo in collect_api_space_files()]
        assert "README.md" in repo_paths, "README.md must be flattened to the Space root"

    def test_collect_includes_requirements_at_root(self):
        from scripts.deploy_spaces import collect_api_space_files
        repo_paths = [repo for _local, repo in collect_api_space_files()]
        assert "requirements.txt" in repo_paths, "requirements.txt must be at the Space root"

    def test_collect_keeps_backend_prefix(self):
        from scripts.deploy_spaces import collect_api_space_files
        repo_paths = [repo for _local, repo in collect_api_space_files()]
        assert any(p.startswith("backend/") for p in repo_paths), "backend must keep its backend/ prefix"

    def test_collect_includes_main(self):
        from scripts.deploy_spaces import collect_api_space_files
        repo_paths = [repo for _local, repo in collect_api_space_files()]
        assert "backend/main.py" in repo_paths

    def test_collect_excludes_cache_artifacts(self):
        from scripts.deploy_spaces import collect_api_space_files
        repo_paths = [repo for _local, repo in collect_api_space_files()]
        assert not any("__pycache__" in p or p.endswith((".pyc", ".pyo")) for p in repo_paths)

    def test_deploy_api_dry_run_outputs_files(self, capsys):
        from scripts.deploy_spaces import deploy_api_space
        rc = deploy_api_space(
            space_repo="test/api-repo",
            hf_token="fake-token",
            dry_run=True,
        )
        captured = capsys.readouterr()
        assert rc == 0
        assert "[DRY RUN]" in captured.out
        assert "Dockerfile" in captured.out
        assert "backend/main.py" in captured.out

    def test_deploy_api_requires_token(self, capsys):
        from scripts.deploy_spaces import deploy_api_space
        rc = deploy_api_space(
            space_repo="test/api-repo",
            hf_token="",
            dry_run=False,
        )
        captured = capsys.readouterr()
        assert rc == 1

    def test_script_has_main(self):
        from scripts.deploy_spaces import main
        assert callable(main)


class TestDeployApiWorkflow:
    def test_deployment_doc_documents_api_deploy(self):
        assert DEPLOYMENT_DOC.exists(), "DEPLOYMENT.md must exist"
        text = DEPLOYMENT_DOC.read_text()
        assert "sprite-generator-api" in text, "DEPLOYMENT.md must document the API Space repo"
        assert "sdk: docker" in text or "Docker" in text, "DEPLOYMENT.md must describe the Docker SDK deploy"

    def test_deployment_doc_documents_deploy_command(self):
        text = DEPLOYMENT_DOC.read_text()
        assert "deploy_spaces.py" in text
        assert "--api" in text, "DEPLOYMENT.md must document the --api deploy mode"

    def test_deployment_doc_mentions_port_7860(self):
        text = DEPLOYMENT_DOC.read_text()
        assert "7860" in text, "DEPLOYMENT.md must document the Space port 7860"

    def test_main_accepts_api_flag(self):
        from scripts.deploy_spaces import main
        import inspect
        src = inspect.getsource(main)
        assert "--api" in src, "main() must expose the --api deploy mode"
