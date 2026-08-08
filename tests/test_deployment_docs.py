"""Tests for the deployment documentation (roadmap item: Docker + deployment docs, CI/CD)."""
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
DEPLOYMENT_DOC = REPO_ROOT / "DEPLOYMENT.md"


class TestDeploymentDocExists:
    def test_deployment_doc_exists(self):
        assert DEPLOYMENT_DOC.exists(), "DEPLOYMENT.md must exist"

    def test_deployment_doc_is_markdown(self):
        assert DEPLOYMENT_DOC.suffix == ".md"


class TestDeploymentDocContent:
    def _read(self) -> str:
        return DEPLOYMENT_DOC.read_text()

    def test_covers_local_backend(self):
        text = self._read()
        assert "uvicorn backend.main:app" in text

    def test_covers_local_frontend(self):
        text = self._read()
        assert "npm run dev" in text

    def test_covers_docker_compose(self):
        text = self._read()
        assert "docker compose up" in text

    def test_covers_backend_service(self):
        text = self._read()
        assert "backend" in text
        assert "8000" in text

    def test_covers_frontend_service(self):
        text = self._read()
        assert "frontend" in text
        assert "3000" in text

    def test_covers_postgres(self):
        text = self._read()
        assert "postgres" in text.lower()

    def test_covers_redis(self):
        text = self._read()
        assert "redis" in text.lower()

    def test_covers_hf_spaces(self):
        text = self._read()
        assert "Spaces" in text
        assert "sprite-generator-demo" in text

    def test_covers_environment_variables(self):
        text = self._read()
        assert "DATABASE_URL" in text
        assert "REDIS_URL" in text
        assert "ALLOWED_ORIGINS" in text

    def test_covers_ci_cd(self):
        text = self._read()
        assert "scripts/ci.yml" in text
        assert "pytest" in text

    def test_covers_r2_storage(self):
        text = self._read()
        assert "R2" in text

    def test_covers_health_check(self):
        text = self._read()
        assert "/health" in text


class TestDeploymentDocConsistency:
    """The docs must reference files that actually exist in the repo."""

    def test_docker_compose_file_exists(self):
        assert (REPO_ROOT / "docker-compose.yml").exists()

    def test_dockerfile_exists(self):
        assert (REPO_ROOT / "Dockerfile").exists()

    def test_ci_workflow_exists(self):
        assert (REPO_ROOT / "scripts" / "ci.yml").exists()

    def test_deploy_script_exists(self):
        assert (REPO_ROOT / "scripts" / "deploy_spaces.py").exists()

    def test_gradio_app_readme_has_space_config(self):
        readme = (REPO_ROOT / "gradio_app" / "README.md").read_text()
        assert "sdk: gradio" in readme
        assert "app_file" in readme

    def test_env_vars_documented_match_code(self):
        """Every environment variable referenced in DEPLOYMENT.md must exist in the backend."""
        text = DEPLOYMENT_DOC.read_text()
        import re

        documented = set(re.findall(r"`([A-Z][A-Z0-9_]+)`", text))
        backend_files = list((REPO_ROOT / "backend").rglob("*.py"))
        used_in_code = set()
        for path in backend_files:
            src = path.read_text()
            used_in_code.update(re.findall(r'os\.(?:environ\.get|getenv)\(\s*"([A-Z][A-Z0-9_]+)"', src))
        for var in documented:
            if var in {"HF_TOKEN", "HF_SPACE_REPO", "NEXT_PUBLIC_API_URL"}:
                continue
            assert var in used_in_code, (
                f"{var} documented in DEPLOYMENT.md but not read by the backend"
            )
