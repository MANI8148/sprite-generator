from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent


class TestDockerfile:
    def test_dockerfile_exists(self):
        assert (REPO_ROOT / "Dockerfile").exists()

    def test_dockerfile_uses_python310(self):
        text = (REPO_ROOT / "Dockerfile").read_text()
        assert "python:3.10" in text

    def test_dockerfile_exposes_port_8000(self):
        text = (REPO_ROOT / "Dockerfile").read_text()
        assert "EXPOSE 8000" in text

    def test_dockerfile_runs_uvicorn(self):
        text = (REPO_ROOT / "Dockerfile").read_text()
        assert "uvicorn" in text
        assert "backend.main:app" in text

    def test_dockerfile_frontend_stage(self):
        text = (REPO_ROOT / "Dockerfile").read_text()
        assert "frontend-build" in text
        assert "frontend-prod" in text

    def test_dockerfile_frontend_exposes_port_3000(self):
        text = (REPO_ROOT / "Dockerfile").read_text()
        assert "EXPOSE 3000" in text

    def test_dockerfile_installs_requirements(self):
        text = (REPO_ROOT / "Dockerfile").read_text()
        assert "requirements.txt" in text
        assert "pip install" in text


class TestDockerCompose:
    def test_docker_compose_exists(self):
        assert (REPO_ROOT / "docker-compose.yml").exists()

    def test_docker_compose_backend_service(self):
        text = (REPO_ROOT / "docker-compose.yml").read_text()
        assert "backend:" in text
        assert "8000:8000" in text

    def test_docker_compose_frontend_service(self):
        text = (REPO_ROOT / "docker-compose.yml").read_text()
        assert "frontend:" in text
        assert "3000:3000" in text

    def test_docker_compose_backend_depends_on(self):
        text = (REPO_ROOT / "docker-compose.yml").read_text()
        assert "depends_on" in text


class TestDockerignore:
    def test_dockerignore_exists(self):
        assert (REPO_ROOT / ".dockerignore").exists()

    def test_dockerignore_excludes_pycache(self):
        text = (REPO_ROOT / ".dockerignore").read_text()
        assert "__pycache__" in text or "*.pyc" in text

    def test_dockerignore_excludes_git(self):
        text = (REPO_ROOT / ".dockerignore").read_text()
        assert ".git" in text
