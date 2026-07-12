from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

ROOT = Path(__file__).parent.parent
PYPROJECT = ROOT / "pyproject.toml"
REQUIREMENTS = ROOT / "requirements.txt"


class TestPyProjectToml:
    def test_pyproject_exists(self):
        assert PYPROJECT.exists(), "pyproject.toml not found"

    def test_is_valid_toml(self):
        data = tomllib.loads(PYPROJECT.read_text())
        assert isinstance(data, dict)

    def test_has_name(self):
        data = tomllib.loads(PYPROJECT.read_text())
        assert data["project"]["name"] == "sprite-generator"

    def test_has_version(self):
        data = tomllib.loads(PYPROJECT.read_text())
        assert data["project"]["version"] == "0.1.0"

    def test_has_python_version(self):
        data = tomllib.loads(PYPROJECT.read_text())
        assert data["project"]["requires-python"] == ">=3.10"

    def test_has_description(self):
        data = tomllib.loads(PYPROJECT.read_text())
        desc = data["project"]["description"]
        assert len(desc) > 10

    def test_has_required_fields(self):
        data = tomllib.loads(PYPROJECT.read_text())
        required = ["name", "version", "description", "requires-python", "dependencies"]
        for field in required:
            assert field in data["project"], f"Missing required field: {field}"

    def test_pyproject_dependencies_match_requirements_txt(self):
        if not REQUIREMENTS.exists():
            return
        req_deps = set()
        for line in REQUIREMENTS.read_text().strip().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                req_deps.add(line.split(";")[0].strip())

        data = tomllib.loads(PYPROJECT.read_text())
        pyproject_deps = set(data["project"]["dependencies"])

        missing = req_deps - pyproject_deps
        extra = pyproject_deps - req_deps
        assert not missing, f"Dependencies in requirements.txt but missing from pyproject.toml: {missing}"
        assert not extra, f"Dependencies in pyproject.toml but missing from requirements.txt: {extra}"

    def test_build_system_defined(self):
        data = tomllib.loads(PYPROJECT.read_text())
        assert "build-system" in data
        assert "requires" in data["build-system"]
        assert "setuptools" in str(data["build-system"]["requires"])

    def test_has_urls(self):
        data = tomllib.loads(PYPROJECT.read_text())
        assert "urls" in data["project"]
        assert "Homepage" in data["project"]["urls"]
        assert "Repository" in data["project"]["urls"]


class TestProjectDirectories:
    def test_models_package_init(self):
        assert (ROOT / "models" / "__init__.py").exists()

    def test_vqvae_init(self):
        assert (ROOT / "models" / "vqvae" / "__init__.py").exists()

    def test_transformer_init(self):
        assert (ROOT / "models" / "transformer" / "__init__.py").exists()

    def test_tests_init(self):
        assert (ROOT / "tests" / "__init__.py").exists()

    def test_readme_exists(self):
        assert (ROOT / "README.md").exists()

    def test_requirements_txt_exists(self):
        assert (ROOT / "requirements.txt").exists()
