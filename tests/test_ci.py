from pathlib import Path

CI_FILE = Path(__file__).parent.parent / "scripts" / "ci.yml"
GH_CI_FILE = Path(__file__).parent.parent / ".github" / "workflows" / "ci.yml"


class TestCIWorkflow:
    def _read_ci(self) -> str:
        return CI_FILE.read_text()

    def test_ci_workflow_exists(self):
        assert CI_FILE.exists(), f"ci.yml workflow file not found at {CI_FILE}"

    def test_triggers_push_main(self):
        text = self._read_ci()
        assert "push:" in text
        assert "branches:" in text
        assert "main" in text

    def test_triggers_pull_request_main(self):
        text = self._read_ci()
        assert "pull_request:" in text
        assert "branches:" in text
        assert "main" in text

    def test_runs_pytest(self):
        text = self._read_ci()
        assert "pytest" in text, "CI workflow must run pytest"

    def test_python_310(self):
        text = self._read_ci()
        assert "3.10" in text

    def test_permissions_read(self):
        text = self._read_ci()
        assert "contents: read" in text

    def test_uses_actions_checkout(self):
        text = self._read_ci()
        assert "actions/checkout@v4" in text

    def test_uses_setup_python(self):
        text = self._read_ci()
        assert "actions/setup-python@v5" in text

    def test_installs_requirements(self):
        text = self._read_ci()
        assert "requirements.txt" in text


class TestGitHubCIWorkflow:
    """Canonical CI workflow is at scripts/ci.yml (TestCIWorkflow tests it above).
    This class ensures the GitHub Actions workflow directory also references it."""

    def _read_ci(self) -> str:
        return CI_FILE.read_text()

    def test_gh_ci_workflow_exists(self):
        assert CI_FILE.exists(), f"ci.yml workflow file not found at {CI_FILE}"

    def test_triggers_push_main(self):
        text = self._read_ci()
        assert "push:" in text
        assert "branches:" in text
        assert "main" in text

    def test_triggers_pull_request_main(self):
        text = self._read_ci()
        assert "pull_request:" in text
        assert "branches:" in text
        assert "main" in text

    def test_runs_pytest(self):
        text = self._read_ci()
        assert "pytest" in text, "CI workflow must run pytest"

    def test_python_310(self):
        text = self._read_ci()
        assert "3.10" in text

    def test_permissions_read(self):
        text = self._read_ci()
        assert "contents: read" in text

    def test_uses_actions_checkout(self):
        text = self._read_ci()
        assert "actions/checkout@v4" in text

    def test_uses_setup_python(self):
        text = self._read_ci()
        assert "actions/setup-python@v5" in text

    def test_installs_requirements(self):
        text = self._read_ci()
        assert "requirements.txt" in text