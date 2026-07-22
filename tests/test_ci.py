from pathlib import Path

CI_FILE = Path(__file__).parent.parent / ".github" / "workflows" / "ci.yml"
CI_FILE_LEGACY = Path(__file__).parent.parent / "scripts" / "ci.yml"


class TestCIWorkflow:
    def _read_ci(self) -> str:
        if CI_FILE.exists():
            return CI_FILE.read_text()
        return CI_FILE_LEGACY.read_text()

    def test_workflow_exists(self):
        assert CI_FILE.exists() or CI_FILE_LEGACY.exists(), "ci.yml workflow file not found"

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