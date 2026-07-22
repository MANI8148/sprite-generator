from pathlib import Path

CI_FILE = Path(__file__).parent.parent / ".github" / "workflows" / "ci.yml"


class TestCIWorkflow:
    def test_workflow_exists(self):
        assert CI_FILE.exists(), "ci.yml workflow file not found"

    def test_triggers_push_main(self):
        text = CI_FILE.read_text()
        assert "push:" in text
        assert "branches:" in text
        assert "main" in text

    def test_triggers_pull_request_main(self):
        text = CI_FILE.read_text()
        assert "pull_request:" in text
        assert "branches:" in text
        assert "main" in text

    def test_runs_pytest(self):
        text = CI_FILE.read_text()
        assert "pytest" in text, "CI workflow must run pytest"

    def test_python_310(self):
        text = CI_FILE.read_text()
        assert "3.10" in text

    def test_permissions_read(self):
        text = CI_FILE.read_text()
        assert "contents: read" in text

    def test_uses_actions_checkout(self):
        text = CI_FILE.read_text()
        assert "actions/checkout@v4" in text

    def test_uses_setup_python(self):
        text = CI_FILE.read_text()
        assert "actions/setup-python@v5" in text

    def test_installs_requirements(self):
        text = CI_FILE.read_text()
        assert "requirements.txt" in text