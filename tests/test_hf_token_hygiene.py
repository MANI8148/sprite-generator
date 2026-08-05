"""Tests for scripts/hf_token_hygiene.py (roadmap Phase 0 item: "Rotate/confirm
the HF token issue from earlier is resolved").

The scanner is strict: it scans every tracked file. To keep the test module
itself clean, fake credentials are constructed at runtime via concatenation so
no full literal secret ever appears in source.
"""
from pathlib import Path

from scripts.hf_token_hygiene import (
    GITIGNORE_REQUIRED,
    SECRET_PATTERNS,
    check_gitignore,
    check_workflows,
    iter_tracked_files,
    main,
    scan_file,
    scan_repository,
    scan_text,
)

REPO_ROOT = Path(__file__).parent.parent


def _fake(prefix: str, length: int) -> str:
    return prefix + "A" * length


class TestSecretPatterns:
    def test_detects_hf_token(self):
        assert SECRET_PATTERNS[0][1].search(_fake("hf_", 30))

    def test_detects_github_pat(self):
        token = _fake("ghp_", 36)
        assert any(p.search(token) for _, p in SECRET_PATTERNS)

    def test_detects_github_fine_grained_pat(self):
        token = _fake("github_pat_", 50)
        assert any(p.search(token) for _, p in SECRET_PATTERNS)

    def test_detects_openai_key(self):
        assert any(p.search(_fake("sk-", 30)) for _, p in SECRET_PATTERNS)

    def test_detects_stripe_live_key(self):
        assert any(p.search(_fake("sk_live_", 20)) for _, p in SECRET_PATTERNS)

    def test_ignores_stripe_test_key(self):
        assert not any(p.search("sk_test_abc123") for _, p in SECRET_PATTERNS)

    def test_detects_aws_access_key(self):
        token = "AKIA" + "0" * 16
        assert any(p.search(token) for _, p in SECRET_PATTERNS)

    def test_detects_slack_token(self):
        assert any(p.search(_fake("xoxb-", 20)) for _, p in SECRET_PATTERNS)

    def test_detects_google_api_key(self):
        token = "AIza" + "F" * 35
        assert any(p.search(token) for _, p in SECRET_PATTERNS)

    def test_detects_private_key_block(self):
        header = "-----BEGIN " + "PRIVATE KEY-----"
        assert any(p.search(header) for _, p in SECRET_PATTERNS)

    def test_detects_jwt(self):
        jwt = "eyJ" + "a" * 12 + "." + "b" * 12 + "." + "c" * 12
        assert any(p.search(jwt) for _, p in SECRET_PATTERNS)

    def test_ignores_innocuous_strings(self):
        for s in ["hf_", "sk_test_abc", "secret-123", "kaggle.json", "HF_TOKEN"]:
            assert not any(p.search(s) for _, p in SECRET_PATTERNS), s


class TestScanText:
    def test_scan_text_finds_secret_with_line_number(self):
        text = "password=123\n" + "token = " + _fake("hf_", 30) + "\n"
        findings = scan_text(text, "sample.txt")
        assert len(findings) == 1
        kind, source, line, _snippet = findings[0]
        assert kind == "HuggingFace token"
        assert source == "sample.txt"
        assert line == 2

    def test_scan_text_clean_text_no_findings(self):
        text = "api_key = os.environ['HF_TOKEN']\nprint('hi')\n"
        assert scan_text(text, "clean.py") == []

    def test_scan_text_reports_both_hits(self):
        text = _fake("ghp_", 36) + "\n" + _fake("sk_live_", 20) + "\n"
        findings = scan_text(text, "two.txt")
        kinds = {f[0] for f in findings}
        assert "GitHub PAT" in kinds
        assert "Stripe live key" in kinds


class TestScanFile:
    def test_scan_file_reports_findings(self, tmp_path):
        target = tmp_path / "config.py"
        target.write_text(_fake("hf_", 30))
        findings = scan_file(target)
        assert len(findings) == 1
        assert findings[0][1] == str(target)

    def test_scan_file_missing_file(self):
        assert scan_file(Path("/nonexistent/file.py")) == []


class TestWorkflowChecks:
    def test_workflow_with_wrapped_secret_is_clean(self, tmp_path):
        wf_dir = tmp_path / ".github" / "workflows"
        wf_dir.mkdir(parents=True)
        (wf_dir / "ci.yml").write_text(
            "env:\n  HF_TOKEN: ${{ secrets.HF_TOKEN }}\n"
        )
        assert check_workflows(tmp_path) == []

    def test_workflow_with_inline_secret_is_flagged(self, tmp_path):
        wf_dir = tmp_path / ".github" / "workflows"
        wf_dir.mkdir(parents=True)
        (wf_dir / "ci.yml").write_text("env:\n  HF_TOKEN: 'hf_inline_secret_here'\n")
        findings = check_workflows(tmp_path)
        assert len(findings) == 1
        assert findings[0][0] == "Inline secret literal in workflow"

    def test_workflow_with_credential_value_is_flagged(self, tmp_path):
        wf_dir = tmp_path / ".github" / "workflows"
        wf_dir.mkdir(parents=True)
        (wf_dir / "ci.yml").write_text(
            "env:\n  HF_TOKEN: " + _fake("hf_", 30) + "\n"
        )
        findings = check_workflows(tmp_path)
        assert len(findings) == 1

    def test_workflow_with_code_variable_is_clean(self, tmp_path):
        wf_dir = tmp_path / ".github" / "workflows"
        wf_dir.mkdir(parents=True)
        (wf_dir / "ci.yml").write_text(
            "run: |\n  token = token,\n  api.upload_file(token=token)\n"
        )
        assert check_workflows(tmp_path) == []

    def test_workflow_dir_missing_is_clean(self, tmp_path):
        assert check_workflows(tmp_path) == []


class TestGitignoreChecks:
    def test_gitignore_covers_required_entries(self, tmp_path):
        (tmp_path / ".gitignore").write_text(".env\nkaggle.json\n")
        assert check_gitignore(tmp_path) == []

    def test_gitignore_missing_entry_is_flagged(self, tmp_path):
        (tmp_path / ".gitignore").write_text("*.pyc\n")
        findings = check_gitignore(tmp_path)
        names = {f[0] for f in findings}
        assert "Missing gitignore entry" in names

    def test_gitignore_required_entries_sane(self):
        assert ".env" in GITIGNORE_REQUIRED
        assert "kaggle.json" in GITIGNORE_REQUIRED


class TestRepositoryScan:
    def test_iter_tracked_files_returns_repo_files(self):
        files = iter_tracked_files(REPO_ROOT)
        assert files, "repo should contain tracked files"
        assert any(p.name == "deploy_spaces.py" for p in files)
        assert any(p.name == "pyproject.toml" for p in files)

    def test_real_repo_has_no_findings(self):
        assert scan_repository(REPO_ROOT) == []

    def test_main_clean_repo_exit_zero(self):
        assert main(["--root", str(REPO_ROOT)]) == 0

    def test_main_dirty_repo_exit_one(self, tmp_path):
        (tmp_path / ".gitignore").write_text("*.pyc\n")
        (tmp_path / "settings.py").write_text(_fake("hf_", 30))
        assert main(["--root", str(tmp_path)]) == 1

    def test_main_json_output(self, tmp_path, capsys):
        (tmp_path / "settings.py").write_text(_fake("hf_", 30))
        rc = main(["--root", str(tmp_path), "--json"])
        captured = capsys.readouterr()
        assert rc == 1
        assert '"type": "HuggingFace token"' in captured.out


class TestCIIntegration:
    def test_ci_runs_hygiene_check(self):
        ci = (REPO_ROOT / "scripts" / "ci.yml").read_text()
        assert "hf_token_hygiene" in ci
        assert "python -m scripts.hf_token_hygiene" in ci
