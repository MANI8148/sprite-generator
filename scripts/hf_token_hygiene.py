"""Automated confirmation that no API tokens are committed to the repository.

Roadmap Phase 0 item: "Rotate/confirm the HF token issue from earlier is
resolved (if not already done)".

Rotating a leaked token is a manual ops step; this module turns the "confirm
it is resolved" half of the item into a repeatable check so the issue cannot
silently regress:

* Scans every *tracked* file (via ``git ls-files``) for hardcoded credential
  patterns (Hugging Face, GitHub, OpenAI, Stripe live, AWS, Slack, Google
  API keys, private keys, JWTs).
* Verifies every GitHub Actions workflow only references tokens through
  ``${{ secrets.* }}`` and never inlines a literal value.
* Verifies ``.env`` / ``kaggle.json`` are gitignored so credentials cannot be
  accidentally staged.

Usage:
    python -m scripts.hf_token_hygiene [--root PATH] [--json]

Exit code 0 = clean, 1 = one or more findings.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Pattern, Tuple

SECRET_PATTERNS: List[Tuple[str, Pattern[str]]] = [
    ("HuggingFace token", re.compile(r"hf_[A-Za-z0-9]{20,}")),
    ("GitHub PAT", re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}\b")),
    ("GitHub fine-grained PAT", re.compile(r"github_pat_[A-Za-z0-9_]{40,}")),
    ("OpenAI key", re.compile(r"\bsk-[A-Za-z0-9]{20,}\b")),
    ("Stripe live key", re.compile(r"\b(?:sk|rk)_live_[A-Za-z0-9]{16,}\b")),
    ("AWS access key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("Slack token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{20,}\b")),
    ("Google API key", re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b")),
    ("Private key block", re.compile(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----")),
    ("JWT", re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b")),
]

GITIGNORE_REQUIRED = [".env", "kaggle.json"]

_EXCLUDED_DIRS = {
    ".git",
    "node_modules",
    ".next",
    "__pycache__",
    ".venv",
    "venv",
    "htmlcov",
    "dist",
    "build",
}

Finding = Tuple[str, str, int, str]


def _run_git(root: Path, *args: str) -> Optional[List[str]]:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *args],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.splitlines()


def iter_tracked_files(root: Path) -> List[Path]:
    """Return tracked files under ``root``.

    Prefers ``git ls-files`` so untracked/local credentials and VCS internals
    are never scanned. Falls back to a manual walk for non-git trees.
    """
    files = _run_git(root, "ls-files")
    if files is not None:
        paths = [root / f for f in files]
        return [p for p in paths if p.is_file()]

    tracked: List[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in _EXCLUDED_DIRS for part in path.relative_to(root).parts):
            continue
        tracked.append(path)
    return tracked


def scan_text(text: str, source: str) -> List[Finding]:
    """Scan ``text`` for credential patterns. Returns (name, source, line, snippet)."""
    findings: List[Finding] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        for name, pattern in SECRET_PATTERNS:
            if pattern.search(line):
                findings.append((name, source, line_no, line.strip()[:120]))
    return findings


def scan_file(path: Path) -> List[Finding]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    return scan_text(text, str(path))


_SECRET_KEYWORD = re.compile(
    r"(TOKEN|PASSWORD|SECRET|PRIVATE[_ ]*KEY|ACCESS[_A-Z0-9]*KEY|API[_A-Z0-9]*KEY)",
    re.IGNORECASE,
)
_ASSIGN = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s*[:=]\s*(.*)$")


def check_workflows(root: Path) -> List[Finding]:
    """Ensure workflow files only reference secrets via ``${{ secrets.* }}``.

    Three rules:
    * Any reference containing ``secrets.`` must be wrapped in ``${{ ... }}``.
    * A quoted string literal assigned to a secret-named variable (e.g.
      ``HF_TOKEN: '...'``) is flagged unless it is a ``${{ ... }}`` expression.
    * A value matching a credential pattern is always flagged (defense in
      depth on top of the file-wide scan).
    """
    findings: List[Finding] = []
    workflows_dir = root / ".github" / "workflows"
    if not workflows_dir.is_dir():
        return findings
    for wf in sorted(workflows_dir.glob("*.yml")):
        text = wf.read_text(encoding="utf-8", errors="replace")
        for line_no, line in enumerate(text.splitlines(), start=1):
            if "secrets." in line:
                if "${{" not in line or "}}" not in line:
                    findings.append(
                        (
                            "Workflow secret not wrapped in ${{ ... }}",
                            str(wf),
                            line_no,
                            line.strip()[:120],
                        )
                    )
                continue
            match = _ASSIGN.match(line.strip())
            if not match or not _SECRET_KEYWORD.search(match.group(1)):
                continue
            value = match.group(2).strip().rstrip(",")
            if not value or value.startswith("${{"):
                continue
            if value.startswith(("'", '"')) or any(
                p.search(value) for _, p in SECRET_PATTERNS
            ):
                findings.append(
                    (
                        "Inline secret literal in workflow",
                        str(wf),
                        line_no,
                        line.strip()[:120],
                    )
                )
    return findings


def check_gitignore(root: Path) -> List[Finding]:
    """Verify credential files are gitignored so they cannot be committed."""
    findings: List[Finding] = []
    gitignore = root / ".gitignore"
    if not gitignore.is_file():
        return findings
    text = gitignore.read_text(encoding="utf-8", errors="replace")
    for entry in GITIGNORE_REQUIRED:
        if not any(line.strip() == entry for line in text.splitlines()):
            findings.append(
                (
                    "Missing gitignore entry",
                    str(gitignore),
                    0,
                    f"{entry} is not ignored",
                )
            )
    return findings


def scan_repository(root: Path) -> List[Finding]:
    findings: List[Finding] = []
    for path in iter_tracked_files(root):
        findings.extend(scan_file(path))
    findings.extend(check_workflows(root))
    findings.extend(check_gitignore(root))
    return findings


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check the repository for committed API tokens."
    )
    parser.add_argument("--root", default=".", help="Repository root (default: .)")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit findings as JSON instead of human-readable text.",
    )
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    findings = scan_repository(root)

    if args.json:
        print(
            json.dumps(
                [
                    {"type": f[0], "file": f[1], "line": f[2], "snippet": f[3]}
                    for f in findings
                ],
                indent=2,
            )
        )
    else:
        if not findings:
            print("OK: no hardcoded credentials found in tracked files.")
        for kind, source, line, snippet in findings:
            loc = source if line == 0 else f"{source}:{line}"
            print(f"[{kind}] {loc}: {snippet}")

    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
