"""Deploy gradio_app/ and backend/ to Hugging Face Spaces."""
import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Directories that are never needed at Space runtime and only bloat the upload.
IGNORED_DIRS = {"__pycache__", ".pytest_cache", ".ipynb_checkpoints", ".git"}
# File suffixes that are never needed at Space runtime.
IGNORED_SUFFIXES = {".pyc", ".pyo"}
# Other editor / OS noise files that must not be uploaded.
IGNORED_FILENAMES = {".DS_Store", "Thumbs.db"}


def should_skip(local_path: Path) -> bool:
    """Return True for files that must not be uploaded to a Space.

    Python bytecode caches (``__pycache__``, ``*.pyc``/``*.pyo``), test caches
    and editor/OS noise are not needed by the running app and would pollute
    the HF Spaces repo, so the deployment script excludes them.
    """
    if local_path.name in IGNORED_FILENAMES:
        return True
    if local_path.suffix in IGNORED_SUFFIXES:
        return True
    return any(part in IGNORED_DIRS for part in local_path.parts)


def collect_files(source_dirs: list[Path], flatten_dirs: set[str] | None = None) -> list[tuple[str, str]]:
    """Collect (local_path, repo_path) pairs for upload.

    Args:
        source_dirs: Directories to walk for files.
        flatten_dirs: Set of source dir names whose repo_path should be
                      relative to themselves (for HF Spaces app root).
                      E.g. flatten_dirs={"gradio_app"} means gradio_app/app.py
                      is uploaded as app.py, not gradio_app/app.py.
    """
    if flatten_dirs is None:
        flatten_dirs = set()
    files = []
    for src_dir in source_dirs:
        src_dir = src_dir.resolve()
        if not src_dir.exists():
            print(f"Warning: {src_dir} does not exist, skipping", file=sys.stderr)
            continue
        for root, _dirs, fnames in os.walk(str(src_dir)):
            root_path = Path(root)
            for fname in fnames:
                local_path = root_path / fname
                if should_skip(local_path):
                    continue
                if src_dir.name in flatten_dirs:
                    repo_path = str(local_path.relative_to(src_dir))
                else:
                    repo_path = str(local_path.relative_to(REPO_ROOT))
                files.append((str(local_path), repo_path))
    return files


def deploy(
    space_repo: str,
    hf_token: str,
    source_dirs: list[Path],
    dry_run: bool = False,
    flatten_dirs: set[str] | None = None,
) -> int:
    """Upload files to HF Spaces. Returns 0 on success, 1 on failure."""
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("Error: huggingface_hub not installed. Run: pip install huggingface_hub", file=sys.stderr)
        return 1

    files = collect_files(source_dirs, flatten_dirs=flatten_dirs)
    if not files:
        print("Error: no files found to deploy", file=sys.stderr)
        return 1

    api = HfApi()

    for local_path, repo_path in files:
        if dry_run:
            print(f"[DRY RUN] Would upload: {local_path} -> {repo_path}")
        else:
            try:
                api.upload_file(
                    path_or_fileobj=local_path,
                    path_in_repo=repo_path,
                    repo_id=space_repo,
                    repo_type="space",
                    token=hf_token,
                )
                print(f"Uploaded {repo_path}")
            except Exception as e:
                print(f"Failed to upload {repo_path}: {e}", file=sys.stderr)
                return 1

    print("Deploy complete")
    return 0


# The API Space is deployed as a Docker SDK space: the FastAPI backend is
# copied under backend/ and the space config (Dockerfile + README.md) lives in
# spaces/api/, flattened to the Space root so the Docker build finds them.
API_SPACE_DIR = REPO_ROOT / "spaces" / "api"


def collect_api_space_files() -> list[tuple[str, str]]:
    """Collect (local_path, repo_path) pairs for the FastAPI API Space.

    The backend package is uploaded preserving its ``backend/`` prefix so the
    Space's ``backend.main:app`` import path works, the ``spaces/api/`` config
    is flattened to the Space root (Dockerfile, README.md), and the repository
    ``requirements.txt`` is placed at the Space root for the Docker build.
    """
    files = collect_files(
        [
            REPO_ROOT / "backend",
            API_SPACE_DIR,
        ],
        flatten_dirs={"api"},
    )
    requirements = REPO_ROOT / "requirements.txt"
    if requirements.is_file():
        files.append((str(requirements), "requirements.txt"))
    return files


def deploy_api_space(
    space_repo: str,
    hf_token: str,
    dry_run: bool = False,
) -> int:
    """Upload the FastAPI model+API Space. Returns 0 on success, 1 on failure."""
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("Error: huggingface_hub not installed. Run: pip install huggingface_hub", file=sys.stderr)
        return 1

    files = collect_api_space_files()
    if not files:
        print("Error: no files found to deploy", file=sys.stderr)
        return 1

    api = HfApi()

    for local_path, repo_path in files:
        if dry_run:
            print(f"[DRY RUN] Would upload: {local_path} -> {repo_path}")
        else:
            try:
                api.upload_file(
                    path_or_fileobj=local_path,
                    path_in_repo=repo_path,
                    repo_id=space_repo,
                    repo_type="space",
                    token=hf_token,
                )
                print(f"Uploaded {repo_path}")
            except Exception as e:
                print(f"Failed to upload {repo_path}: {e}", file=sys.stderr)
                return 1

    print("Deploy complete")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Deploy to Hugging Face Spaces")
    parser.add_argument("--space-repo", default=os.environ.get("HF_SPACE_REPO", "darklord8777/sprite-generator-demo"))
    parser.add_argument("--api", action="store_true", help="Deploy the FastAPI model+API Space (Docker SDK) instead of the Gradio demo")
    parser.add_argument("--api-space-repo", default=os.environ.get("HF_API_SPACE_REPO", "darklord8777/sprite-generator-api"))
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN", ""))
    parser.add_argument("--dry-run", action="store_true", help="Print files without uploading")
    args = parser.parse_args()

    if not args.token and not args.dry_run:
        print("Error: HF_TOKEN not set. Use --token or set HF_TOKEN env var.", file=sys.stderr)
        return 1

    if args.api:
        return deploy_api_space(
            args.api_space_repo,
            args.token,
            dry_run=args.dry_run,
        )

    source_dirs = [
        REPO_ROOT / "gradio_app",
        REPO_ROOT / "backend",
    ]

    return deploy(
        args.space_repo,
        args.token,
        source_dirs,
        dry_run=args.dry_run,
        flatten_dirs={"gradio_app"},
    )


if __name__ == "__main__":
    sys.exit(main())
