"""Deploy gradio_app/ and backend/ to Hugging Face Spaces."""
import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent


def collect_files(source_dirs: list[Path]) -> list[tuple[str, str]]:
    """Collect (local_path, repo_path) pairs for upload."""
    files = []
    for src_dir in source_dirs:
        if not src_dir.exists():
            print(f"Warning: {src_dir} does not exist, skipping", file=sys.stderr)
            continue
        for root, _dirs, fnames in os.walk(str(src_dir)):
            root_path = Path(root)
            for fname in fnames:
                local_path = root_path / fname
                repo_path = str(local_path.relative_to(REPO_ROOT))
                files.append((str(local_path), repo_path))
    return files


def deploy(
    space_repo: str,
    hf_token: str,
    source_dirs: list[Path],
    dry_run: bool = False,
) -> int:
    """Upload files to HF Spaces. Returns 0 on success, 1 on failure."""
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("Error: huggingface_hub not installed. Run: pip install huggingface_hub", file=sys.stderr)
        return 1

    files = collect_files(source_dirs)
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
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN", ""))
    parser.add_argument("--dry-run", action="store_true", help="Print files without uploading")
    args = parser.parse_args()

    if not args.token and not args.dry_run:
        print("Error: HF_TOKEN not set. Use --token or set HF_TOKEN env var.", file=sys.stderr)
        return 1

    source_dirs = [
        REPO_ROOT / "gradio_app",
        REPO_ROOT / "backend",
    ]

    return deploy(args.space_repo, args.token, source_dirs, dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
