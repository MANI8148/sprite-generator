"""
Download CC0 sprite packs from direct-download URLs (Itch.io, GitHub, etc).
Extracts PNG/JPG/GIF/BMP images from downloaded ZIP archives.

Use --bundle-urls to pass URLs on the command line, or add them to
KNOWN_CC0_BUNDLES for persistent inclusion.
"""
import os
import sys
import io
import zipfile
import argparse
from pathlib import Path

import requests
from tqdm import tqdm


# Known CC0 sprite bundles with direct download URLs
# Add more as discovered
KNOWN_CC0_BUNDLES = [
    # Format: (name, download_url)
    # GitHub-hosted CC0 sprite repositories (archive ZIPs are stable direct-download URLs)
    ("pixelart-icons", "https://github.com/tstamborski/pixelart-icons/archive/refs/heads/main.zip"),
    ("cavalier-sprites", "https://github.com/vllsystems/cavalier-sprite-pack/archive/refs/heads/main.zip"),
    ("project-cordon-sprites", "https://github.com/doficia/project-cordon-sprites/archive/refs/heads/main.zip"),
]

# Kenney's assets on Itch.io mirror Kenney.nl packs (same content)
# Use scrape_sources.py for Kenney content instead


def download_bundle(name: str, url: str, output_dir: Path) -> int:
    """Download a ZIP bundle and extract PNGs. Returns number of images saved."""
    try:
        resp = requests.get(url, timeout=120, headers={"User-Agent": "Mozilla/5.0"}, allow_redirects=True)
        if resp.status_code != 200:
            print(f"  {name}: HTTP {resp.status_code}")
            return 0
        if "html" in resp.headers.get("Content-Type", "").lower():
            print(f"  {name}: URL returned HTML (not a ZIP) — may need manual download")
            return 0
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            count = 0
            for zip_name in zf.namelist():
                ext = os.path.splitext(zip_name)[1].lower()
                if ext in (".png", ".jpg", ".gif", ".bmp"):
                    dest = output_dir / name / Path(zip_name).name
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(zip_name) as src, open(dest, "wb") as dst:
                        dst.write(src.read())
                    count += 1
        return count
    except zipfile.BadZipFile:
        print(f"  {name}: not a valid ZIP archive")
        return 0
    except requests.RequestException as e:
        print(f"  {name}: network error — {e}")
        return 0
    except Exception as e:
        print(f"  {name}: unexpected error — {e}")
        return 0


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Download CC0 sprite packs from Itch.io")
    parser.add_argument("--output", "-o", default="data/raw", help="Output directory")
    parser.add_argument("--bundle-urls", nargs="+", default=None,
                        help="Specific bundle URLs (overrides known list)")
    args = parser.parse_args(argv)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    bundles = []
    if args.bundle_urls:
        for i, url in enumerate(args.bundle_urls):
            bundles.append((f"itchio_bundle_{i:03d}", url))
    else:
        bundles = KNOWN_CC0_BUNDLES

    if not bundles:
        print("No CC0 bundles configured.")
        print("Add download URLs to KNOWN_CC0_BUNDLES in this script")
        print("or pass them via --bundle-urls.")
        return 0

    total = 0
    for name, url in tqdm(bundles, desc="Downloading"):
        count = download_bundle(name, url, output_dir)
        if count > 0:
            print(f"  {name}: {count} images")
        total += count

    print(f"\nDownloaded {total} images to {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
