"""
Download sprite packs from Kenney.nl (CC0) and prepare raw dataset.
Primary source: Kenney.nl GitHub releases (all CC0).
"""
import os
import sys
import zipfile
import io
import re
import argparse
import requests
from pathlib import Path

KENNEY_BASE = "https://kenney.nl/assets"

# Kenney sprite packs that are character/spritesheet focused (CC0)
# Only currently-available packs (verified working as of 2026)
PACKS = [
    # Pixel-art character sprites
    "platformer-art-deluxe",
    "top-down-shooter",
    "platformer-kit",
    "pixel-platformer",
    "abstract-platformer",
    "blocky-characters",
    "alien-ufo-pack",
    "new-platformer-pack",
    "mini-dungeon",
    "cube-pets",
    "tiny-farm",
    # UI / icons / misc sprites
    "fantasy-ui-borders",
    "game-icons",
    "puzzle-pack-1",
    "puzzle-pack-2",
    "space-kit",
    "sports-pack",
    "ui-pack",
    # New packs (2025+ releases)
    "car-kit",
    "factory-kit",
    "flag-pack",
    "graveyard-kit",
    "input-prompts",
    "light-masks",
    "modular-dungeon-kit",
    "modular-space-kit",
    "pirate-kit",
    "retro-textures-fantasy",
    "development-essentials",
]


def get_download_url(pack_name: str) -> str | None:
    """Scrape the direct ZIP download URL from the Kenney asset page."""
    page_url = f"https://kenney.nl/assets/{pack_name}"
    try:
        resp = requests.get(page_url, timeout=30)
        resp.raise_for_status()
        # Find download link from asset page
        matches = re.findall(
            rf'(/media/pages/assets/[^"\']+\.zip)',
            resp.text,
        )
        if matches:
            return "https://kenney.nl" + matches[0]
        # Fallback: find any .zip link
        matches = re.findall(r'(https?://[^"\']+\.zip)', resp.text)
        if matches:
            return matches[0]
        return None
    except Exception:
        return None


def download_pack(pack_name: str, output_dir: Path) -> int:
    zip_url = get_download_url(pack_name)
    if not zip_url:
        return 0

    try:
        resp = requests.get(zip_url, timeout=60)
        if resp.status_code != 200:
            return 0

        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            count = 0
            for name in zf.namelist():
                ext = os.path.splitext(name)[1].lower()
                if ext in (".png", ".jpg", ".gif"):
                    parts = Path(name).parts
                    if len(parts) > 1:
                        relative = Path(*parts[1:])
                    else:
                        relative = Path(name)
                    dest = output_dir / pack_name / relative
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(name) as src, open(dest, "wb") as dst:
                        dst.write(src.read())
                    count += 1

        return count

    except Exception:
        return 0


def main():
    parser = argparse.ArgumentParser(description="Scrape sprite packs from Kenney.nl")
    parser.add_argument("--output", "-o", default="data/raw", help="Output directory")
    parser.add_argument("--packs", nargs="+", help="Specific packs to download (default: all)")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    packs = args.packs if args.packs else PACKS
    total = 0

    for pack in packs:
        count = download_pack(pack, output_dir)
        if count > 0:
            print(f"Downloaded {pack}: {count} images")
        total += count

    print(f"\nTotal: {total} images downloaded to {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
