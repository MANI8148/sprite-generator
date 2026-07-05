"""
Download sprite packs from Kenney.nl (CC0) and prepare raw dataset.
Primary source: Kenney.nl GitHub releases (all CC0).
"""
import os
import sys
import zipfile
import io
import argparse
import requests
from pathlib import Path

KENNEY_BASE = "https://kenney.nl/assets"

# Kenney sprite packs that are character/spritesheet focused (CC0)
PACKS = [
    "platformer-art-deluxe",
    "platformer-pack-redux-360-assets",
    "platformer-art-pixel-adventure-1",
    "platformer-art-pixel-adventure-2",
    "top-down-tanks-redux",
    "top-down-shooter",
    "top-down-vehicles-pack-1",
    "rpg-base-pack",
    "rpg-audio-pack",
    "fantasy-characters-pixel-16x16",
    "fantasy-rpg-items",
    "fantasy-ui-borders",
    "game-icons",
    "game-icons-redux",
    "abstract-platformer",
    "animals-redux",
    "animal-pack-redux",
    "alphabot",
    "alien-ufo-pack",
    "battlebots",
    "blocky-characters",
    "bobble-heads",
    "castle-platformer",
    "cat-pack",
    "cavernas",
    "chicken",
    "christmas-pack",
    "city-kit",
    "coffee-pack",
    "cowboy",
    "crosshairs-pack",
    "cute-characters-pixelart",
    "cute-fantasy-pixel-art",
    "dog-pack",
    "dungeon-pack",
    "dungeon-redux",
    "emoji-pack",
    "fps-weapons",
    "furniture-pack",
    "ghost-pack",
    "golf-pack",
    "halloween-pack",
    "hobbies-pack",
    "holiday-pack",
    "impact-sounds",
    "interior-pack",
    "kitty-pack",
    "lego-pack-1",
    "lego-pack-2",
    "m1-pixel-heroes",
    "m1-pixel-monsters",
    "m1-pixel-vehicles",
    "m1-pixel-weapons",
    "mars-platformer",
    "medieval-fantasy-pack",
    "minidungeon-pixel-art",
    "monster-pack",
    "music-pack",
    "nijis-pixel-characters",
    "nijis-pixel-rpg",
    "nijis-platformer",
    "office-pack",
    "pixel-bit-pack",
    "pixel-food-pack",
    "pixel-monsters",
    "pixel-platformer",
    "pixel-rpg",
    "pixel-shooter-1",
    "pixel-shooter-2",
    "pixel-vehicles",
    "platformer-character-pack-1",
    "platformer-character-pack-2",
    "platformer-kit",
    "politics-pack",
    "post-apocalyptic-pack",
    "puzzle-pack-1",
    "puzzle-pack-2",
    "retro-gaming-pack",
    "retro-platformer",
    "rpg-characters-pixel-16x16",
    "rpg-characters-pixel-32x32",
    "rpg-world-pack",
    "sci-fi-pack",
    "sci-fi-platformer",
    "skate-pack",
    "smileys-pack",
    "space-kit",
    "space-pack",
    "space-shooter-redux",
    "spino-pixel-adventure",
    "sports-pack",
    "top-down-fantasy",
    "top-down-roguelike",
    "travel-pack",
    "ui-pack",
    "ukraine-pack",
    "vector-pack",
    "western-pack",
    "whitemeat-pixel-characters",
    "wild-west-pack",
    "winter-platformer",
    "wizards-pack",
    "xmas-platformer",
]


def download_pack(pack_name: str, output_dir: Path) -> int:
    url = f"https://kenney.nl/assets/{pack_name}"
    zip_url = f"https://kenney.nl/Content/Downloads/{pack_name}.zip"

    try:
        resp = requests.get(zip_url, timeout=30)
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
