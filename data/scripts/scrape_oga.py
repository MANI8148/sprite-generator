"""
Scrape CC0 sprite packs from OpenGameArt.org.
Uses search API to find CC0 content, downloads ZIPs, extracts PNGs.
"""
import os
import sys
import io
import re
import zipfile
import argparse
import time
from pathlib import Path

import requests
from tqdm import tqdm

OGA_BASE = "https://opengameart.org"


def search_cc0_packs(keys: str, max_pages: int = 3) -> list:
    """Search OGA for CC0 (license=9) sprite packs."""
    packs = []
    for page in range(max_pages):
        url = f"{OGA_BASE}/art-search-advanced?keys={keys}&field_art_licenses=9&page={page}&sort_by=created&sort_order=DESC"
        try:
            resp = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
            if resp.status_code != 200:
                break
            # Extract content links
            for m in re.finditer(r'href=[\"\'](/content/[a-z0-9-]+)[\"\']', resp.text):
                name = m.group(1)
                if not any(x in name for x in ["faq", "chat", "comment", "login", "user", "node", "page"]):
                    packs.append(f"{OGA_BASE}{name}")
        except Exception:
            break
    return list(set(packs))


def get_download_urls(content_url: str) -> list:
    """Extract ZIP download URLs from an OGA content page."""
    try:
        resp = requests.get(content_url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
        if resp.status_code != 200:
            return []
        html = resp.text
        # Find ZIP download links
        zips = re.findall(r'href=[\"\']([^\"\']+\.zip)[\"\']', html)
        # Filter to OGA-hosted downloads
        downloads = []
        for z in zips:
            if z.startswith("/"):
                z = OGA_BASE + z
            if "opengameart" in z or "sites/default" in z:
                downloads.append(z)
        return downloads
    except Exception:
        return []


def download_and_extract(zip_url: str, output_dir: Path, pack_name: str) -> int:
    """Download a ZIP from OGA and extract PNGs."""
    try:
        resp = requests.get(zip_url, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
        if resp.status_code != 200:
            return 0
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            count = 0
            for name in zf.namelist():
                ext = os.path.splitext(name)[1].lower()
                if ext in (".png", ".jpg", ".gif"):
                    dest = output_dir / pack_name / Path(name).name
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(name) as src, open(dest, "wb") as dst:
                        dst.write(src.read())
                    count += 1
        return count
    except Exception:
        return 0


def main():
    parser = argparse.ArgumentParser(description="Scrape CC0 sprite packs from OpenGameArt")
    parser.add_argument("--output", "-o", default="data/raw", help="Output directory")
    parser.add_argument("--search", nargs="+", default=["sprite+32x32", "pixel+art+character", "rpg+sprite"],
                        help="Search keywords for OGA")
    parser.add_argument("--max-packs", type=int, default=50, help="Max packs to process")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Search for packs
    all_packs = []
    for kw in args.search:
        packs = search_cc0_packs(kw)
        print(f"Search '{kw}': found {len(packs)} CC0 packs")
        all_packs.extend(packs)

    all_packs = list(set(all_packs))
    print(f"Total unique CC0 packs found: {len(all_packs)}")

    # Download
    total_images = 0
    processed = 0
    for pack_url in tqdm(all_packs[:args.max_packs], desc="Downloading packs"):
        pack_name = pack_url.rstrip("/").split("/")[-1]
        # Skip non-sprite packs by checking title
        if any(skip in pack_name for skip in ["music", "sound", "audio", "font", "voice"]):
            continue

        dl_urls = get_download_urls(pack_url)
        for dl_url in dl_urls:
            count = download_and_extract(dl_url, output_dir, pack_name)
            if count > 0:
                total_images += count
                processed += 1
                break  # One download per pack is enough
        time.sleep(0.5)  # Rate limiting

    print(f"\nDownloaded {total_images} images from {processed} packs to {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
