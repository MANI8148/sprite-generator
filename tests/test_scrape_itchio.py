"""
Tests for scrape_itchio.py (roadmap item #1 — data pipeline).
Covers KNOWN_CC0_BUNDLES population, download_bundle, and main entrypoint.
"""
import io
import zipfile
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from data.scripts.scrape_itchio import (KNOWN_CC0_BUNDLES, download_bundle, main)


class TestKnownBundles:
    def test_bundles_are_populated(self):
        assert len(KNOWN_CC0_BUNDLES) > 0, "KNOWN_CC0_BUNDLES must not be empty"

    def test_bundles_have_name_and_url(self):
        for name, url in KNOWN_CC0_BUNDLES:
            assert isinstance(name, str) and len(name) > 0
            assert isinstance(url, str) and url.startswith("http")

    def test_bundles_have_unique_names(self):
        names = [n for n, _ in KNOWN_CC0_BUNDLES]
        assert len(names) == len(set(names)), "bundle names must be unique"


class TestDownloadBundle:
    def make_zip(self, files: list[tuple[str, bytes]]) -> bytes:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            for path, content in files:
                zf.writestr(path, content)
        return buf.getvalue()

    @patch("data.scripts.scrape_itchio.requests.get")
    def test_extracts_png_files(self, mock_get):
        zip_data = self.make_zip([
            ("sprite1.png", b"PNG content 1"),
            ("sprite2.png", b"PNG content 2"),
            ("readme.txt", b"not an image"),
        ])
        mock_get.return_value = MagicMock(status_code=200, content=zip_data,
                                          headers={"Content-Type": "application/zip"})

        with tempfile.TemporaryDirectory() as tmp:
            count = download_bundle("test_pack", "https://example.com/pack.zip", Path(tmp))
            assert count == 2
            assert (Path(tmp) / "test_pack" / "sprite1.png").exists()
            assert (Path(tmp) / "test_pack" / "sprite2.png").exists()
            readme = Path(tmp) / "test_pack" / "readme.txt"
            assert not readme.exists()

    @patch("data.scripts.scrape_itchio.requests.get")
    def test_extracts_jpg_and_gif(self, mock_get):
        zip_data = self.make_zip([
            ("img.jpg", b"JPEG content"),
            ("anim.gif", b"GIF content"),
            ("sprite.bmp", b"BMP content"),
        ])
        mock_get.return_value = MagicMock(status_code=200, content=zip_data,
                                          headers={"Content-Type": "application/zip"})

        with tempfile.TemporaryDirectory() as tmp:
            count = download_bundle("multi_ext", "https://example.com/pack.zip", Path(tmp))
            assert count == 3

    @patch("data.scripts.scrape_itchio.requests.get")
    def test_http_error_returns_zero(self, mock_get):
        mock_get.return_value = MagicMock(status_code=404)
        count = download_bundle("missing", "https://example.com/404.zip",
                                Path("/tmp"))
        assert count == 0

    @patch("data.scripts.scrape_itchio.requests.get")
    def test_html_response_returns_zero(self, mock_get):
        mock_get.return_value = MagicMock(
            status_code=200, content=b"<html>not a zip</html>",
            headers={"Content-Type": "text/html"},
        )
        count = download_bundle("html", "https://example.com/page", Path("/tmp"))
        assert count == 0

    @patch("data.scripts.scrape_itchio.requests.get")
    def test_bad_zip_returns_zero(self, mock_get):
        mock_get.return_value = MagicMock(status_code=200, content=b"not zip data",
                                          headers={"Content-Type": "application/zip"})
        count = download_bundle("bad_zip", "https://example.com/bad.zip",
                                Path("/tmp"))
        assert count == 0

    @patch("data.scripts.scrape_itchio.requests.get")
    def test_network_error_returns_zero(self, mock_get):
        from requests.exceptions import ConnectionError
        mock_get.side_effect = ConnectionError("connection failed")
        count = download_bundle("offline", "https://example.com/pack.zip",
                                Path("/tmp"))
        assert count == 0

    @patch("data.scripts.scrape_itchio.requests.get")
    def test_empty_zip_returns_zero(self, mock_get):
        zip_data = self.make_zip([])
        mock_get.return_value = MagicMock(status_code=200, content=zip_data,
                                          headers={"Content-Type": "application/zip"})

        with tempfile.TemporaryDirectory() as tmp:
            count = download_bundle("empty", "https://example.com/empty.zip", Path(tmp))
            assert count == 0


class TestMainEntryPoint:
    @patch("data.scripts.scrape_itchio.requests.get")
    def test_main_uses_bundle_urls_argument(self, mock_get):
        zip_data = io.BytesIO()
        with zipfile.ZipFile(zip_data, "w") as zf:
            zf.writestr("sprite.png", b"PNG content")
        mock_get.return_value = MagicMock(status_code=200, content=zip_data.getvalue(),
                                          headers={"Content-Type": "application/zip"})

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            ret = main([
                "--output", str(out),
                "--bundle-urls", "https://example.com/pack1.zip",
            ])
            assert ret == 0
            assert mock_get.called

    @patch("data.scripts.scrape_itchio.requests.get")
    def test_main_with_empty_known_bundles(self, mock_get):
        with patch("data.scripts.scrape_itchio.KNOWN_CC0_BUNDLES", []):
            ret = main(["--output", "/tmp/test_out"])
            assert ret == 0
            mock_get.assert_not_called()

    @patch("data.scripts.scrape_itchio.download_bundle")
    @patch("data.scripts.scrape_itchio.KNOWN_CC0_BUNDLES", [
        ("pack1", "https://example.com/pack1.zip"),
        ("pack2", "https://example.com/pack2.zip"),
    ])
    def test_main_iterates_all_known_bundles(self, mock_dl):
        mock_dl.return_value = 5
        with tempfile.TemporaryDirectory() as tmp:
            ret = main(["--output", str(tmp)])
            assert ret == 0
            assert mock_dl.call_count == 2

    def test_main_creates_output_directory(self):
        with patch("data.scripts.scrape_itchio.KNOWN_CC0_BUNDLES", []):
            with tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp) / "new_dir"
                assert not out.exists()
                ret = main(["--output", str(out)])
                assert ret == 0
                assert out.exists()
