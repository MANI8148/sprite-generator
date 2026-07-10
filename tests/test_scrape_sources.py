"""
Tests for scrape_sources.py (roadmap item #1 — data pipeline).
Covers PACKS population, download URL scraping, zip extraction, and main entrypoint.
"""
import io
import zipfile
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from data.scripts.scrape_sources import PACKS, get_download_url, download_pack, main


class TestPacks:
    def test_packs_are_populated(self):
        assert len(PACKS) > 0

    def test_packs_have_valid_names(self):
        for name in PACKS:
            assert isinstance(name, str) and len(name) > 0

    def test_packs_are_unique(self):
        assert len(PACKS) == len(set(PACKS))


class TestGetDownloadUrl:
    @patch("data.scripts.scrape_sources.requests.get")
    def test_returns_url_when_zip_found(self, mock_get):
        mock_get.return_value = MagicMock(
            status_code=200,
            text='<a href="/media/pages/assets/some-pack/game.zip">Download</a>',
        )
        url = get_download_url("some-pack")
        assert url == "https://kenney.nl/media/pages/assets/some-pack/game.zip"

    @patch("data.scripts.scrape_sources.requests.get")
    def test_returns_none_when_no_zip(self, mock_get):
        mock_get.return_value = MagicMock(
            status_code=200,
            text="<html>No downloads here</html>",
        )
        url = get_download_url("empty-pack")
        assert url is None

    @patch("data.scripts.scrape_sources.requests.get")
    def test_http_error_returns_none(self, mock_get):
        mock_get.return_value = MagicMock(status_code=404)
        url = get_download_url("missing-pack")
        assert url is None

    @patch("data.scripts.scrape_sources.requests.get")
    def test_network_error_returns_none(self, mock_get):
        from requests.exceptions import ConnectionError
        mock_get.side_effect = ConnectionError("connection failed")
        url = get_download_url("offline-pack")
        assert url is None

    @patch("data.scripts.scrape_sources.requests.get")
    def test_fallback_to_absolute_zip(self, mock_get):
        mock_get.return_value = MagicMock(
            status_code=200,
            text='<a href="https://example.com/pack.zip">Download</a>',
        )
        url = get_download_url("fallback-pack")
        assert url == "https://example.com/pack.zip"

    @patch("data.scripts.scrape_sources.requests.get")
    def test_returns_none_on_exception(self, mock_get):
        mock_get.side_effect = Exception("unexpected error")
        url = get_download_url("error-pack")
        assert url is None


class TestDownloadPack:
    def make_zip(self, files):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            for path, content in files:
                zf.writestr(path, content)
        return buf.getvalue()

    @patch("data.scripts.scrape_sources.get_download_url")
    @patch("data.scripts.scrape_sources.requests.get")
    def test_extracts_png_files(self, mock_get, mock_url):
        mock_url.return_value = "https://example.com/pack.zip"
        zip_data = self.make_zip([
            ("pack/sprite1.png", b"PNG one"),
            ("pack/sprite2.png", b"PNG two"),
            ("pack/readme.txt", b"not an image"),
        ])
        mock_get.return_value = MagicMock(status_code=200, content=zip_data)

        with tempfile.TemporaryDirectory() as tmp:
            count = download_pack("test-pack", Path(tmp))
            assert count == 2
            assert (Path(tmp) / "test-pack" / "sprite1.png").exists()
            assert (Path(tmp) / "test-pack" / "sprite2.png").exists()
            assert not (Path(tmp) / "test-pack" / "readme.txt").exists()

    @patch("data.scripts.scrape_sources.get_download_url")
    @patch("data.scripts.scrape_sources.requests.get")
    def test_extracts_jpg_and_gif(self, mock_get, mock_url):
        mock_url.return_value = "https://example.com/pack.zip"
        zip_data = self.make_zip([
            ("pack/img.jpg", b"JPEG"),
            ("pack/anim.gif", b"GIF"),
            ("pack/icon.bmp", b"BMP"),
        ])
        mock_get.return_value = MagicMock(status_code=200, content=zip_data)

        with tempfile.TemporaryDirectory() as tmp:
            count = download_pack("multi-ext", Path(tmp))
            assert count == 2

    @patch("data.scripts.scrape_sources.get_download_url")
    def test_no_url_returns_zero(self, mock_url):
        mock_url.return_value = None
        count = download_pack("no-url", Path("/tmp"))
        assert count == 0

    @patch("data.scripts.scrape_sources.get_download_url")
    @patch("data.scripts.scrape_sources.requests.get")
    def test_http_error_returns_zero(self, mock_get, mock_url):
        mock_url.return_value = "https://example.com/pack.zip"
        mock_get.return_value = MagicMock(status_code=404)
        count = download_pack("missing", Path("/tmp"))
        assert count == 0

    @patch("data.scripts.scrape_sources.get_download_url")
    @patch("data.scripts.scrape_sources.requests.get")
    def test_bad_zip_returns_zero(self, mock_get, mock_url):
        mock_url.return_value = "https://example.com/bad.zip"
        mock_get.return_value = MagicMock(status_code=200, content=b"not a zip")
        count = download_pack("bad-zip", Path("/tmp"))
        assert count == 0

    @patch("data.scripts.scrape_sources.get_download_url")
    @patch("data.scripts.scrape_sources.requests.get")
    def test_empty_zip_returns_zero(self, mock_get, mock_url):
        mock_url.return_value = "https://example.com/empty.zip"
        zip_data = self.make_zip([])
        mock_get.return_value = MagicMock(status_code=200, content=zip_data)

        with tempfile.TemporaryDirectory() as tmp:
            count = download_pack("empty", Path(tmp))
            assert count == 0

    @patch("data.scripts.scrape_sources.get_download_url")
    @patch("data.scripts.scrape_sources.requests.get")
    def test_network_error_returns_zero(self, mock_get, mock_url):
        from requests.exceptions import ConnectionError
        mock_url.return_value = "https://example.com/pack.zip"
        mock_get.side_effect = ConnectionError("connection failed")
        count = download_pack("offline", Path("/tmp"))
        assert count == 0

    @patch("data.scripts.scrape_sources.get_download_url")
    @patch("data.scripts.scrape_sources.requests.get")
    def test_handles_zip_with_subdirectory(self, mock_get, mock_url):
        mock_url.return_value = "https://example.com/pack.zip"
        zip_data = self.make_zip([
            ("assets/sprites/hero.png", b"PNG hero"),
            ("assets/sprites/enemy.png", b"PNG enemy"),
        ])
        mock_get.return_value = MagicMock(status_code=200, content=zip_data)

        with tempfile.TemporaryDirectory() as tmp:
            count = download_pack("nested", Path(tmp))
            assert count == 2
            assert (Path(tmp) / "nested" / "sprites" / "hero.png").exists()
            assert (Path(tmp) / "nested" / "sprites" / "enemy.png").exists()


class TestMainEntryPoint:
    @patch("data.scripts.scrape_sources.download_pack")
    def test_main_uses_specific_packs(self, mock_dl):
        mock_dl.return_value = 5
        with tempfile.TemporaryDirectory() as tmp:
            ret = main([
                "--output", str(tmp),
                "--packs", "pack-one", "pack-two",
            ])
            assert ret == 0
            assert mock_dl.call_count == 2

    @patch("data.scripts.scrape_sources.download_pack")
    def test_main_uses_default_packs(self, mock_dl):
        mock_dl.return_value = 3
        with patch("data.scripts.scrape_sources.PACKS", ["default-pack"]):
            with tempfile.TemporaryDirectory() as tmp:
                ret = main(["--output", str(tmp)])
                assert ret == 0
                mock_dl.assert_called_once()

    def test_main_creates_output_directory(self):
        with patch("data.scripts.scrape_sources.PACKS", []):
            with tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp) / "new_raw_dir"
                assert not out.exists()
                ret = main(["--output", str(out)])
                assert ret == 0
                assert out.exists()
