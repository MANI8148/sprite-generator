"""Tests for Next.js frontend (roadmap: Phase 1 Item 2)."""

from pathlib import Path

ROOT = Path(__file__).parent.parent
FRONTEND = ROOT / "frontend"


class TestFrontendStructure:
    def test_frontend_directory_exists(self):
        assert FRONTEND.exists(), "frontend/ directory not found"

    def test_package_json_exists(self):
        pkg = FRONTEND / "package.json"
        assert pkg.exists(), "package.json not found"
        import json
        data = json.loads(pkg.read_text())
        assert data["name"] == "sprite-generator-frontend"
        assert "next" in data["dependencies"]
        assert "react" in data["dependencies"]

    def test_tsconfig_exists(self):
        assert (FRONTEND / "tsconfig.json").exists()

    def test_next_config_exists(self):
        assert (FRONTEND / "next.config.js").exists()

    def test_pages_exist(self):
        pages = FRONTEND / "pages"
        assert pages.exists()
        expected = ["_app.tsx", "index.tsx", "history.tsx", "downloads.tsx", "settings.tsx"]
        for p in expected:
            assert (pages / p).exists(), f"Missing page: {p}"

    def test_components_exist(self):
        comp = FRONTEND / "components"
        expected = ["Layout.tsx", "Navbar.tsx", "GenerateForm.tsx", "HistoryList.tsx", "DownloadList.tsx"]
        for c in expected:
            assert (comp / c).exists(), f"Missing component: {c}"

    def test_lib_api_exists(self):
        assert (FRONTEND / "lib" / "api.ts").exists()

    def test_styles_exist(self):
        assert (FRONTEND / "styles" / "globals.css").exists()


class TestFrontendPages:
    def test_index_exports_default(self):
        content = (FRONTEND / "pages" / "index.tsx").read_text()
        assert "GenerateForm" in content
        assert "export default" in content

    def test_history_exports_default(self):
        content = (FRONTEND / "pages" / "history.tsx").read_text()
        assert "HistoryList" in content
        assert "export default" in content

    def test_downloads_exports_default(self):
        content = (FRONTEND / "pages" / "downloads.tsx").read_text()
        assert "DownloadList" in content
        assert "export default" in content

    def test_settings_exports_default(self):
        content = (FRONTEND / "pages" / "settings.tsx").read_text()
        assert "SettingsPage" in content or "export default" in content
        assert "checkHealth" in content

    def test_app_uses_layout(self):
        content = (FRONTEND / "pages" / "_app.tsx").read_text()
        assert "Layout" in content
        assert "styles/globals.css" in content


class TestFrontendComponents:
    def test_navbar_has_links(self):
        content = (FRONTEND / "components" / "Navbar.tsx").read_text()
        for page in ["/history", "/downloads", "/settings"]:
            assert page in content, f"Navbar missing link: {page}"

    def test_generate_form_has_fields(self):
        content = (FRONTEND / "components" / "GenerateForm.tsx").read_text()
        for field in ["asset_type", "view", "animation", "palette", "sprite_size", "theme"]:
            assert field in content, f"GenerateForm missing field: {field}"

    def test_history_list_uses_api(self):
        content = (FRONTEND / "components" / "HistoryList.tsx").read_text()
        assert "getHistory" in content

    def test_download_list_filters_by_zip(self):
        content = (FRONTEND / "components" / "DownloadList.tsx").read_text()
        assert "zip_path" in content


class TestFrontendApi:
    def test_api_has_all_endpoints(self):
        content = (FRONTEND / "lib" / "api.ts").read_text()
        for endpoint in ["checkHealth", "generateAsset", "getHistory", "getDownloadUrl"]:
            assert endpoint in content, f"API missing function: {endpoint}"

    def test_api_has_interfaces(self):
        content = (FRONTEND / "lib" / "api.ts").read_text()
        for iface in ["GenerateRequest", "GenerateResponse", "HealthResponse", "HistoryEntry"]:
            assert iface in content, f"API missing interface: {iface}"

    def test_api_base_url_configurable(self):
        content = (FRONTEND / "lib" / "api.ts").read_text()
        assert "NEXT_PUBLIC_API_URL" in content
