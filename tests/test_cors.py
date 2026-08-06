"""Tests for the FastAPI CORS configuration (roadmap: Phase 1 Item 1).

The Next.js frontend runs on a different origin than the FastAPI backend, so
browser calls to ``/generate``, ``/history``, etc. are cross-origin requests.
Without CORS middleware these calls are blocked by the browser and the Phase 1
"working website" cannot talk to the API. This covers configuration, env
override, preflight handling, and scoping of allowed origins.
"""

import os

import pytest
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient

from backend.main import app, resolve_allowed_origins


ALLOWED_ORIGIN = "http://localhost:3000"
DISALLOWED_ORIGIN = "https://evil.example.com"


class TestAllowedOriginsConfig:
    def test_default_origins_include_development_origins(self):
        origins = resolve_allowed_origins()
        assert "http://localhost:3000" in origins
        assert "http://127.0.0.1:3000" in origins
        assert "http://localhost:8000" in origins

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("ALLOWED_ORIGINS", "https://app.example.com, https://cdn.example.com")
        assert resolve_allowed_origins() == [
            "https://app.example.com",
            "https://cdn.example.com",
        ]

    def test_env_fallback_to_wildcard_when_empty(self, monkeypatch):
        monkeypatch.setenv("ALLOWED_ORIGINS", "   , ,")
        assert resolve_allowed_origins() == ["*"]

    def test_middleware_is_registered(self):
        classes = [m.cls for m in app.user_middleware]
        assert CORSMiddleware in classes

    def test_cors_is_outermost_middleware(self):
        # Starlette applies the last-added middleware outermost; CORS must be
        # there so preflight requests are handled before auth/rate-limit.
        registered = list(app.user_middleware)
        assert registered, "no middleware registered"
        assert registered[0].cls is CORSMiddleware


class TestCrossOriginRequests:
    def test_allowed_origin_gets_allow_origin_header(self):
        client = TestClient(app)
        resp = client.get("/health", headers={"Origin": ALLOWED_ORIGIN})
        assert resp.status_code == 200
        assert resp.headers.get("access-control-allow-origin") == ALLOWED_ORIGIN

    def test_json_post_allowed_origin(self):
        client = TestClient(app)
        resp = client.get("/palettes", headers={"Origin": ALLOWED_ORIGIN})
        assert resp.status_code == 200
        assert resp.headers.get("access-control-allow-origin") == ALLOWED_ORIGIN

    def test_disallowed_origin_gets_no_allow_origin_header(self):
        client = TestClient(app)
        resp = client.get("/palettes", headers={"Origin": DISALLOWED_ORIGIN})
        assert resp.headers.get("access-control-allow-origin") is None

    def test_backend_origin_is_allowed_for_self_calls(self):
        client = TestClient(app)
        resp = client.get("/health", headers={"Origin": "http://127.0.0.1:8000"})
        assert resp.headers.get("access-control-allow-origin") == "http://127.0.0.1:8000"


class TestPreflight:
    def test_generate_preflight_returns_200_with_cors_headers(self):
        client = TestClient(app)
        resp = client.options(
            "/generate",
            headers={
                "Origin": ALLOWED_ORIGIN,
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type,authorization",
            },
        )
        assert resp.status_code == 200
        assert resp.headers.get("access-control-allow-origin") == ALLOWED_ORIGIN
        methods = resp.headers.get("access-control-allow-methods", "")
        assert "POST" in methods
        headers = resp.headers.get("access-control-allow-headers", "")
        assert "content-type" in headers
        assert "authorization" in headers
        assert resp.headers.get("access-control-allow-credentials") == "true"

    def test_preflight_does_not_require_loaded_generator(self):
        # Preflight must be answered by CORS before it reaches the route,
        # otherwise it would 503/405 instead of returning the CORS envelope.
        client = TestClient(app)
        resp = client.options(
            "/generate",
            headers={
                "Origin": ALLOWED_ORIGIN,
                "Access-Control-Request-Method": "POST",
            },
        )
        assert resp.status_code == 200
        assert resp.headers.get("access-control-allow-origin") == ALLOWED_ORIGIN

    def test_preflight_disallowed_origin_returns_no_allow_origin(self):
        client = TestClient(app)
        resp = client.options(
            "/generate",
            headers={
                "Origin": DISALLOWED_ORIGIN,
                "Access-Control-Request-Method": "POST",
            },
        )
        assert resp.headers.get("access-control-allow-origin") is None