import json
import os
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.modules.auth import (
    AuthHandler,
    get_auth_handler,
    set_auth_handler,
    UserRecord,
    TokenData,
)
from backend.modules.rate_limiter import RateLimiter, set_rate_limiter, get_rate_limiter


@pytest.fixture(autouse=True)
def test_setup():
    old_auth = get_auth_handler()
    old_limiter = get_rate_limiter()
    tmp = tempfile.mkdtemp()
    handler = AuthHandler(users_path=os.path.join(tmp, "users.json"))
    set_auth_handler(handler)
    limiter = RateLimiter(max_requests=1000, window_seconds=60)
    set_rate_limiter(limiter)
    yield
    set_rate_limiter(old_limiter)
    set_auth_handler(old_auth)


client = TestClient(app)


class TestAuthHandler:
    def test_hash_and_verify(self):
        handler = get_auth_handler()
        h = handler.hash_password("test123")
        assert h != "test123"
        assert handler.verify_password("test123", h)
        assert not handler.verify_password("wrong", h)

    def test_register_new_user(self):
        handler = get_auth_handler()
        record = handler.register("alice", "secret123")
        assert record.username == "alice"
        assert record.user_id
        assert record.hashed_password != "secret123"

    def test_register_duplicate_raises(self):
        handler = get_auth_handler()
        handler.register("alice", "secret123")
        with pytest.raises(Exception):
            handler.register("alice", "other456")

    def test_authenticate_valid(self):
        handler = get_auth_handler()
        handler.register("bob", "pass1234")
        td = handler.authenticate("bob", "pass1234")
        assert td is not None
        assert td.username == "bob"

    def test_authenticate_wrong_password(self):
        handler = get_auth_handler()
        handler.register("bob", "pass1234")
        td = handler.authenticate("bob", "wrongpass")
        assert td is None

    def test_authenticate_unknown_user(self):
        handler = get_auth_handler()
        td = handler.authenticate("nobody", "pass1234")
        assert td is None

    def test_create_and_decode_token(self):
        handler = get_auth_handler()
        td = TokenData(username="charlie", user_id="abc123")
        token = handler.create_access_token(td)
        decoded = handler.decode_token(token)
        assert decoded is not None
        assert decoded.username == "charlie"
        assert decoded.user_id == "abc123"

    def test_decode_invalid_token(self):
        handler = get_auth_handler()
        decoded = handler.decode_token("invalid.token.here")
        assert decoded is None

    def test_decode_expired_token(self):
        import jwt
        from datetime import datetime, timedelta, timezone
        handler = get_auth_handler()
        payload = {
            "sub": "test",
            "user_id": "test123",
            "exp": datetime.now(timezone.utc) - timedelta(hours=1),
            "iat": datetime.now(timezone.utc) - timedelta(hours=2),
        }
        token = jwt.encode(payload, "dev-secret-key-change-in-production", algorithm="HS256")
        decoded = handler.decode_token(token)
        assert decoded is None


class TestAuthAPI:
    def test_register_creates_user(self):
        resp = client.post("/auth/register", json={
            "username": "newuser",
            "password": "test123456",
        })
        assert resp.status_code == 201
        data = resp.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"
        assert data["username"] == "newuser"
        assert data["user_id"]

    def test_register_duplicate_returns_409(self):
        client.post("/auth/register", json={
            "username": "dupuser",
            "password": "test123456",
        })
        resp = client.post("/auth/register", json={
            "username": "dupuser",
            "password": "otherpass",
        })
        assert resp.status_code == 409

    def test_register_short_username_returns_422(self):
        resp = client.post("/auth/register", json={
            "username": "ab",
            "password": "test123456",
        })
        assert resp.status_code == 422

    def test_register_short_password_returns_422(self):
        resp = client.post("/auth/register", json={
            "username": "validuser",
            "password": "short",
        })
        assert resp.status_code == 422

    def test_login_valid_returns_token(self):
        client.post("/auth/register", json={
            "username": "loginuser",
            "password": "mypassword1",
        })
        resp = client.post("/auth/login", json={
            "username": "loginuser",
            "password": "mypassword1",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "access_token" in data
        assert data["username"] == "loginuser"

    def test_login_wrong_password_returns_401(self):
        client.post("/auth/register", json={
            "username": "loginuser2",
            "password": "mypassword1",
        })
        resp = client.post("/auth/login", json={
            "username": "loginuser2",
            "password": "wrongpassword",
        })
        assert resp.status_code == 401

    def test_login_unknown_user_returns_401(self):
        resp = client.post("/auth/login", json={
            "username": "unknown",
            "password": "somepass",
        })
        assert resp.status_code == 401

    def test_me_with_valid_token(self):
        resp = client.post("/auth/register", json={
            "username": "metest",
            "password": "testpass123",
        })
        assert resp.status_code == 201
        token = resp.json()["access_token"]
        resp = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["username"] == "metest"

    def test_me_without_token_returns_401(self):
        resp = client.get("/auth/me")
        assert resp.status_code == 401

    def test_me_with_invalid_token_returns_401(self):
        resp = client.get("/auth/me", headers={"Authorization": "Bearer invalidtoken"})
        assert resp.status_code == 401

    def test_register_strips_whitespace(self):
        resp = client.post("/auth/register", json={
            "username": "  spaceduser  ",
            "password": "test123456",
        })
        assert resp.status_code == 201
        assert resp.json()["username"] == "spaceduser"

    def test_register_then_login_with_same_creds(self):
        reg = client.post("/auth/register", json={
            "username": "persist_user",
            "password": "persist_pass1",
        })
        assert reg.status_code == 201
        login = client.post("/auth/login", json={
            "username": "persist_user",
            "password": "persist_pass1",
        })
        assert login.status_code == 200
        assert "access_token" in login.json()
