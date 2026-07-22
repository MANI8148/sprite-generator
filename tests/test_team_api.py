import pytest
import tempfile
import os
from fastapi.testclient import TestClient
from backend.main import app
from backend.modules.teams import TeamManager, set_team_manager, get_team_manager
from backend.modules.storage.asset_library import AssetLibrary, AssetRecord
from backend.modules.auth import AuthHandler, set_auth_handler
from backend.modules.rate_limiter import RateLimiter, set_rate_limiter


@pytest.fixture
def client():
    tmpdir = tempfile.mkdtemp()
    auth = AuthHandler(users_path=os.path.join(tmpdir, "auth", "users.json"))
    set_auth_handler(auth)
    lib = AssetLibrary(base_dir=os.path.join(tmpdir, "library"))
    mgr = TeamManager(data_dir=os.path.join(tmpdir, "teams"), library=lib)
    set_team_manager(mgr)
    set_rate_limiter(RateLimiter(max_requests=10000, window_seconds=3600))
    return TestClient(app)


@pytest.fixture
def auth_headers(client):
    client.post("/auth/register", json={"username": "testuser", "password": "testpass"})
    resp = client.post("/auth/login", json={"username": "testuser", "password": "testpass"})
    data = resp.json()
    return {
        "Authorization": f"Bearer {data['access_token']}",
        "user_id": data["user_id"],
    }


class TestCreateTeamAPI:
    def test_create_team(self, client, auth_headers):
        resp = client.post("/teams", json={"name": "My Team", "description": "desc"}, headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "My Team"
        assert data["description"] == "desc"
        assert "team_id" in data
        assert data["owner_id"] is not None

    def test_create_team_empty_name_returns_422(self, client, auth_headers):
        resp = client.post("/teams", json={"name": "  "}, headers=auth_headers)
        assert resp.status_code == 422

    def test_create_team_requires_auth(self, client):
        resp = client.post("/teams", json={"name": "No Auth"})
        assert resp.status_code == 401


class TestListTeamsAPI:
    def test_list_teams_empty(self, client, auth_headers):
        resp = client.get("/teams", headers=auth_headers)
        assert resp.status_code == 200
        assert resp.json() == []

    def test_list_teams_returns_user_teams(self, client, auth_headers):
        client.post("/teams", json={"name": "Team A"}, headers=auth_headers)
        client.post("/teams", json={"name": "Team B"}, headers=auth_headers)
        resp = client.get("/teams", headers=auth_headers)
        data = resp.json()
        assert len(data) == 2
        names = [t["name"] for t in data]
        assert "Team A" in names
        assert "Team B" in names

    def test_list_teams_does_not_show_other_user_teams(self, client, auth_headers):
        client.post("/teams", json={"name": "Mine"}, headers=auth_headers)
        client.post("/auth/register", json={"username": "other", "password": "otherpass"})
        resp = client.post("/auth/login", json={"username": "other", "password": "otherpass"})
        other_token = resp.json()["access_token"]
        other_headers = {"Authorization": f"Bearer {other_token}"}
        resp = client.get("/teams", headers=other_headers)
        assert len(resp.json()) == 0


class TestGetTeamAPI:
    def test_get_team(self, client, auth_headers):
        create_resp = client.post("/teams", json={"name": "My Team"}, headers=auth_headers)
        team_id = create_resp.json()["team_id"]
        resp = client.get(f"/teams/{team_id}", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "My Team"
        assert data["role"] == "owner"

    def test_get_nonexistent_team(self, client, auth_headers):
        resp = client.get("/teams/nonexistent", headers=auth_headers)
        assert resp.status_code == 404

    def test_get_team_not_member(self, client, auth_headers):
        create_resp = client.post("/teams", json={"name": "My Team"}, headers=auth_headers)
        team_id = create_resp.json()["team_id"]
        client.post("/auth/register", json={"username": "other", "password": "otherpass"})
        resp = client.post("/auth/login", json={"username": "other", "password": "otherpass"})
        other_headers = {"Authorization": f"Bearer {resp.json()['access_token']}"}
        resp = client.get(f"/teams/{team_id}", headers=other_headers)
        assert resp.status_code == 403


class TestUpdateTeamAPI:
    def test_update_team_name(self, client, auth_headers):
        create_resp = client.post("/teams", json={"name": "Original"}, headers=auth_headers)
        team_id = create_resp.json()["team_id"]
        resp = client.patch(f"/teams/{team_id}", json={"name": "Renamed"}, headers=auth_headers)
        assert resp.status_code == 200
        assert resp.json()["name"] == "Renamed"

    def test_update_team_non_owner(self, client, auth_headers):
        create_resp = client.post("/teams", json={"name": "Original"}, headers=auth_headers)
        team_id = create_resp.json()["team_id"]
        client.post("/auth/register", json={"username": "other", "password": "otherpass"})
        resp = client.post("/auth/login", json={"username": "other", "password": "otherpass"})
        other_headers = {"Authorization": f"Bearer {resp.json()['access_token']}"}
        resp = client.patch(f"/teams/{team_id}", json={"name": "Hacked"}, headers=other_headers)
        assert resp.status_code == 403


class TestDeleteTeamAPI:
    def test_delete_team(self, client, auth_headers):
        create_resp = client.post("/teams", json={"name": "DeleteMe"}, headers=auth_headers)
        team_id = create_resp.json()["team_id"]
        resp = client.delete(f"/teams/{team_id}", headers=auth_headers)
        assert resp.status_code == 200
        resp = client.get(f"/teams/{team_id}", headers=auth_headers)
        assert resp.status_code == 404

    def test_delete_team_not_owner(self, client, auth_headers):
        create_resp = client.post("/teams", json={"name": "Mine"}, headers=auth_headers)
        team_id = create_resp.json()["team_id"]
        client.post("/auth/register", json={"username": "other", "password": "otherpass"})
        resp = client.post("/auth/login", json={"username": "other", "password": "otherpass"})
        other_headers = {"Authorization": f"Bearer {resp.json()['access_token']}"}
        resp = client.delete(f"/teams/{team_id}", headers=other_headers)
        assert resp.status_code == 403


class TestMembersAPI:
    def test_list_members(self, client, auth_headers):
        create_resp = client.post("/teams", json={"name": "Test"}, headers=auth_headers)
        team_id = create_resp.json()["team_id"]
        resp = client.get(f"/teams/{team_id}/members", headers=auth_headers)
        assert resp.status_code == 200
        members = resp.json()
        assert len(members) == 1
        assert members[0]["role"] == "owner"

    def test_add_member(self, client, auth_headers):
        reg = client.post("/auth/register", json={"username": "member1", "password": "memberpass"})
        member_user_id = reg.json()["user_id"]
        create_resp = client.post("/teams", json={"name": "Test"}, headers=auth_headers)
        team_id = create_resp.json()["team_id"]
        resp = client.post(
            f"/teams/{team_id}/members",
            json={"user_id": member_user_id, "role": "member"},
            headers=auth_headers,
        )
        assert resp.status_code == 200
        assert resp.json()["user_id"] == member_user_id

    def test_add_duplicate_member(self, client, auth_headers):
        reg = client.post("/auth/register", json={"username": "member1", "password": "memberpass"})
        member_user_id = reg.json()["user_id"]
        create_resp = client.post("/teams", json={"name": "Test"}, headers=auth_headers)
        team_id = create_resp.json()["team_id"]
        client.post(f"/teams/{team_id}/members", json={"user_id": member_user_id}, headers=auth_headers)
        resp = client.post(f"/teams/{team_id}/members", json={"user_id": member_user_id}, headers=auth_headers)
        assert resp.status_code == 409

    def test_add_member_not_owner(self, client, auth_headers):
        client.post("/auth/register", json={"username": "other", "password": "otherpass"})
        create_resp = client.post("/teams", json={"name": "Test"}, headers=auth_headers)
        team_id = create_resp.json()["team_id"]
        resp = client.post("/auth/login", json={"username": "other", "password": "otherpass"})
        other_headers = {"Authorization": f"Bearer {resp.json()['access_token']}"}
        resp = client.post(f"/teams/{team_id}/members", json={"user_id": "someone"}, headers=other_headers)
        assert resp.status_code == 403

    def test_add_member_invalid_role(self, client, auth_headers):
        create_resp = client.post("/teams", json={"name": "Test"}, headers=auth_headers)
        team_id = create_resp.json()["team_id"]
        resp = client.post(
            f"/teams/{team_id}/members",
            json={"user_id": "someone", "role": "superadmin"},
            headers=auth_headers,
        )
        assert resp.status_code == 422

    def test_remove_member(self, client, auth_headers):
        reg = client.post("/auth/register", json={"username": "member1", "password": "memberpass"})
        member_user_id = reg.json()["user_id"]
        create_resp = client.post("/teams", json={"name": "Test"}, headers=auth_headers)
        team_id = create_resp.json()["team_id"]
        client.post(f"/teams/{team_id}/members", json={"user_id": member_user_id}, headers=auth_headers)
        resp = client.delete(f"/teams/{team_id}/members/{member_user_id}", headers=auth_headers)
        assert resp.status_code == 200
        members_resp = client.get(f"/teams/{team_id}/members", headers=auth_headers)
        assert len(members_resp.json()) == 1

    def test_remove_owner_returns_422(self, client, auth_headers):
        create_resp = client.post("/teams", json={"name": "Test"}, headers=auth_headers)
        team_id = create_resp.json()["team_id"]
        owner_id = auth_headers["user_id"]
        resp = client.delete(f"/teams/{team_id}/members/{owner_id}", headers=auth_headers)
        assert resp.status_code == 422

    def test_remove_member_not_owner(self, client, auth_headers):
        client.post("/auth/register", json={"username": "other", "password": "otherpass"})
        create_resp = client.post("/teams", json={"name": "Test"}, headers=auth_headers)
        team_id = create_resp.json()["team_id"]
        owner_id = auth_headers["user_id"]
        other_login = client.post("/auth/login", json={"username": "other", "password": "otherpass"})
        other_user_id = other_login.json()["user_id"]
        client.post(f"/teams/{team_id}/members", json={"user_id": other_user_id}, headers=auth_headers)
        other_headers = {"Authorization": f"Bearer {other_login.json()['access_token']}"}
        resp = client.delete(f"/teams/{team_id}/members/{owner_id}", headers=other_headers)
        assert resp.status_code == 403


class TestTeamAssetsAPI:
    def test_share_asset_with_team(self, client, auth_headers):
        create_resp = client.post("/teams", json={"name": "Test"}, headers=auth_headers)
        team_id = create_resp.json()["team_id"]
        mgr = get_team_manager()
        asset = AssetRecord(
            asset_id="test_asset",
            job_id="job1",
            asset_type="character",
            prompt="hero",
            quality_tier="clean",
        )
        mgr._library.add_asset(asset)
        resp = client.post(f"/teams/{team_id}/assets/test_asset", headers=auth_headers)
        assert resp.status_code == 200

    def test_list_team_assets(self, client, auth_headers):
        create_resp = client.post("/teams", json={"name": "Test"}, headers=auth_headers)
        team_id = create_resp.json()["team_id"]
        mgr = get_team_manager()
        asset = AssetRecord(
            asset_id="test_asset",
            job_id="job1",
            asset_type="character",
            prompt="hero",
            quality_tier="clean",
        )
        mgr._library.add_asset(asset)
        client.post(f"/teams/{team_id}/assets/test_asset", headers=auth_headers)
        resp = client.get(f"/teams/{team_id}/assets", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["asset_id"] == "test_asset"

    def test_share_asset_requires_membership(self, client, auth_headers):
        create_resp = client.post("/teams", json={"name": "Test"}, headers=auth_headers)
        team_id = create_resp.json()["team_id"]
        client.post("/auth/register", json={"username": "other", "password": "otherpass"})
        resp = client.post("/auth/login", json={"username": "other", "password": "otherpass"})
        other_headers = {"Authorization": f"Bearer {resp.json()['access_token']}"}
        resp = client.post(f"/teams/{team_id}/assets/test_asset", headers=other_headers)
        assert resp.status_code == 403
