import pytest
import tempfile
import os

from backend.modules.teams import TeamManager, Team, TeamMember
from backend.modules.storage.asset_library import AssetLibrary, AssetRecord


@pytest.fixture
def team_mgr():
    tmpdir = tempfile.mkdtemp()
    lib = AssetLibrary(base_dir=os.path.join(tmpdir, "library"))
    mgr = TeamManager(data_dir=os.path.join(tmpdir, "teams"), library=lib)
    yield mgr


class TestCreateTeam:
    def test_create_team_returns_team_with_owner(self, team_mgr):
        team = team_mgr.create_team(name="My Team", owner_id="user1", description="test")
        assert team.team_id is not None
        assert team.name == "My Team"
        assert team.owner_id == "user1"
        assert team.description == "test"
        assert team.created_at != ""

    def test_create_team_stores_persistently(self, team_mgr):
        team = team_mgr.create_team(name="Persistent", owner_id="user1")
        loaded = team_mgr.get_team(team.team_id)
        assert loaded is not None
        assert loaded.name == "Persistent"
        assert loaded.owner_id == "user1"

    def test_create_team_adds_owner_as_member(self, team_mgr):
        team = team_mgr.create_team(name="Test", owner_id="user1")
        members = team_mgr.get_members(team.team_id)
        assert len(members) == 1
        assert members[0].user_id == "user1"
        assert members[0].role == "owner"


class TestGetTeam:
    def test_get_nonexistent_team_returns_none(self, team_mgr):
        assert team_mgr.get_team("nonexistent") is None

    def test_get_existing_team_returns_team(self, team_mgr):
        t = team_mgr.create_team(name="Exists", owner_id="user1")
        assert team_mgr.get_team(t.team_id) is not None


class TestUpdateTeam:
    def test_update_name(self, team_mgr):
        t = team_mgr.create_team(name="Original", owner_id="user1")
        updated = team_mgr.update_team(t.team_id, name="Renamed")
        assert updated.name == "Renamed"
        fetched = team_mgr.get_team(t.team_id)
        assert fetched.name == "Renamed"

    def test_update_description(self, team_mgr):
        t = team_mgr.create_team(name="Test", owner_id="user1")
        updated = team_mgr.update_team(t.team_id, description="new desc")
        assert updated.description == "new desc"

    def test_update_nonexistent_returns_none(self, team_mgr):
        assert team_mgr.update_team("nonexistent", name="x") is None

    def test_cannot_update_owner_id(self, team_mgr):
        t = team_mgr.create_team(name="Test", owner_id="user1")
        updated = team_mgr.update_team(t.team_id, owner_id="user2")
        assert updated.owner_id == "user1"


class TestDeleteTeam:
    def test_delete_existing_returns_true(self, team_mgr):
        t = team_mgr.create_team(name="DeleteMe", owner_id="user1")
        assert team_mgr.delete_team(t.team_id) is True
        assert team_mgr.get_team(t.team_id) is None

    def test_delete_nonexistent_returns_false(self, team_mgr):
        assert team_mgr.delete_team("nonexistent") is False

    def test_delete_removes_members(self, team_mgr):
        t = team_mgr.create_team(name="Test", owner_id="user1")
        team_mgr.add_member(t.team_id, "user2")
        team_mgr.delete_team(t.team_id)
        assert team_mgr.get_members(t.team_id) == []


class TestListUserTeams:
    def test_user_with_no_teams(self, team_mgr):
        assert team_mgr.list_user_teams("nobody") == []

    def test_user_sees_only_own_teams(self, team_mgr):
        t1 = team_mgr.create_team(name="Team A", owner_id="user1")
        team_mgr.create_team(name="Team B", owner_id="user2")

        user1_teams = team_mgr.list_user_teams("user1")
        assert len(user1_teams) == 1
        assert user1_teams[0].team_id == t1.team_id

    def test_member_sees_team(self, team_mgr):
        t = team_mgr.create_team(name="Shared", owner_id="user1")
        team_mgr.add_member(t.team_id, "user2")
        teams = team_mgr.list_user_teams("user2")
        assert len(teams) == 1
        assert teams[0].team_id == t.team_id


class TestAddMember:
    def test_add_member_returns_member(self, team_mgr):
        t = team_mgr.create_team(name="Test", owner_id="user1")
        tm = team_mgr.add_member(t.team_id, "user2", role="admin")
        assert tm is not None
        assert tm.user_id == "user2"
        assert tm.role == "admin"

    def test_add_member_to_nonexistent_team_returns_none(self, team_mgr):
        assert team_mgr.add_member("nonexistent", "user2") is None

    def test_add_duplicate_member_returns_none(self, team_mgr):
        t = team_mgr.create_team(name="Test", owner_id="user1")
        assert team_mgr.add_member(t.team_id, "user2") is not None
        assert team_mgr.add_member(t.team_id, "user2") is None

    def test_default_role_is_member(self, team_mgr):
        t = team_mgr.create_team(name="Test", owner_id="user1")
        tm = team_mgr.add_member(t.team_id, "user2")
        assert tm.role == "member"


class TestRemoveMember:
    def test_remove_existing_member(self, team_mgr):
        t = team_mgr.create_team(name="Test", owner_id="user1")
        team_mgr.add_member(t.team_id, "user2")
        assert team_mgr.remove_member(t.team_id, "user2") is True
        members = team_mgr.get_members(t.team_id)
        assert all(m.user_id != "user2" for m in members)

    def test_remove_nonexistent_member(self, team_mgr):
        t = team_mgr.create_team(name="Test", owner_id="user1")
        assert team_mgr.remove_member(t.team_id, "nobody") is False

    def test_remove_from_nonexistent_team(self, team_mgr):
        assert team_mgr.remove_member("nonexistent", "user1") is False


class TestGetMembers:
    def test_get_members_returns_all(self, team_mgr):
        t = team_mgr.create_team(name="Test", owner_id="user1")
        team_mgr.add_member(t.team_id, "user2")
        team_mgr.add_member(t.team_id, "user3")
        members = team_mgr.get_members(t.team_id)
        assert len(members) == 3
        user_ids = {m.user_id for m in members}
        assert user_ids == {"user1", "user2", "user3"}

    def test_get_members_returns_role(self, team_mgr):
        t = team_mgr.create_team(name="Test", owner_id="user1")
        assert team_mgr.get_members(t.team_id)[0].role == "owner"


class TestGetUserRole:
    def test_owner_role(self, team_mgr):
        t = team_mgr.create_team(name="Test", owner_id="user1")
        assert team_mgr.get_user_role(t.team_id, "user1") == "owner"

    def test_admin_role(self, team_mgr):
        t = team_mgr.create_team(name="Test", owner_id="user1")
        team_mgr.add_member(t.team_id, "user2", role="admin")
        assert team_mgr.get_user_role(t.team_id, "user2") == "admin"

    def test_non_member_returns_none(self, team_mgr):
        t = team_mgr.create_team(name="Test", owner_id="user1")
        assert team_mgr.get_user_role(t.team_id, "nobody") is None


class TestShareAsset:
    def test_share_asset_with_team(self, team_mgr):
        t = team_mgr.create_team(name="Test", owner_id="user1")
        asset = AssetRecord(
            asset_id="asset1",
            job_id="job1",
            asset_type="character",
            prompt="hero",
            quality_tier="clean",
        )
        team_mgr._library.add_asset(asset)
        assert team_mgr.share_asset_with_team(t.team_id, "asset1") is True
        shared = team_mgr._library.get_asset("asset1")
        assert t.team_id in shared.metadata.get("shared_with_teams", [])

    def test_share_asset_with_nonexistent_team(self, team_mgr):
        assert team_mgr.share_asset_with_team("nonexistent", "asset1") is False

    def test_share_nonexistent_asset(self, team_mgr):
        t = team_mgr.create_team(name="Test", owner_id="user1")
        assert team_mgr.share_asset_with_team(t.team_id, "nobody") is False

    def test_double_share_is_idempotent(self, team_mgr):
        t = team_mgr.create_team(name="Test", owner_id="user1")
        asset = AssetRecord(
            asset_id="asset1",
            job_id="job1",
            asset_type="character",
            prompt="hero",
            quality_tier="clean",
        )
        team_mgr._library.add_asset(asset)
        assert team_mgr.share_asset_with_team(t.team_id, "asset1") is True
        assert team_mgr.share_asset_with_team(t.team_id, "asset1") is True


class TestGetTeamAssets:
    def test_get_team_assets_returns_shared(self, team_mgr):
        t = team_mgr.create_team(name="Test", owner_id="user1")
        asset1 = AssetRecord(
            asset_id="asset1", job_id="j1", asset_type="character",
            prompt="hero", quality_tier="clean",
        )
        asset2 = AssetRecord(
            asset_id="asset2", job_id="j2", asset_type="enemy",
            prompt="goblin", quality_tier="clean",
        )
        team_mgr._library.add_asset(asset1)
        team_mgr._library.add_asset(asset2)
        team_mgr.share_asset_with_team(t.team_id, "asset1")
        team_assets = team_mgr.get_team_assets(t.team_id)
        assert len(team_assets) == 1
        assert team_assets[0].asset_id == "asset1"

    def test_get_team_assets_empty_for_new_team(self, team_mgr):
        t = team_mgr.create_team(name="Test", owner_id="user1")
        assert team_mgr.get_team_assets(t.team_id) == []

    def test_get_team_assets_nonexistent_team(self, team_mgr):
        assert team_mgr.get_team_assets("nonexistent") == []


class TestGetSetTeamManager:
    def test_get_team_manager_returns_default(self):
        from backend.modules.teams import get_team_manager, set_team_manager
        mgr = get_team_manager()
        assert mgr is not None
        assert isinstance(mgr, TeamManager)

        custom = TeamManager()
        set_team_manager(custom)
        assert get_team_manager() is custom

        set_team_manager(mgr)
