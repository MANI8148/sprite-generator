import json
import os
import threading
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional

from backend.modules.storage.asset_library import AssetLibrary, AssetRecord


@dataclass
class Team:
    team_id: str
    name: str
    owner_id: str
    description: str = ""
    created_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()


@dataclass
class TeamMember:
    team_id: str
    user_id: str
    role: str = "member"
    joined_at: str = ""

    def __post_init__(self):
        if not self.joined_at:
            self.joined_at = datetime.now(timezone.utc).isoformat()


class TeamManager:
    def __init__(self, data_dir: str = "data/teams", library: Optional[AssetLibrary] = None):
        self.data_dir = data_dir
        self._teams_path = os.path.join(data_dir, "teams.json")
        self._members_path = os.path.join(data_dir, "members.json")
        self._lock = threading.Lock()
        self._library = library or AssetLibrary()
        os.makedirs(data_dir, exist_ok=True)

    def _load_json(self, path: str) -> dict:
        if not os.path.isfile(path):
            return {}
        with open(path, "r") as f:
            return json.load(f)

    def _save_json(self, path: str, data: dict):
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def create_team(self, name: str, owner_id: str, description: str = "") -> Team:
        team_id = str(uuid.uuid4())[:8]
        team = Team(
            team_id=team_id,
            name=name,
            owner_id=owner_id,
            description=description,
        )
        with self._lock:
            teams = self._load_json(self._teams_path)
            teams[team_id] = asdict(team)
            self._save_json(self._teams_path, teams)

            members = self._load_json(self._members_path)
            if team_id not in members:
                members[team_id] = []
            owner_member = TeamMember(team_id=team_id, user_id=owner_id, role="owner")
            members[team_id].append(asdict(owner_member))
            self._save_json(self._members_path, members)

        return team

    def get_team(self, team_id: str) -> Optional[Team]:
        with self._lock:
            teams = self._load_json(self._teams_path)
            data = teams.get(team_id)
        if data is None:
            return None
        return Team(**data)

    def update_team(self, team_id: str, **updates) -> Optional[Team]:
        with self._lock:
            teams = self._load_json(self._teams_path)
            if team_id not in teams:
                return None
            for k, v in updates.items():
                if k in ("team_id", "owner_id", "created_at"):
                    continue
                teams[team_id][k] = v
            self._save_json(self._teams_path, teams)
            return Team(**teams[team_id])

    def delete_team(self, team_id: str) -> bool:
        with self._lock:
            teams = self._load_json(self._teams_path)
            if team_id not in teams:
                return False
            del teams[team_id]
            self._save_json(self._teams_path, teams)

            members = self._load_json(self._members_path)
            members.pop(team_id, None)
            self._save_json(self._members_path, members)
        return True

    def list_user_teams(self, user_id: str) -> list:
        with self._lock:
            teams = self._load_json(self._teams_path)
            members = self._load_json(self._members_path)

        result = []
        for tid, team_data in teams.items():
            team_members = members.get(tid, [])
            if any(m["user_id"] == user_id for m in team_members):
                result.append(Team(**team_data))
        return result

    def add_member(self, team_id: str, user_id: str, role: str = "member") -> Optional[TeamMember]:
        with self._lock:
            teams = self._load_json(self._teams_path)
            if team_id not in teams:
                return None

            members = self._load_json(self._members_path)
            if team_id not in members:
                members[team_id] = []

            if any(m["user_id"] == user_id for m in members[team_id]):
                return None

            tm = TeamMember(team_id=team_id, user_id=user_id, role=role)
            members[team_id].append(asdict(tm))
            self._save_json(self._members_path, members)
            return tm

    def remove_member(self, team_id: str, user_id: str) -> bool:
        with self._lock:
            members = self._load_json(self._members_path)
            if team_id not in members:
                return False
            before = len(members[team_id])
            members[team_id] = [m for m in members[team_id] if m["user_id"] != user_id]
            if len(members[team_id]) == before:
                return False
            if not members[team_id]:
                del members[team_id]
            else:
                self._save_json(self._members_path, members)
            return True

    def get_members(self, team_id: str) -> list:
        with self._lock:
            members = self._load_json(self._members_path)
            raw = members.get(team_id, [])
        return [TeamMember(**m) for m in raw]

    def get_user_role(self, team_id: str, user_id: str) -> Optional[str]:
        with self._lock:
            members = self._load_json(self._members_path)
            team_members = members.get(team_id, [])
        for m in team_members:
            if m["user_id"] == user_id:
                return m["role"]
        return None

    def share_asset_with_team(self, team_id: str, asset_id: str) -> bool:
        team = self.get_team(team_id)
        if team is None:
            return False
        asset = self._library.get_asset(asset_id)
        if asset is None:
            return False
        shared_with = set(asset.metadata.get("shared_with_teams", []))
        if team_id in shared_with:
            return True
        shared_with.add(team_id)
        self._library.update_asset(asset_id, metadata={**asset.metadata, "shared_with_teams": list(shared_with)})
        return True

    def get_team_assets(self, team_id: str) -> list:
        team = self.get_team(team_id)
        if team is None:
            return []
        all_assets = self._library.list_assets(limit=10000)
        return [a for a in all_assets if team_id in a.metadata.get("shared_with_teams", [])]


_default_team_manager: Optional[TeamManager] = None


def get_team_manager() -> TeamManager:
    global _default_team_manager
    if _default_team_manager is None:
        _default_team_manager = TeamManager()
    return _default_team_manager


def set_team_manager(mgr: TeamManager):
    global _default_team_manager
    _default_team_manager = mgr
