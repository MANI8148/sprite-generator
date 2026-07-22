from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from typing import Optional

from backend.modules.auth import get_current_user, TokenData
from backend.modules.teams import TeamManager, get_team_manager

router = APIRouter(prefix="/teams", tags=["teams"])


class CreateTeamRequest(BaseModel):
    name: str
    description: str = ""


class UpdateTeamRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None


class AddMemberRequest(BaseModel):
    user_id: str
    role: str = "member"


@router.post("")
def create_team(
    req: CreateTeamRequest,
    user: TokenData = Depends(get_current_user),
    mgr: TeamManager = Depends(get_team_manager),
):
    if not req.name.strip():
        raise HTTPException(status_code=422, detail="Team name cannot be empty")
    team = mgr.create_team(name=req.name.strip(), owner_id=user.user_id, description=req.description)
    return {
        "team_id": team.team_id,
        "name": team.name,
        "description": team.description,
        "owner_id": team.owner_id,
        "created_at": team.created_at,
    }


@router.get("")
def list_teams(
    user: TokenData = Depends(get_current_user),
    mgr: TeamManager = Depends(get_team_manager),
):
    teams = mgr.list_user_teams(user.user_id)
    return [
        {
            "team_id": t.team_id,
            "name": t.name,
            "description": t.description,
            "owner_id": t.owner_id,
            "created_at": t.created_at,
        }
        for t in teams
    ]


@router.get("/{team_id}")
def get_team(
    team_id: str,
    user: TokenData = Depends(get_current_user),
    mgr: TeamManager = Depends(get_team_manager),
):
    team = mgr.get_team(team_id)
    if team is None:
        raise HTTPException(status_code=404, detail="Team not found")
    role = mgr.get_user_role(team_id, user.user_id)
    if role is None:
        raise HTTPException(status_code=403, detail="Not a member of this team")
    return {
        "team_id": team.team_id,
        "name": team.name,
        "description": team.description,
        "owner_id": team.owner_id,
        "created_at": team.created_at,
        "role": role,
    }


@router.patch("/{team_id}")
def update_team(
    team_id: str,
    req: UpdateTeamRequest,
    user: TokenData = Depends(get_current_user),
    mgr: TeamManager = Depends(get_team_manager),
):
    team = mgr.get_team(team_id)
    if team is None:
        raise HTTPException(status_code=404, detail="Team not found")
    role = mgr.get_user_role(team_id, user.user_id)
    if role not in ("owner", "admin"):
        raise HTTPException(status_code=403, detail="Only owner or admin can update team")

    updates = {}
    if req.name is not None:
        updates["name"] = req.name
    if req.description is not None:
        updates["description"] = req.description

    updated = mgr.update_team(team_id, **updates)
    return {
        "team_id": updated.team_id,
        "name": updated.name,
        "description": updated.description,
        "owner_id": updated.owner_id,
        "created_at": updated.created_at,
    }


@router.delete("/{team_id}")
def delete_team(
    team_id: str,
    user: TokenData = Depends(get_current_user),
    mgr: TeamManager = Depends(get_team_manager),
):
    team = mgr.get_team(team_id)
    if team is None:
        raise HTTPException(status_code=404, detail="Team not found")
    if team.owner_id != user.user_id:
        raise HTTPException(status_code=403, detail="Only the owner can delete the team")
    mgr.delete_team(team_id)
    return {"detail": "Team deleted"}


@router.get("/{team_id}/members")
def list_members(
    team_id: str,
    user: TokenData = Depends(get_current_user),
    mgr: TeamManager = Depends(get_team_manager),
):
    team = mgr.get_team(team_id)
    if team is None:
        raise HTTPException(status_code=404, detail="Team not found")
    role = mgr.get_user_role(team_id, user.user_id)
    if role is None:
        raise HTTPException(status_code=403, detail="Not a member of this team")
    members = mgr.get_members(team_id)
    return [
        {
            "user_id": m.user_id,
            "role": m.role,
            "joined_at": m.joined_at,
        }
        for m in members
    ]


@router.post("/{team_id}/members")
def add_member(
    team_id: str,
    req: AddMemberRequest,
    user: TokenData = Depends(get_current_user),
    mgr: TeamManager = Depends(get_team_manager),
):
    team = mgr.get_team(team_id)
    if team is None:
        raise HTTPException(status_code=404, detail="Team not found")
    role = mgr.get_user_role(team_id, user.user_id)
    if role not in ("owner", "admin"):
        raise HTTPException(status_code=403, detail="Only owner or admin can add members")
    if req.role not in ("admin", "member"):
        raise HTTPException(status_code=422, detail="Role must be 'admin' or 'member'")

    tm = mgr.add_member(team_id, req.user_id, req.role)
    if tm is None:
        raise HTTPException(status_code=409, detail="User is already a member")
    return {
        "user_id": tm.user_id,
        "role": tm.role,
        "joined_at": tm.joined_at,
    }


@router.delete("/{team_id}/members/{target_user_id}")
def remove_member(
    team_id: str,
    target_user_id: str,
    user: TokenData = Depends(get_current_user),
    mgr: TeamManager = Depends(get_team_manager),
):
    team = mgr.get_team(team_id)
    if team is None:
        raise HTTPException(status_code=404, detail="Team not found")
    caller_role = mgr.get_user_role(team_id, user.user_id)
    if caller_role not in ("owner", "admin"):
        raise HTTPException(status_code=403, detail="Only owner or admin can remove members")
    if user.user_id == target_user_id:
        raise HTTPException(status_code=422, detail="Cannot remove yourself; delete the team instead")
    target_role = mgr.get_user_role(team_id, target_user_id)
    if target_role == "owner":
        raise HTTPException(status_code=422, detail="Cannot remove the team owner")
    if not mgr.remove_member(team_id, target_user_id):
        raise HTTPException(status_code=404, detail="Member not found")
    return {"detail": "Member removed"}


@router.post("/{team_id}/assets/{asset_id}")
def share_asset(
    team_id: str,
    asset_id: str,
    user: TokenData = Depends(get_current_user),
    mgr: TeamManager = Depends(get_team_manager),
):
    team = mgr.get_team(team_id)
    if team is None:
        raise HTTPException(status_code=404, detail="Team not found")
    role = mgr.get_user_role(team_id, user.user_id)
    if role is None:
        raise HTTPException(status_code=403, detail="Not a member of this team")
    if not mgr.share_asset_with_team(team_id, asset_id):
        raise HTTPException(status_code=404, detail="Team or asset not found")
    return {"detail": f"Asset {asset_id} shared with team {team_id}"}


@router.get("/{team_id}/assets")
def list_team_assets(
    team_id: str,
    user: TokenData = Depends(get_current_user),
    mgr: TeamManager = Depends(get_team_manager),
):
    team = mgr.get_team(team_id)
    if team is None:
        raise HTTPException(status_code=404, detail="Team not found")
    role = mgr.get_user_role(team_id, user.user_id)
    if role is None:
        raise HTTPException(status_code=403, detail="Not a member of this team")
    assets = mgr.get_team_assets(team_id)
    return [
        {
            "asset_id": a.asset_id,
            "asset_type": a.asset_type,
            "prompt": a.prompt,
            "quality_tier": a.quality_tier,
            "created_at": a.created_at,
        }
        for a in assets
    ]
