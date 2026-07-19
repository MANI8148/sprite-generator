from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from typing import Optional

from backend.modules.auth import (
    AuthHandler,
    get_auth_handler,
    get_current_user,
    TokenData,
)

router = APIRouter(prefix="/auth", tags=["auth"])


class RegisterRequest(BaseModel):
    username: str
    password: str


class LoginRequest(BaseModel):
    username: str
    password: str


class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    username: str
    user_id: str


class UserInfoResponse(BaseModel):
    username: str
    user_id: str


@router.post("/register", response_model=AuthResponse, status_code=201)
def register(
    req: RegisterRequest,
    auth: AuthHandler = Depends(get_auth_handler),
):
    if len(req.username.strip()) < 3:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Username must be at least 3 characters",
        )
    if len(req.password) < 6:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Password must be at least 6 characters",
        )
    record = auth.register(req.username.strip(), req.password)
    token_data = TokenData(username=record.username, user_id=record.user_id)
    token = auth.create_access_token(token_data)
    return AuthResponse(
        access_token=token,
        username=record.username,
        user_id=record.user_id,
    )


@router.post("/login", response_model=AuthResponse)
def login(
    req: LoginRequest,
    auth: AuthHandler = Depends(get_auth_handler),
):
    token_data = auth.authenticate(req.username, req.password)
    if token_data is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
        )
    token = auth.create_access_token(token_data)
    return AuthResponse(
        access_token=token,
        username=token_data.username,
        user_id=token_data.user_id,
    )


@router.get("/me", response_model=UserInfoResponse)
def get_me(
    current_user: TokenData = Depends(get_current_user),
):
    return UserInfoResponse(
        username=current_user.username,
        user_id=current_user.user_id,
    )
