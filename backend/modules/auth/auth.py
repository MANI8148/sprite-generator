import hashlib
import json
import os
import secrets
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

SECRET_KEY = os.environ.get("AUTH_SECRET_KEY", "dev-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.environ.get("AUTH_TOKEN_EXPIRE_MINUTES", "60"))

security = HTTPBearer(auto_error=False)

PBKDF2_ITERATIONS = 600000


def _hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), PBKDF2_ITERATIONS)
    return f"{PBKDF2_ITERATIONS}${salt}${dk.hex()}"


def _verify_password(password: str, hashed: str) -> bool:
    parts = hashed.split("$")
    if len(parts) != 3:
        return False
    iterations_str, salt, stored_hash = parts
    try:
        iterations = int(iterations_str)
    except ValueError:
        return False
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), iterations)
    return secrets.compare_digest(dk.hex(), stored_hash)


@dataclass
class UserRecord:
    user_id: str
    username: str
    hashed_password: str
    created_at: str = ""
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()


@dataclass
class TokenData:
    username: str
    user_id: str


class AuthHandler:
    def __init__(self, users_path: str = "data/auth/users.json"):
        self.users_path = users_path
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(self.users_path) or ".", exist_ok=True)

    def _load_users(self) -> dict:
        if not os.path.isfile(self.users_path):
            return {}
        with open(self.users_path, "r") as f:
            return json.load(f)

    def _save_users(self, users: dict):
        with open(self.users_path, "w") as f:
            json.dump(users, f, indent=2)

    def hash_password(self, password: str) -> str:
        return _hash_password(password)

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        return _verify_password(plain_password, hashed_password)

    def create_access_token(self, data: TokenData) -> str:
        expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        payload = {
            "sub": data.username,
            "user_id": data.user_id,
            "exp": expire,
            "iat": datetime.now(timezone.utc),
        }
        return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

    def decode_token(self, token: str) -> Optional[TokenData]:
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            return TokenData(
                username=payload.get("sub", ""),
                user_id=payload.get("user_id", ""),
            )
        except jwt.PyJWTError:
            return None

    def register(self, username: str, password: str) -> UserRecord:
        with self._lock:
            users = self._load_users()
            if username in users:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="Username already exists",
                )
            user_id = str(uuid.uuid4())[:8]
            record = UserRecord(
                user_id=user_id,
                username=username,
                hashed_password=self.hash_password(password),
            )
            users[username] = {
                "user_id": user_id,
                "username": username,
                "hashed_password": record.hashed_password,
                "created_at": record.created_at,
                "metadata": {},
            }
            self._save_users(users)
            return record

    def authenticate(self, username: str, password: str) -> Optional[TokenData]:
        with self._lock:
            users = self._load_users()
            user_data = users.get(username)
            if user_data is None:
                return None
            if not self.verify_password(password, user_data["hashed_password"]):
                return None
            return TokenData(
                username=username,
                user_id=user_data["user_id"],
            )


_default_auth = AuthHandler()


def get_auth_handler() -> AuthHandler:
    return _default_auth


def set_auth_handler(handler: AuthHandler):
    global _default_auth
    _default_auth = handler


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    auth: AuthHandler = Depends(get_auth_handler),
) -> TokenData:
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token_data = auth.decode_token(credentials.credentials)
    if token_data is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return token_data


async def OptionalAuth(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    auth: AuthHandler = Depends(get_auth_handler),
) -> Optional[TokenData]:
    if credentials is None:
        return None
    return auth.decode_token(credentials.credentials)
