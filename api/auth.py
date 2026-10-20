"""Authentication, authorization, and rate-limiting utilities."""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import secrets
import time
from collections import defaultdict
from datetime import datetime, timedelta
from functools import wraps
from typing import Any, Callable, Optional

from fastapi import Depends, Header, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from api import database as db
from api.models import UserRole

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration (env-var driven with safe defaults)
# ---------------------------------------------------------------------------

_JWT_SECRET = os.getenv("JWT_SECRET", secrets.token_urlsafe(48))
_JWT_ALGORITHM = "HS256"
_JWT_EXPIRY_HOURS = int(os.getenv("JWT_EXPIRY_HOURS", "24"))
_API_KEY_HEADER = "X-API-Key"
_API_KEYS: dict[str, dict[str, Any]] = {}  # populated from env or DB


def _hash_password(password: str) -> str:
    salt = os.getenv("PASSWORD_SALT", "voice-biomarker-v2-default-salt")
    return hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100_000).hex()


def _verify_password(password: str, password_hash: str) -> bool:
    return hmac.compare_digest(_hash_password(password), password_hash)


# ---------------------------------------------------------------------------
# JWT helpers (no external dependency – lightweight HMAC-SHA256 JWT)
# ---------------------------------------------------------------------------

def _b64url_encode(data: bytes) -> str:
    import base64
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _b64url_decode(s: str) -> bytes:
    import base64
    padding = 4 - len(s) % 4
    s += "=" * padding
    return base64.urlsafe_b64decode(s)


def generate_token(user_id: str, role: str, expires_hours: Optional[int] = None) -> str:
    """Create a HS256 JWT."""
    import json
    expires_hours = expires_hours or _JWT_EXPIRY_HOURS
    header = _b64url_encode(json.dumps({"alg": _JWT_ALGORITHM, "typ": "JWT"}).encode())
    payload_data = {
        "sub": user_id,
        "role": role,
        "iat": int(time.time()),
        "exp": int(time.time()) + expires_hours * 3600,
    }
    payload = _b64url_encode(json.dumps(payload_data).encode())
    signature = _b64url_encode(
        hmac.new(
            _JWT_SECRET.encode(),
            f"{header}.{payload}".encode(),
            hashlib.sha256,
        ).digest()
    )
    return f"{header}.{payload}.{signature}"


def verify_token(token: str) -> Optional[dict[str, Any]]:
    """Decode and verify a JWT. Returns payload dict or None."""
    import json
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return None
        header_b64, payload_b64, signature_b64 = parts
        expected_sig = _b64url_encode(
            hmac.new(
                _JWT_SECRET.encode(),
                f"{header_b64}.{payload_b64}".encode(),
                hashlib.sha256,
            ).digest()
        )
        if not hmac.compare_digest(signature_b64, expected_sig):
            return None
        payload = json.loads(_b64url_decode(payload_b64))
        if payload.get("exp", 0) < time.time():
            return None
        return payload
    except Exception as exc:
        logger.debug("JWT verification failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# API Key helpers
# ---------------------------------------------------------------------------

def register_api_key(
    key: str,
    *,
    user_id: str,
    role: str = "researcher",
    description: str = "",
) -> None:
    _API_KEYS[key] = {
        "user_id": user_id,
        "role": role,
        "description": description,
        "created_at": datetime.utcnow().isoformat(),
    }


def verify_api_key(key: str) -> Optional[dict[str, Any]]:
    return _API_KEYS.get(key)


# ---------------------------------------------------------------------------
# Rate limiting (in-memory sliding window)
# ---------------------------------------------------------------------------

class RateLimiter:
    """Simple in-memory rate limiter with per-user/per-endpoint windows."""

    def __init__(self) -> None:
        self._windows: dict[str, list[float]] = defaultdict(list)

    def is_allowed(
        self,
        key: str,
        max_requests: int = 60,
        window_seconds: int = 60,
    ) -> bool:
        now = time.time()
        window = self._windows[key]
        cutoff = now - window_seconds
        self._windows[key] = [t for t in window if t > cutoff]
        if len(self._windows[key]) >= max_requests:
            return False
        self._windows[key].append(now)
        return True

    def remaining(self, key: str, max_requests: int = 60, window_seconds: int = 60) -> int:
        now = time.time()
        window = self._windows[key]
        cutoff = now - window_seconds
        current = sum(1 for t in window if t > cutoff)
        return max(0, max_requests - current)


rate_limiter = RateLimiter()

# Default limits per role
_ROLE_LIMITS: dict[str, tuple[int, int]] = {
    "admin": (300, 60),
    "clinician": (120, 60),
    "researcher": (60, 60),
    "anonymous": (20, 60),
}


# ---------------------------------------------------------------------------
# FastAPI dependencies
# ---------------------------------------------------------------------------

_bearer_scheme = HTTPBearer(auto_error=False)


async def _resolve_identity(
    request: Request,
    authorization: Optional[HTTPAuthorizationCredentials] = Depends(_bearer_scheme),
    x_api_key: Optional[str] = Header(None, alias=_API_KEY_HEADER),
) -> dict[str, Any]:
    """Resolve the calling user from JWT or API key. Returns identity dict."""
    # JWT
    if authorization and authorization.credentials:
        payload = verify_token(authorization.credentials)
        if payload:
            return {
                "user_id": payload["sub"],
                "role": payload.get("role", "clinician"),
                "method": "jwt",
            }

    # API key
    if x_api_key:
        info = verify_api_key(x_api_key)
        if info:
            return {
                "user_id": info["user_id"],
                "role": info["role"],
                "method": "api_key",
            }

    return {"user_id": "anonymous", "role": "anonymous", "method": "none"}


async def require_auth(
    request: Request,
    identity: dict[str, Any] = Depends(_resolve_identity),
) -> dict[str, Any]:
    """Enforce that a valid JWT or API key is present."""
    if identity["method"] == "none":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Provide a Bearer token or API key.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    endpoint = request.url.path
    role = identity["role"]
    max_req, window = _ROLE_LIMITS.get(role, (60, 60))
    rate_key = f"{identity['user_id']}:{endpoint}"
    if not rate_limiter.is_allowed(rate_key, max_req, window):
        remaining = rate_limiter.remaining(rate_key, max_req, window)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded for {endpoint}. Try again later.",
            headers={"X-RateLimit-Remaining": str(remaining)},
        )
    return identity


async def require_admin(
    identity: dict[str, Any] = Depends(require_auth),
) -> dict[str, Any]:
    if identity["role"] != UserRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin role required.",
        )
    return identity


# ---------------------------------------------------------------------------
# Auth router (mounted in main.py)
# ---------------------------------------------------------------------------

from fastapi import APIRouter  # noqa: E402  (avoid circular import at module level)

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/login")
async def login(body: dict[str, Any]) -> dict[str, Any]:
    username = body.get("username", "")
    password = body.get("password", "")
    if not username or not password:
        raise HTTPException(status_code=400, detail="username and password required")

    user = db.get_user_by_username(username)
    if user is None or not _verify_password(password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    if not user.get("is_active", True):
        raise HTTPException(status_code=403, detail="Account disabled")

    token = generate_token(user["id"], user["role"])
    return {
        "access_token": token,
        "token_type": "bearer",
        "expires_in": _JWT_EXPIRY_HOURS * 3600,
        "role": user["role"],
    }


@router.post("/register")
async def register(body: dict[str, Any]) -> dict[str, Any]:
    username = body.get("username", "")
    password = body.get("password", "")
    role = body.get("role", "clinician")

    if not username or not password:
        raise HTTPException(status_code=400, detail="username and password required")

    if db.get_user_by_username(username):
        raise HTTPException(status_code=409, detail="Username already exists")

    try:
        UserRole(role)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid role: {role}")

    user = db.create_user(
        username=username,
        password_hash=_hash_password(password),
        role=role,
    )
    return {
        "id": user["id"],
        "username": user["username"],
        "role": user["role"],
        "created_at": user["created_at"],
    }


@router.get("/users")
async def list_users(identity: dict[str, Any] = Depends(require_admin)) -> list[dict[str, Any]]:
    return db.list_users()
