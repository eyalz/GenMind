from __future__ import annotations

import hashlib
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any

import jwt
from fastapi import HTTPException, Request, status

from src.config import settings
from src.models.schemas import TenantContext


def hash_api_key(raw_key: str) -> tuple[str, str]:
    digest = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()
    return digest, raw_key[:8]


def issue_api_key() -> str:
    return f"gm_{secrets.token_urlsafe(32)}"


def create_jwt(
    customer_id: str,
    workspace_id: str,
    scopes: list[str],
    *,
    end_user_id: str = "system",
    session_id: str = "system",
    expires_minutes: int = 60,
) -> str:
    now = datetime.now(timezone.utc)
    payload: dict[str, Any] = {
        "iss": settings.jwt_issuer,
        "aud": settings.jwt_audience,
        "iat": int(now.timestamp()),
        "nbf": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=expires_minutes)).timestamp()),
        "customer_id": customer_id,
        "workspace_id": workspace_id,
        "end_user_id": end_user_id,
        "session_id": session_id,
        "scopes": scopes,
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def decode_jwt(token: str) -> dict[str, Any]:
    try:
        claims = jwt.decode(
            token,
            settings.jwt_secret,
            algorithms=[settings.jwt_algorithm],
            audience=settings.jwt_audience,
            issuer=settings.jwt_issuer,
        )
    except jwt.PyJWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {exc}",
        ) from exc

    for required in ("customer_id", "workspace_id", "scopes"):
        if required not in claims:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Missing claim: {required}",
            )

    return claims


def require_auth(request: Request) -> dict[str, Any]:
    auth_header = request.headers.get("authorization", "")
    if not auth_header.lower().startswith("bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Bearer token.",
        )

    token = auth_header.split(" ", 1)[1].strip()
    claims = decode_jwt(token)
    request.state.auth_claims = claims
    return claims


def require_scope(request: Request, accepted: set[str]) -> dict[str, Any]:
    claims = require_auth(request)
    scopes = set(claims.get("scopes", []))
    if "admin:*" in scopes:
        return claims
    if not scopes.intersection(accepted):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient scope.",
        )
    return claims


def tenant_from_claims(claims: dict[str, Any], *, session_id: str | None = None) -> TenantContext:
    claim_session = claims.get("session_id") or "system"
    return TenantContext(
        customer_id=claims["customer_id"],
        workspace_id=claims["workspace_id"],
        end_user_id=claims.get("end_user_id", "system"),
        session_id=session_id or claim_session,
    )
