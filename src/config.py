from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    app_env: str = os.getenv("GENMIND_APP_ENV", "dev")
    app_name: str = os.getenv("GENMIND_APP_NAME", "genmind-mcp")
    app_version: str = os.getenv("GENMIND_APP_VERSION", "0.2.0")

    db_host: str = os.getenv("GENMIND_DB_HOST", "127.0.0.1")
    db_port: int = int(os.getenv("GENMIND_DB_PORT", "5432"))
    db_user: str = os.getenv("GENMIND_DB_USER", "genmind")
    db_password: str = os.getenv("GENMIND_DB_PASSWORD", "genmind")
    db_name: str = os.getenv("GENMIND_DB_NAME", "genmind")

    jwt_algorithm: str = os.getenv("GENMIND_JWT_ALGORITHM", "HS256")
    jwt_secret: str = os.getenv("GENMIND_JWT_SECRET", "dev-secret-change-me")
    jwt_issuer: str = os.getenv("GENMIND_JWT_ISSUER", "genmind")
    jwt_audience: str = os.getenv("GENMIND_JWT_AUDIENCE", "genmind-api")

    @property
    def db_dsn(self) -> str:
        return (
            f"postgresql://{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )


settings = Settings()
