"""
Centralized configuration for the NLP service.

Values are read from environment variables (see .env.example). Keeping all
settings in one place makes it easy to see what the service depends on and
avoids scattering os.getenv() calls throughout the codebase.
"""
import os
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()


class Settings:
    """Runtime configuration for the NLP service."""

    # Network
    host: str = os.getenv("NLP_SERVICE_HOST", "0.0.0.0")
    port: int = int(os.getenv("NLP_SERVICE_PORT", "8000"))

    # CORS - the Node backend (and, during development, the Vite dev server)
    # need to be able to call this service directly if desired.
    allowed_origins: list[str] = [
        origin.strip()
        for origin in os.getenv("NLP_ALLOWED_ORIGINS", "*").split(",")
        if origin.strip()
    ]

    # Logging
    log_level: str = os.getenv("NLP_LOG_LEVEL", "INFO")

    # Service metadata
    service_name: str = "nlp-service"
    service_version: str = "0.1.0"


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance (avoids re-reading env vars per request)."""
    return Settings()
