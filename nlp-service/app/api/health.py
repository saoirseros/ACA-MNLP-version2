"""
Health-check route.

The Node backend polls this endpoint to decide whether the NLP service is
reachable. If it is not, chat must keep working without analysis (see
PROJECT spec: "NORMAL CHAT MUST STILL WORK").
"""
from fastapi import APIRouter

from app.config import get_settings
from app.schemas.health import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    settings = get_settings()
    return HealthResponse(
        status="ok",
        service=settings.service_name,
        version=settings.service_version,
    )
