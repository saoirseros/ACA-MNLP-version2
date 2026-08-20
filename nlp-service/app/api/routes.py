"""
Aggregates all API routers so main.py only needs a single import.

As new endpoints are added (e.g. /analyze in Phase 4), register their
routers here rather than growing main.py directly.
"""
from fastapi import APIRouter

from app.api import analysis, context, conversation, health

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(analysis.router)
api_router.include_router(conversation.router)
api_router.include_router(context.router)
