"""
FastAPI application entrypoint for the NLP service.

Run locally with:
    uvicorn app.main:app --reload --port 8000

This started as a Phase 3 service skeleton and now (Phase 5) hosts a
real multi-module NLP pipeline: sentiment, emotion, and toxicity, with
summarization/topic to follow.
"""
import asyncio

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import api_router
from app.config import get_settings
from app.inference.warmup import warm_up_all_models
from app.utils.logging import get_logger

logger = get_logger(__name__)


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title=settings.service_name,
        version=settings.service_version,
        description="Multi-module NLP service for real-time conversational analysis.",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(api_router)

    @app.on_event("startup")
    async def on_startup() -> None:
        logger.info(
            "%s v%s starting up (host=%s, port=%s)",
            settings.service_name,
            settings.service_version,
            settings.host,
            settings.port,
        )
        # Every chat message runs through all currently registered models,
        # so warm them up in a background thread now rather than paying
        # the load cost on the first real message. /health stays
        # responsive immediately since this doesn't block the event loop.
        loop = asyncio.get_event_loop()
        loop.run_in_executor(None, warm_up_all_models)

    return app


app = create_app()
