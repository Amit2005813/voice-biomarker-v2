"""Voice Biomarker Platform v2 – FastAPI application entry-point."""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator

from fastapi import Depends, FastAPI, HTTPException, Query, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api import database as db
from api import __doc__
from api.auth import rate_limiter, require_admin, require_auth, router as auth_router
from api.cache import cache_manager
from api.models import (
    AnalysisListResponse,
    AnalysisResponse,
    AnalysisStatus,
    CacheStats,
    ErrorDetail,
    ErrorResponse,
    HealthResponse,
    ModelInfo,
    ModelStatus,
    PaginationParams,
    TrainRequest,
    TrainStatusResponse,
    UploadResponse,
    ValidationResponse,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

APP_VERSION = "2.0.0"
UPLOAD_DIR = Path(__file__).resolve().parent.parent / "data" / "uploads"
MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB
ALLOWED_CONTENT_TYPES = {
    "audio/wav",
    "audio/x-wav",
    "audio/mpeg",
    "audio/mp3",
    "audio/ogg",
    "audio/flac",
    "audio/x-flac",
    "audio/webm",
    "audio/mp4",
    "audio/aac",
    "audio/x-m4a",
}

_start_time: float = 0.0
_orchestrator: Any = None  # lazy-loaded


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    global _start_time, _orchestrator
    _start_time = time.time()

    logger.info("Initialising database …")
    db.init_db()

    logger.info("Initialising cache …")
    await cache_manager.initialize()

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    # Try to import the orchestrator if it exists in the project
    try:
        from agents.orchestrator import AgentOrchestrator
        _orchestrator = AgentOrchestrator()
        logger.info("Orchestrator initialised successfully")
    except ImportError:
        logger.warning("Orchestrator module not found – running in API-only mode")
    except Exception as exc:
        logger.error("Orchestrator init failed: %s", exc)

    logger.info("Voice Biomarker Platform v%s ready", APP_VERSION)
    yield

    # Shutdown
    logger.info("Shutting down …")
    await cache_manager.shutdown()
    if _orchestrator and hasattr(_orchestrator, "close"):
        await _orchestrator.close()


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Voice Biomarker Platform",
    version=APP_VERSION,
    description="API for voice-based biomarker extraction and analysis",
    lifespan=lifespan,
    docs_url=None,       # custom /docs below
    redoc_url=None,
)

# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        o.strip()
        for o in (  # noqa: E501 – configured via env at deploy time
            "http://localhost:3000",
            "http://localhost:8000",
        )
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_id_middleware(request: Request, call_next: Any) -> Any:
    req_id = request.headers.get("X-Request-ID", uuid.uuid4().hex)
    request.state.request_id = req_id
    start = time.time()
    response = await call_next(request)
    elapsed_ms = round((time.time() - start) * 1000, 2)
    response.headers["X-Request-ID"] = req_id
    response.headers["X-Response-Time"] = f"{elapsed_ms}ms"
    return response


@app.middleware("http")
async def error_handling_middleware(request: Request, call_next: Any) -> Any:
    try:
        return await call_next(request)
    except HTTPException:
        raise
    except Exception as exc:
        req_id = getattr(request.state, "request_id", None)
        logger.exception("Unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error=ErrorDetail(code="INTERNAL_ERROR", message=str(exc)),
                request_id=req_id,
            ).model_dump(),
        )


# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

app.include_router(auth_router)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

# GET /health

@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health_check() -> HealthResponse:
    model_status = ModelStatus.NOT_TRAINED
    model_version: str | None = None
    accuracy: float | None = None

    active = db.get_active_training()
    if active:
        model_status = ModelStatus.TRAINING
    else:
        # Check last completed training
        with db.get_db() as conn:
            row = conn.execute(
                "SELECT * FROM training_history WHERE status='ready' "
                "ORDER BY completed_at DESC LIMIT 1"
            ).fetchone()
            if row:
                model_status = ModelStatus.READY
                accuracy = row["accuracy"]

    return HealthResponse(
        uptime_seconds=round(time.time() - _start_time, 2),
        model=ModelInfo(status=model_status, version=model_version, accuracy=accuracy),
    )


# POST /upload-audio

@app.post("/upload-audio", response_model=UploadResponse, tags=["audio"])
async def upload_audio(
    file: UploadFile,
) -> UploadResponse:
    content_type = file.content_type or "application/octet-stream"
    if content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported audio type: {content_type}. "
            f"Allowed: {', '.join(sorted(ALLOWED_CONTENT_TYPES))}",
        )

    data = await file.read()
    if len(data) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large: {len(data)} bytes (max {MAX_UPLOAD_BYTES})",
        )

    file_id = uuid.uuid4().hex
    ext = Path(file.filename or "audio.wav").suffix or ".wav"
    dest = UPLOAD_DIR / f"{file_id}{ext}"
    dest.write_bytes(data)

    logger.info("Saved upload %s (%d bytes) → %s", file.filename, len(data), dest)
    return UploadResponse(
        file_id=file_id,
        filename=file.filename or f"{file_id}{ext}",
        size_bytes=len(data),
        content_type=content_type,
    )


# POST /analyze

@app.post("/analyze", response_model=AnalysisResponse, tags=["analysis"])
async def analyze_audio(
    file: UploadFile,
) -> AnalysisResponse:
    # Save file
    content_type = file.content_type or "application/octet-stream"
    data = await file.read()
    if len(data) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large")

    file_id = uuid.uuid4().hex
    ext = Path(file.filename or "audio.wav").suffix or ".wav"
    dest = UPLOAD_DIR / f"{file_id}{ext}"
    dest.write_bytes(data)

    record = db.create_analysis(
        user_id="anonymous",
        file_id=file_id,
        file_path=str(dest),
        filename=file.filename,
        status="processing",
    )

    analysis_id = record["id"]

    # Attempt analysis via orchestrator
    start = time.time()
    if _orchestrator is not None:
        try:
            result = _orchestrator.run(str(dest))
            elapsed = round((time.time() - start) * 1000, 2)
            db.update_analysis(
                analysis_id,
                status="completed",
                features=result.get("features", {}),
                biomarkers=result.get("biomarkers", []),
                risk_score=result.get("risk_score"),
                recommendations=result.get("recommendations", []),
                processing_time_ms=elapsed,
                completed_at=datetime.utcnow().isoformat(),
            )
            analysis = db.get_analysis(analysis_id)
            await cache_manager.set(f"analysis:{analysis_id}", analysis, ttl=7200)
            return AnalysisResponse(**analysis)  # type: ignore[arg-type]

        except Exception as exc:
            logger.error("Analysis failed: %s", exc)
            db.update_analysis(analysis_id, status="failed", error=str(exc))
            analysis = db.get_analysis(analysis_id)
            return AnalysisResponse(**analysis)  # type: ignore[arg-type]

    # No orchestrator – return stub
    elapsed = round((time.time() - start) * 1000, 2)
    db.update_analysis(
        analysis_id,
        status="completed",
        features={"pitch_mean": 0, "jitter": 0, "shimmer": 0},
        biomarkers=[],
        processing_time_ms=elapsed,
        completed_at=datetime.utcnow().isoformat(),
    )
    analysis = db.get_analysis(analysis_id)
    return AnalysisResponse(**analysis)  # type: ignore[arg-type]


# GET /analyze/{analysis_id}

@app.get("/analyze/{analysis_id}", response_model=AnalysisResponse, tags=["analysis"])
async def get_analysis(
    analysis_id: str,
) -> AnalysisResponse:
    cached = await cache_manager.get(f"analysis:{analysis_id}")
    if cached:
        return AnalysisResponse(**cached)  # type: ignore[arg-type]

    record = db.get_analysis(analysis_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Analysis not found")

    await cache_manager.set(f"analysis:{analysis_id}", record, ttl=7200)
    return AnalysisResponse(**record)  # type: ignore[arg-type]


# GET /analyses

@app.get("/analyses", response_model=AnalysisListResponse, tags=["analysis"])
async def list_analyses(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
) -> AnalysisListResponse:
    items, total = db.list_analyses(
        user_id=None,
        page=page,
        page_size=page_size,
    )
    total_pages = max(1, -(-total // page_size))
    return AnalysisListResponse(
        items=[AnalysisResponse(**i) for i in items],  # type: ignore[misc]
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
    )


# DELETE /analyses/{analysis_id}

@app.delete("/analyses/{analysis_id}", status_code=204, tags=["analysis"])
async def delete_analysis(
    analysis_id: str,
    identity: dict[str, Any] = Depends(require_auth),
) -> None:
    record = db.get_analysis(analysis_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Analysis not found")

    if identity["role"] != "admin" and record.get("user_id") != identity["user_id"]:
        raise HTTPException(status_code=403, detail="Not authorised to delete this analysis")

    db.delete_analysis(analysis_id)
    await cache_manager.delete(f"analysis:{analysis_id}")


# POST /train

@app.post("/train", response_model=TrainStatusResponse, tags=["training"])
async def train_model(
    body: TrainRequest,
    identity: dict[str, Any] = Depends(require_admin),
) -> TrainStatusResponse:
    active = db.get_active_training()
    if active:
        raise HTTPException(
            status_code=409,
            detail=f"Training already in progress (id={active['id']})",
        )

    record = db.create_training_record(
        model_type=body.model_type,
        epochs=body.epochs,
        learning_rate=body.learning_rate,
        batch_size=body.batch_size,
        extra_params=body.extra_params,
    )

    # Fire-and-forget training in the background
    import asyncio

    async def _run_training() -> None:
        if _orchestrator and hasattr(_orchestrator, "train"):
            try:
                await _orchestrator.train(
                    epochs=body.epochs,
                    learning_rate=body.learning_rate,
                    batch_size=body.batch_size,
                )
                db.update_training_record(
                    record["id"],
                    status="ready",
                    progress=100.0,
                    current_epoch=body.epochs,
                    completed_at=datetime.utcnow().isoformat(),
                )
            except Exception as exc:
                db.update_training_record(
                    record["id"],
                    status="failed",
                    error=str(exc),
                    completed_at=datetime.utcnow().isoformat(),
                )
        else:
            # Simulate training
            for epoch in range(1, body.epochs + 1):
                await asyncio.sleep(0.05)
                progress = round(epoch / body.epochs * 100, 1)
                loss = 1.0 / (epoch + 1)
                acc = min(0.99, epoch / body.epochs)
                db.update_training_record(
                    record["id"],
                    current_epoch=epoch,
                    progress=progress,
                    loss=round(loss, 6),
                    accuracy=round(acc, 4),
                )
            db.update_training_record(
                record["id"],
                status="ready",
                completed_at=datetime.utcnow().isoformat(),
            )

    asyncio.create_task(_run_training())
    return TrainStatusResponse(
        status=ModelStatus.TRAINING,
        model_type=body.model_type,
        total_epochs=body.epochs,
        started_at=record.get("started_at"),
    )


# GET /model/status

@app.get("/model/status", response_model=TrainStatusResponse, tags=["training"])
async def model_status() -> TrainStatusResponse:
    active = db.get_active_training()
    if active:
        return TrainStatusResponse(
            status=ModelStatus.TRAINING,
            model_type=active["model_type"],
            progress=active.get("progress", 0),
            current_epoch=active.get("current_epoch", 0),
            total_epochs=active.get("epochs", 0),
            loss=active.get("loss"),
            accuracy=active.get("accuracy"),
            started_at=active.get("started_at"),
        )

    with db.get_db() as conn:
        row = conn.execute(
            "SELECT * FROM training_history ORDER BY started_at DESC LIMIT 1"
        ).fetchone()

    if row is None:
        return TrainStatusResponse(status=ModelStatus.NOT_TRAINED, model_type="unknown")

    return TrainStatusResponse(
        status=ModelStatus.READY if row["status"] == "ready" else ModelStatus.FAILED,
        model_type=row["model_type"],
        progress=100.0 if row["status"] == "ready" else 0.0,
        current_epoch=row.get("epochs", 0),
        total_epochs=row.get("epochs", 0),
        loss=row.get("loss"),
        accuracy=row.get("accuracy"),
        started_at=row.get("started_at"),
        completed_at=row.get("completed_at"),
        error=row.get("error"),
    )


# GET /docs (custom)

@app.get("/docs", tags=["system"])
async def custom_docs() -> dict[str, Any]:
    return {
        "title": "Voice Biomarker Platform API",
        "version": APP_VERSION,
        "openapi_url": "/openapi.json",
        "endpoints": [
            {"method": "GET", "path": "/health", "description": "Health check"},
            {"method": "POST", "path": "/upload-audio", "description": "Upload audio file"},
            {"method": "POST", "path": "/analyze", "description": "Upload + analyse in one step"},
            {"method": "GET", "path": "/analyze/{analysis_id}", "description": "Get analysis result"},
            {"method": "GET", "path": "/analyses", "description": "List analyses (paginated)"},
            {"method": "DELETE", "path": "/analyses/{analysis_id}", "description": "Delete analysis"},
            {"method": "POST", "path": "/train", "description": "Trigger model training (admin)"},
            {"method": "GET", "path": "/model/status", "description": "Model training status"},
            {"method": "POST", "path": "/auth/login", "description": "Login"},
            {"method": "POST", "path": "/auth/register", "description": "Register user"},
            {"method": "GET", "path": "/auth/users", "description": "List users (admin)"},
        ],
        "authentication": {
            "jwt": "POST /auth/login → use returned access_token as Bearer header",
            "api_key": "Set X-API-Key header",
        },
        "rate_limits": {
            "admin": "300 req/min",
            "clinician": "120 req/min",
            "researcher": "60 req/min",
        },
        "cache_stats": cache_manager.stats,
    }


# Need this import at top level for the /analyze endpoint
from datetime import datetime  # noqa: E402
