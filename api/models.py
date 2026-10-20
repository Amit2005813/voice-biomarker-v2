"""Pydantic models for request/response validation."""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Generic, Optional, TypeVar

from pydantic import BaseModel, Field, field_validator

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class UserRole(str, Enum):
    ADMIN = "admin"
    CLINICIAN = "clinician"
    RESEARCHER = "researcher"


class AnalysisStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class ModelStatus(str, Enum):
    NOT_TRAINED = "not_trained"
    TRAINING = "training"
    READY = "ready"
    FAILED = "failed"


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

class ModelInfo(BaseModel):
    status: ModelStatus = ModelStatus.NOT_TRAINED
    version: Optional[str] = None
    last_trained: Optional[datetime] = None
    accuracy: Optional[float] = None


class HealthResponse(BaseModel):
    status: str = "healthy"
    version: str = "2.0.0"
    uptime_seconds: float = 0.0
    model: ModelInfo = Field(default_factory=ModelInfo)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Upload / Analysis
# ---------------------------------------------------------------------------

class UploadResponse(BaseModel):
    file_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    filename: str
    size_bytes: int
    content_type: str
    message: str = "File uploaded successfully"


class BiomarkerResult(BaseModel):
    name: str
    value: float
    confidence: float = Field(ge=0.0, le=1.0)
    interpretation: Optional[str] = None


class AnalysisResponse(BaseModel):
    analysis_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    status: AnalysisStatus = AnalysisStatus.PENDING
    filename: Optional[str] = None
    file_id: Optional[str] = None
    user_id: Optional[str] = None
    features: dict[str, Any] = Field(default_factory=dict)
    biomarkers: Any = Field(default_factory=dict)
    risk_score: Optional[float] = None
    recommendations: list[str] = Field(default_factory=list)
    processing_time_ms: Optional[float] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    error: Optional[str] = None


class AnalyzeRequest(BaseModel):
    file_id: Optional[str] = None
    filename: Optional[str] = None
    extra_metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------

class LoginRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=64)
    password: str = Field(..., min_length=6, max_length=128)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int = Field(default=3600, description="Seconds until expiry")
    role: UserRole


class UserCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=64, pattern=r"^[a-zA-Z0-9_]+$")
    password: str = Field(..., min_length=6, max_length=128)
    role: UserRole = UserRole.CLINICIAN

    @field_validator("password")
    @classmethod
    def password_strength(cls, v: str) -> str:
        if not any(c.isupper() for c in v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain at least one digit")
        return v


class UserResponse(BaseModel):
    id: str
    username: str
    role: UserRole
    created_at: datetime
    is_active: bool = True


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

class TrainRequest(BaseModel):
    model_type: str = Field(default="biomarker_classifier", max_length=128)
    epochs: int = Field(default=50, ge=1, le=1000)
    learning_rate: float = Field(default=0.001, gt=0.0, le=1.0)
    batch_size: int = Field(default=32, ge=1, le=512)
    extra_params: dict[str, Any] = Field(default_factory=dict)


class TrainStatusResponse(BaseModel):
    status: ModelStatus
    model_type: str
    progress: float = Field(ge=0.0, le=100.0, default=0.0)
    current_epoch: int = 0
    total_epochs: int = 0
    loss: Optional[float] = None
    accuracy: Optional[float] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Pagination
# ---------------------------------------------------------------------------

class PaginationParams(BaseModel):
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=20, ge=1, le=100)


class AnalysisListResponse(BaseModel, Generic[T]):
    items: list[T]
    total: int
    page: int
    page_size: int
    total_pages: int


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class ErrorDetail(BaseModel):
    code: str
    message: str
    details: Optional[dict[str, Any]] = None


class ErrorResponse(BaseModel):
    error: ErrorDetail
    request_id: Optional[str] = None


class ValidationResponse(BaseModel):
    valid: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Cache statistics
# ---------------------------------------------------------------------------

class CacheStats(BaseModel):
    backend: str = "memory"
    total_keys: int = 0
    hits: int = 0
    misses: int = 0
    hit_rate: float = 0.0
    memory_usage_bytes: int = 0
