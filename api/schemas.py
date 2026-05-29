from typing import Dict, List, Literal
from pydantic import BaseModel, Field, field_validator

class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10_000)
    model: str = Field(default="sentiment_ensemble")
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)

    @field_validator("text")
    @classmethod
    def not_empty(cls, v):
        if not v.strip():
            raise ValueError("text cannot be whitespace only")
        return v.strip()

class BatchPredictRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, max_length=100)
    model: str = Field(default="sentiment_ensemble")
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)

class Probabilities(BaseModel):
    positive: float
    negative: float

class PredictResponse(BaseModel):
    label: Literal["positive", "negative"]
    confidence: float
    confidence_level: Literal["high", "medium", "low"]
    probabilities: Probabilities
    model: str
    threshold_used: float
    text_length: int

class BatchItem(BaseModel):
    label: Literal["positive", "negative"]
    confidence: float
    confidence_level: Literal["high", "medium", "low"]
    probabilities: Probabilities

class BatchResponse(BaseModel):
    predictions: List[BatchItem]
    count: int
    summary: Dict[str, int]   # {"positive": N, "negative": M}

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool