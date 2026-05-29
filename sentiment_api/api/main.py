import time
from collections import Counter
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import (
    BatchItem, BatchResponse, HealthResponse,
    Probabilities, PredictRequest, PredictResponse, BatchPredictRequest,
)
from src.predict import load_model, predict_single, predict_batch, DEFAULT_MODEL

MODELS_DIR = Path("models")


# ── CONCEPT: Lifespan = startup/shutdown hook ─────────────
# Load the model ONCE when the server starts, not on each request.
# The @lru_cache in predict.py handles the actual caching.
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting up — warming model cache...")
    try:
        load_model(DEFAULT_MODEL)
        print("Model ready ✓")
    except FileNotFoundError:
        print("WARNING: No trained model found. Run `python -m src.train` first.")
    yield
    print("Shutting down.")


app = FastAPI(
    title="Sentiment Analysis API",
    description="Day 1 — ML Pipeline from scratch to production API.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# Middleware: add response time header to every response
@app.middleware("http")
async def timing_middleware(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    ms = (time.perf_counter() - t0) * 1000
    response.headers["X-Response-Time-Ms"] = f"{ms:.2f}"
    return response


# Background task: log predictions without blocking the response
def log_prediction(label: str, confidence: float, text_len: int):
    print(f"[LOG] label={label} | confidence={confidence:.3f} | text_len={text_len}")


@app.get("/")
async def root():
    return {"message": "Sentiment Analysis API", "docs": "/docs"}


@app.get("/health", response_model=HealthResponse)
async def health():
    loaded = (MODELS_DIR / f"{DEFAULT_MODEL}.joblib").exists()
    return HealthResponse(status="healthy", model_loaded=loaded)


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest, background_tasks: BackgroundTasks):
    try:
        result = predict_single(req.text, req.model, req.threshold)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    background_tasks.add_task(log_prediction, result["label"], result["confidence"], len(req.text))

    return PredictResponse(
        label=result["label"],
        confidence=result["confidence"],
        confidence_level=result["confidence_level"],
        probabilities=Probabilities(**result["probabilities"]),
        model=result["model"],
        threshold_used=result["threshold_used"],
        text_length=len(req.text),
    )


@app.post("/predict/batch", response_model=BatchResponse)
async def predict_batch_route(req: BatchPredictRequest, background_tasks: BackgroundTasks):
    try:
        results = predict_batch(req.texts, req.model, req.threshold)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    summary = dict(Counter(r["label"] for r in results))
    background_tasks.add_task(log_prediction, "batch", 0.0, sum(len(t) for t in req.texts))

    return BatchResponse(
        predictions=[
            BatchItem(
                label=r["label"],
                confidence=r["confidence"],
                confidence_level=r["confidence_level"],
                probabilities=Probabilities(**r["probabilities"]),
            ) for r in results
        ],
        count=len(results),
        summary=summary,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)