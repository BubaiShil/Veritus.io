"""
Veritus — FastAPI Backend Application (backend/api.py)

Exposes REST API endpoints for:
  - System health and service status checks (Ollama, PostgreSQL)
  - Dataset analysis orchestration (runs AHGT pipeline with live progress tracking)
  - Dataset and hypothesis history retrieval
"""

import asyncio
import io
import logging
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Ensure project root is in Python sys.path
backend_dir = Path(__file__).resolve().parent
project_root = backend_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from backend.ai_agent import (
    is_ollama_available,
    is_llm_available,
    get_active_llm_provider,
)
from backend.db import (
    get_connection,
    get_dataset,
    get_hypotheses_by_dataset,
    init_db,
    list_datasets,
)
from backend.main import run_analysis_pipeline
from backend.query_engine import answer_dataset_query

# Cache active and historical datasets for on-demand conversational Q&A
in_memory_dfs: Dict[int, pd.DataFrame] = {}
last_active_df: Optional[pd.DataFrame] = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("veritus.api")

# -----------------------------------------------------------------------------
# In-Memory Job Store for Progress Tracking
# -----------------------------------------------------------------------------
# Structure: { job_id: { "job_id": str, "status": str, "progress": float, "message": str, "result": Any, "error": str, "created_at": float, "updated_at": float } }
jobs_store: Dict[str, Dict[str, Any]] = {}

def cleanup_stale_jobs(max_age_seconds: int = 86400, max_jobs: int = 200) -> None:
    """Removes completed/failed jobs older than max_age_seconds or trims if too large."""
    now = time.time()
    stale_keys = [
        jid for jid, info in jobs_store.items()
        if now - info.get("updated_at", now) > max_age_seconds
    ]
    for jid in stale_keys:
        jobs_store.pop(jid, None)

    # Trim oldest jobs if still beyond max_jobs
    if len(jobs_store) > max_jobs:
        sorted_jobs = sorted(jobs_store.items(), key=lambda kv: kv[1].get("created_at", 0))
        for jid, _ in sorted_jobs[: len(jobs_store) - max_jobs]:
            jobs_store.pop(jid, None)

def init_job_record(job_id: str) -> Dict[str, Any]:
    record = {
        "job_id": job_id,
        "status": "running",
        "progress": 0.05,
        "message": "Initializing analysis pipeline...",
        "result": None,
        "error": None,
        "created_at": time.time(),
        "updated_at": time.time(),
    }
    jobs_store[job_id] = record
    return record


# -----------------------------------------------------------------------------
# Helper: Database connectivity probe
# -----------------------------------------------------------------------------
def check_db_available() -> bool:
    """Checks if PostgreSQL is accessible and responds to a ping query."""
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1;")
                return True
    except Exception as exc:
        logger.warning(f"Database availability check failed: {exc}")
        return False


# -----------------------------------------------------------------------------
# FastAPI Lifespan & Application Setup
# -----------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Veritus API starting up. Initializing database schema...")
    try:
        init_db()
        logger.info("Database schema initialized successfully.")
    except Exception as exc:
        logger.warning(f"Initial database schema check skipped/failed: {exc}")
    yield
    logger.info("Veritus API shutting down.")


app = FastAPI(
    title="Veritus API",
    description="REST API for Autonomous AI Data Scientist (Automated Hypothesis Generation and Testing)",
    version="1.0.0",
    lifespan=lifespan,
)

# -----------------------------------------------------------------------------
# CORS Configuration
# -----------------------------------------------------------------------------
frontend_origin_env = os.getenv("FRONTEND_ORIGIN", "http://localhost:5173")
# Parse comma-separated list of allowed origins
allowed_origins = [o.strip() for o in frontend_origin_env.split(",") if o.strip()]
# Include common dev origins if not present
for dev_origin in ("http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:3000"):
    if dev_origin not in allowed_origins and "*" not in allowed_origins:
        allowed_origins.append(dev_origin)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins if "*" not in allowed_origins else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------------------------------------------------------
# Health & Status Endpoints
# -----------------------------------------------------------------------------
@app.get("/api/health", summary="Health Check")
def health_check() -> Dict[str, Any]:
    """Simple health check endpoint for Docker and cloud platform healthchecks."""
    return {
        "status": "healthy",
        "service": "veritus-backend",
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }


@app.get("/api/system-status", summary="System Service Status")
def get_system_status() -> Dict[str, Any]:
    """Returns connectivity status for LLM (Gemini, Groq, or Ollama) and PostgreSQL database."""
    llm_ok = is_llm_available(force_refresh=True)
    db_ok = check_db_available()
    provider = get_active_llm_provider()
    return {
        "ollama": bool(llm_ok),  # Backward-compatible boolean for frontend
        "llm": bool(llm_ok),
        "llm_provider": provider,
        "database": bool(db_ok)
    }


# -----------------------------------------------------------------------------
# History Endpoints
# -----------------------------------------------------------------------------
@app.get("/api/history", summary="List Analyzed Datasets")
def get_history() -> Dict[str, Any]:
    """Returns dataset run history and aggregate hypothesis test metrics."""
    try:
        datasets = list_datasets()
        return {"datasets": datasets}
    except Exception as exc:
        logger.error(f"Error fetching dataset history: {exc}", exc_info=True)
        # Fall back to empty list if DB is currently unreachable
        return {"datasets": [], "error": str(exc)}


@app.get("/api/history/{dataset_id}", summary="Get Dataset Details and Hypotheses")
def get_history_detail(dataset_id: int) -> Dict[str, Any]:
    """Returns metadata and tested hypotheses (validated + rejected) for a specific dataset ID."""
    try:
        dataset = get_dataset(dataset_id)
        if not dataset:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Dataset with ID {dataset_id} was not found."
            )
        hypotheses = get_hypotheses_by_dataset(dataset_id)
        validated = [h for h in hypotheses if h.get("verdict") == "validated"]
        rejected = [h for h in hypotheses if h.get("verdict") == "rejected"]
        return {
            "dataset": dataset,
            "hypotheses": hypotheses,
            "validated": validated,
            "rejected": rejected,
            "total_hypotheses": len(hypotheses),
            "validated_count": len(validated),
            "rejected_count": len(rejected)
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Error fetching dataset detail for ID {dataset_id}: {exc}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))


# -----------------------------------------------------------------------------
# Analysis Pipeline & Progress Tracking
# -----------------------------------------------------------------------------
def execute_pipeline_sync(
    df: pd.DataFrame,
    filename: str,
    num_hypotheses: int,
    enable_refinement: bool,
    job_id: str
) -> Dict[str, Any]:
    """Worker function executed in worker thread to run the analysis pipeline."""
    def progress_callback(msg: str, frac: float):
        if job_id in jobs_store:
            jobs_store[job_id]["progress"] = round(frac, 2)
            jobs_store[job_id]["message"] = msg
            jobs_store[job_id]["updated_at"] = time.time()

    try:
        results = run_analysis_pipeline(
            df=df,
            dataset_name=filename,
            num_initial_hypotheses=num_hypotheses,
            enable_refinement=enable_refinement,
            progress_callback=progress_callback
        )
        global last_active_df
        last_active_df = df
        if isinstance(results, dict) and "dataset_id" in results:
            in_memory_dfs[results["dataset_id"]] = df

        if job_id in jobs_store:
            jobs_store[job_id]["status"] = "completed"
            jobs_store[job_id]["progress"] = 1.0
            jobs_store[job_id]["message"] = "Analysis pipeline completed successfully!"
            jobs_store[job_id]["result"] = results
            jobs_store[job_id]["updated_at"] = time.time()
        return results
    except Exception as exc:
        logger.error(f"Pipeline execution failed for job {job_id}: {exc}", exc_info=True)
        if job_id in jobs_store:
            jobs_store[job_id]["status"] = "failed"
            jobs_store[job_id]["error"] = str(exc)
            jobs_store[job_id]["message"] = f"Analysis failed: {str(exc)}"
            jobs_store[job_id]["updated_at"] = time.time()
        raise


@app.get("/api/analyze/{job_id}/status", summary="Get Analysis Job Status")
def get_analysis_status(job_id: str) -> Dict[str, Any]:
    """Returns the live progress, status, and result/error for an active or completed analysis job."""
    if job_id not in jobs_store:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Analysis job '{job_id}' not found."
        )
    return jobs_store[job_id]


@app.post("/api/analyze", summary="Run Analysis on Uploaded CSV")
async def analyze_dataset(
    file: UploadFile = File(...),
    num_hypotheses: int = Query(6, ge=1, le=15, description="Number of initial hypotheses to formulate"),
    enable_refinement: bool = Query(True, description="Enable automated refinement loop for failed tests"),
    job_id: Optional[str] = Query(None, description="Optional custom job ID for progress tracking"),
    run_async: bool = Query(False, description="If true, returns job_id immediately for polling")
) -> Any:
    """
    Accepts a multipart CSV file upload and triggers the Automated Hypothesis Generation and Testing loop.
    
    - If `run_async=False` (default): Runs pipeline and returns the full results JSON upon completion.
      During execution, clients can concurrently poll `GET /api/analyze/{job_id}/status` for live progress.
    - If `run_async=True`: Initiates background task and immediately returns `{"job_id": "...", "status": "running"}`.
    """
    cleanup_stale_jobs()

    # Validate file extension
    filename = file.filename or "uploaded_dataset.csv"
    if not filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file type. Please upload a standard CSV (.csv) file."
        )

    # Read and parse CSV into pandas DataFrame
    try:
        content = await file.read()
        if not content:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="The uploaded CSV file is empty."
            )
        df = pd.read_csv(io.BytesIO(content))
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"CSV parsing error: {exc}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Could not parse CSV file: {str(exc)}"
        )

    if df.empty or len(df.columns) < 2:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The dataset must contain at least 2 columns and 1 row for hypothesis testing."
        )

    # Assign or use provided job ID
    active_job_id = job_id.strip() if job_id and job_id.strip() else str(uuid.uuid4())
    init_job_record(active_job_id)

    if run_async:
        # Launch non-blocking background task
        asyncio.create_task(
            asyncio.to_thread(
                execute_pipeline_sync,
                df,
                filename,
                num_hypotheses,
                enable_refinement,
                active_job_id
            )
        )
        return {
            "job_id": active_job_id,
            "status": "running",
            "message": f"Analysis started for '{filename}'. Poll /api/analyze/{active_job_id}/status for progress."
        }
    else:
        # Run in worker threadpool to avoid blocking FastAPI event loop
        try:
            results = await asyncio.to_thread(
                execute_pipeline_sync,
                df,
                filename,
                num_hypotheses,
                enable_refinement,
                active_job_id
            )
            # Attach job_id for client reference
            if isinstance(results, dict):
                results["job_id"] = active_job_id
            return results
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Analysis pipeline execution failed: {str(exc)}"
            )


# -----------------------------------------------------------------------------
# Natural Language Query Endpoint ("Ask Veritus")
# -----------------------------------------------------------------------------
@app.post("/api/query", summary="Ask Veritus Natural Language Question")
async def ask_veritus(
    question: str = Form(...),
    dataset_id: Optional[int] = Form(None),
    file: Optional[UploadFile] = File(None)
) -> Dict[str, Any]:
    """
    Answers plain-English questions about a dataset by formulating hypothesis tests,
    executing frequentist tests on the fly, ranking features, or computing summary metrics.
    """
    global last_active_df
    target_df: Optional[pd.DataFrame] = None
    target_name = "Dataset"

    # 1. Parse uploaded file if provided
    if file:
        try:
            content = await file.read()
            if content:
                target_df = pd.read_csv(io.BytesIO(content))
                target_name = file.filename or target_name
                last_active_df = target_df
        except Exception as exc:
            logger.warning(f"Failed to parse uploaded CSV for query: {exc}")

    # 2. Check memory cache or disk cache by dataset_id
    if target_df is None and dataset_id is not None:
        if dataset_id in in_memory_dfs:
            target_df = in_memory_dfs[dataset_id]
        else:
            cache_file = project_root / "data_cache" / f"{dataset_id}.csv"
            if cache_file.exists():
                try:
                    target_df = pd.read_csv(cache_file)
                    in_memory_dfs[dataset_id] = target_df
                except Exception as exc:
                    logger.warning(f"Failed to read disk cache for dataset {dataset_id}: {exc}")

    # 3. Fallback to last active dataset in session
    if target_df is None and last_active_df is not None:
        target_df = last_active_df

    if target_df is None or target_df.empty:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No dataset available to query. Please upload a CSV file or analyze a dataset first."
        )

    try:
        response = answer_dataset_query(target_df, question, dataset_name=target_name)
        return response
    except Exception as exc:
        logger.error(f"Error answering dataset query: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query execution failed: {str(exc)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.api:app", host="0.0.0.0", port=8000, reload=True)
