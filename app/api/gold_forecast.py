import threading
import uuid

from fastapi import APIRouter, HTTPException

from app.core.config import settings
from app.services.s3_service import object_exists
from app.services.job_store import create_job, get_job
from app.services.gold_forecast_service import run_gold_forecast_job

router = APIRouter(prefix="/api/gold/forecast", tags=["gold-forecast"])


@router.get("/status")
def gold_forecast_status():
    return {
        "bucket": settings.s3_bucket,
        "silver_key": settings.silver_key,
        "gold_series_key": settings.gold_series_key,
        "gold_forecast_key": settings.gold_forecast_key,
        "silver_exists": object_exists(settings.s3_bucket, settings.silver_key),
        "series_exists": object_exists(settings.s3_bucket, settings.gold_series_key),
        "forecast_exists": object_exists(settings.s3_bucket, settings.gold_forecast_key),
        "ranking_exists": object_exists(settings.s3_bucket, settings.gold_ranking_key),
    }


@router.post("/build")
def build_gold_forecast():
    job_id = str(uuid.uuid4())
    create_job(job_id)

    thread = threading.Thread(
        target=run_gold_forecast_job,
        args=(job_id,),
        daemon=True,
    )
    thread.start()

    return {
        "status": "accepted",
        "job_id": job_id,
        "message": "Proceso de forecast gold iniciado",
    }


@router.get("/jobs/{job_id}")
def get_gold_job_status(job_id: str):
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job no encontrado")
    return job