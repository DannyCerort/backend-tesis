from fastapi import APIRouter, HTTPException, Query

from app.core.config import settings
from app.services.s3_service import object_exists
from app.services.silver_service import build_silver, get_silver_preview

router = APIRouter(prefix="/api/silver", tags=["silver"])


@router.get("/status")
def silver_status():
    return {
        "bucket": settings.s3_bucket,
        "bronze_key": settings.bronze_key,
        "silver_key": settings.silver_key,
        "bronze_exists": object_exists(settings.s3_bucket, settings.bronze_key),
        "silver_exists": object_exists(settings.s3_bucket, settings.silver_key),
    }


@router.post("/build")
def create_silver():
    try:
        return build_silver()
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creando silver: {e}")


@router.get("/preview")
def silver_preview(limit: int = Query(default=10, ge=1, le=100)):
    try:
        return get_silver_preview(limit=limit)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error leyendo silver: {e}")