from fastapi import APIRouter, HTTPException, Query

from app.core.config import settings
from app.services.bronze_service import build_bronze, get_bronze_preview
from app.services.s3_service import object_exists

router = APIRouter(prefix="/api/bronze", tags=["bronze"])


@router.get("/status")
def bronze_status():
    raw_exists = object_exists(settings.s3_bucket, settings.raw_key)
    bronze_exists = object_exists(settings.s3_bucket, settings.bronze_key)

    return {
        "bucket": settings.s3_bucket,
        "raw_key": settings.raw_key,
        "bronze_key": settings.bronze_key,
        "raw_exists": raw_exists,
        "bronze_exists": bronze_exists,
    }


@router.post("/build")
def create_bronze():
    try:
        return build_bronze()
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creando bronze: {e}")


@router.get("/preview")
def bronze_preview(limit: int = Query(default=10, ge=1, le=100)):
    try:
        return get_bronze_preview(limit=limit)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error leyendo bronze: {e}")