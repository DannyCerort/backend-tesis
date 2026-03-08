from fastapi import APIRouter, HTTPException, Query

from app.services.forecast_query_service import (
    get_forecast_options,
    get_forecast_prediction,
    get_forecast_model_ranking,
    get_forecast_montecarlo,
)

router = APIRouter(prefix="/api/forecast", tags=["forecast-query"])


@router.get("/options")
def forecast_options():
    try:
        return get_forecast_options()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo opciones de forecast: {e}")


@router.get("/prediction")
def forecast_prediction(
    scope: str = Query(...),
    h: int = Query(6, ge=1, le=12),
):
    try:
        return get_forecast_prediction(scope, h)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo forecast: {e}")


@router.get("/model-ranking")
def forecast_model_ranking(scope: str = Query(...)):
    try:
        return get_forecast_model_ranking(scope)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo ranking de modelos: {e}")


@router.get("/montecarlo")
def forecast_montecarlo(
    scope: str = Query(...),
    h: int = Query(6, ge=1, le=12),
    n_sims: int = Query(3000, ge=100, le=20000),
    seed: int = Query(42),
):
    try:
        return get_forecast_montecarlo(scope, h, n_sims, seed)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo Monte Carlo: {e}")