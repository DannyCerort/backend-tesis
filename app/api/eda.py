from fastapi import APIRouter, HTTPException, Query

from app.services.eda_service import (
    get_eda_options,
    get_eda_kpis,
    get_eda_desfase_distribution,
    get_eda_prod_vs_fact,
    get_eda_late_summary,
    get_eda_late_heatmap,
)

router = APIRouter(prefix="/api/eda", tags=["eda"])


@router.get("/options")
def eda_options():
    try:
        return get_eda_options()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo opciones EDA: {e}")


@router.get("/kpis")
def eda_kpis(scope: str = Query(...)):
    try:
        return get_eda_kpis(scope)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo KPIs EDA: {e}")


@router.get("/desfase-distribution")
def eda_desfase_distribution(scope: str = Query(...)):
    try:
        return get_eda_desfase_distribution(scope)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo distribución de desfase: {e}")


@router.get("/prod-vs-fact")
def eda_prod_vs_fact(scope: str = Query(...)):
    try:
        return get_eda_prod_vs_fact(scope)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo producción vs facturación: {e}")


@router.get("/late-summary")
def eda_late_summary(
    scope: str = Query(...),
    min_delay: int = Query(2, ge=0, le=24),
):
    try:
        return get_eda_late_summary(scope, min_delay)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo resumen de facturación tardía: {e}")


@router.get("/late-heatmap")
def eda_late_heatmap(
    scope: str = Query(...),
    min_delay: int = Query(2, ge=0, le=24),
    max_delay: int = Query(12, ge=1, le=36),
):
    try:
        return get_eda_late_heatmap(scope, min_delay, max_delay)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo heatmap tardío: {e}")