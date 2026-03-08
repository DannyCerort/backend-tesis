from app.mcp.server import mcp
from app.services.eda_service import (
    get_eda_options,
    get_eda_kpis,
    get_eda_prod_vs_fact,
    get_eda_late_summary,
    get_eda_late_heatmap,
    get_eda_desfase_distribution,
)


@mcp.tool()
def eda_options() -> dict:
    """Retorna las opciones de scope disponibles para análisis EDA."""
    return get_eda_options()


@mcp.tool()
def eda_kpis(scope: str) -> dict:
    """Retorna KPIs principales del scope."""
    return get_eda_kpis(scope=scope)


@mcp.tool()
def eda_prod_vs_fact(scope: str) -> dict:
    """Retorna la comparación de producción vs facturación para un scope."""
    return get_eda_prod_vs_fact(scope=scope)


@mcp.tool()
def eda_late_summary(scope: str, min_delay: int = 2) -> dict:
    """Resumen mensual de facturación tardía."""
    if min_delay < 0 or min_delay > 24:
        raise ValueError("min_delay debe estar entre 0 y 24")
    return get_eda_late_summary(scope=scope, min_delay=min_delay)


@mcp.tool()
def eda_late_heatmap(scope: str, min_delay: int = 2, max_delay: int = 12) -> dict:
    """Mapa de calor de retrasos de facturación."""
    if min_delay < 0:
        raise ValueError("min_delay no puede ser negativo")
    if max_delay < min_delay:
        raise ValueError("max_delay debe ser mayor o igual a min_delay")
    return get_eda_late_heatmap(scope=scope, min_delay=min_delay, max_delay=max_delay)


@mcp.tool()
def eda_desfase_distribution(scope: str) -> dict:
    """Distribución del desfase entre servicio y facturación."""
    return get_eda_desfase_distribution(scope=scope)