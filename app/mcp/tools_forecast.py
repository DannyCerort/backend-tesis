from app.mcp.server import mcp
from app.services.forecast_query_service import (
    get_forecast_options,
    get_forecast_prediction,
    get_forecast_model_ranking,
    get_forecast_montecarlo,
)


@mcp.tool()
def forecast_options() -> dict:
    """Retorna las opciones de scope disponibles para análisis de pronóstico."""
    return get_forecast_options()


@mcp.tool()
def forecast_prediction(scope: str, h: int = 6) -> dict:
    """Retorna histórico, forecast y metadata del modelo para un scope."""
    if h < 1 or h > 24:
        raise ValueError("h debe estar entre 1 y 24")
    return get_forecast_prediction(scope=scope, h=h)


@mcp.tool()
def forecast_model_ranking(scope: str) -> dict:
    """Retorna el ranking de modelos para un scope."""
    return get_forecast_model_ranking(scope=scope)


@mcp.tool()
def forecast_montecarlo(
    scope: str,
    h: int = 6,
    n_sims: int = 3000,
    seed: int = 42,
    k_months: int = 36,
) -> dict:
    """Retorna el análisis Monte Carlo del forecast para un scope."""
    if h < 1 or h > 24:
        raise ValueError("h debe estar entre 1 y 24")
    if n_sims < 100 or n_sims > 10000:
        raise ValueError("n_sims debe estar entre 100 y 10000")
    if k_months < 6 or k_months > 120:
        raise ValueError("k_months debe estar entre 6 y 120")

    return get_forecast_montecarlo(
        scope=scope,
        h=h,
        n_sims=n_sims,
        seed=seed,
        k_months=k_months,
    )