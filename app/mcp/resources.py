from app.mcp.server import mcp
from app.services.forecast_query_service import get_forecast_options


@mcp.resource("config://forecast/scopes")
def forecast_scopes() -> str:
    data = get_forecast_options()
    return "\n".join(data.get("options", []))


@mcp.resource("info://system/methodology")
def methodology() -> str:
    return """
Sistema analítico de consulta sobre resultados de facturación.
Incluye:
- análisis exploratorio (EDA)
- series jerárquicas
- ranking de modelos
- pronósticos con bandas LL/UL
- simulación Monte Carlo
Fuentes:
- capa silver
- capa gold series
- capa gold forecast
- capa gold ranking
"""