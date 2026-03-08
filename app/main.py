from fastapi import FastAPI
from app.api.bronze import router as bronze_router
from app.api.silver import router as silver_router
from app.api.gold_forecast import router as gold_forecast_router
from app.api.eda import router as eda_router
from app.api.forecast_query import router as forecast_query_router

from app.mcp.server import mcp

import app.mcp.tools_forecast
import app.mcp.tools_eda
import app.mcp.prompts
import app.mcp.resources

mcp_app = mcp.http_app(path="/")

app = FastAPI(
    title="anomalias-backend",
    lifespan=mcp_app.lifespan,
)

app.include_router(bronze_router)
app.include_router(silver_router)
app.include_router(gold_forecast_router)
app.include_router(eda_router)
app.include_router(forecast_query_router)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/debug-routes")
def debug_routes():
    return {"routes": [str(r.path) for r in app.routes]}

app.mount("/mcp", mcp_app)