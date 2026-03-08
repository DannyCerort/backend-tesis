from app.mcp.server import mcp


@mcp.prompt()
def executive_forecast_summary(scope: str, h: int = 6) -> str:
    return f"""
Actúa como un analista senior de facturación y riesgo operativo.

Analiza el scope: {scope}
Horizonte: {h} meses

Entrega:
1. Resumen ejecutivo
2. Tendencia esperada
3. Riesgos principales
4. Posibles causas
5. Recomendaciones accionables
"""


@mcp.prompt()
def billing_delay_diagnosis(scope: str, min_delay: int = 2) -> str:
    return f"""
Analiza el scope: {scope}
Considera como facturación tardía un desfase mayor o igual a {min_delay} meses.

Entrega:
1. Diagnóstico del problema
2. Impacto financiero
3. Riesgo operativo
4. Recomendaciones de mejora
"""