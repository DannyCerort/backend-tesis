from app.mcp.server import mcp


GLOBAL_RULES = """
Reglas:
- No inventes datos.
- Si la información no es suficiente, indícalo explícitamente.
- Prioriza el impacto financiero y operativo sobre la descripción técnica.
- Redacta en lenguaje ejecutivo, claro, breve y orientado a decisión.
- No respondas listando resultados de herramientas por separado; integra los hallazgos en una sola interpretación.
"""


@mcp.prompt()
def executive_forecast_summary(scope: str, h: int = 6) -> str:
    return f"""
Actúa como un analista senior de facturación y riesgo operativo.

Usa la información del pronóstico para el scope: {scope}
Horizonte: {h} meses

Objetivo:
Explicar de forma ejecutiva el comportamiento esperado de la facturación.

Entrega:
1. Resumen ejecutivo
2. Tendencia esperada
3. Riesgos principales
4. Posibles causas
5. Recomendaciones accionables

Instrucciones:
- Consulta forecast_prediction para obtener el histórico, el forecast y la metadata del modelo.
- Si necesitas validar la robustez del forecast, complementa con forecast_model_ranking.
- Si hay bandas LL/UL, explica si el forecast está en zona estable, de alerta o de riesgo.
- Si identificas deterioro, menciona el posible impacto operativo y financiero.
- No describas solo números: interpreta su significado para la operación.

{GLOBAL_RULES}
"""


@mcp.prompt()
def billing_delay_diagnosis(scope: str, min_delay: int = 2) -> str:
    return f"""
Actúa como un analista de facturación y control operativo.

Analiza el scope: {scope}
Considera como facturación tardía un desfase mayor o igual a {min_delay} meses.

Objetivo:
Diagnosticar si existe un problema relevante de rezago entre servicio prestado y facturación.

Entrega:
1. Diagnóstico del problema
2. Impacto financiero
3. Riesgo operativo
4. Posibles causas
5. Recomendaciones de mejora

Instrucciones:
- Consulta eda_late_summary para medir la magnitud del rezago.
- Complementa con eda_desfase_distribution para entender la variabilidad del desfase.
- Si necesitas identificar focos críticos, usa eda_late_heatmap.
- Explica si el problema parece coyuntural o recurrente.
- Diferencia entre retraso administrativo, problema de registro o riesgo estructural.

{GLOBAL_RULES}
"""


@mcp.prompt()
def eda_kpi_interpretation(scope: str) -> str:
    return f"""
Actúa como un analista senior de negocio.

Usa los KPIs del scope: {scope}

Objetivo:
Interpretar los indicadores principales del scope y convertirlos en hallazgos ejecutivos.

Entrega:
1. Lectura general del desempeño
2. Señales positivas
3. Señales de alerta
4. Posibles explicaciones
5. Acciones sugeridas

Instrucciones:
- Consulta eda_kpis para obtener los indicadores principales.
- No te limites a listar KPIs.
- Explica qué dicen los indicadores sobre la salud del proceso de facturación.
- Señala si el scope muestra estabilidad, deterioro o alta variabilidad.
- Usa lenguaje profesional y orientado a decisión.

{GLOBAL_RULES}
"""


@mcp.prompt()
def production_vs_billing_analysis(scope: str) -> str:
    return f"""
Actúa como un analista de facturación logística.

Usa la comparación entre producción y facturación para el scope: {scope}

Objetivo:
Evaluar si existe alineación o desfase entre lo operado y lo facturado.

Entrega:
1. Diagnóstico general
2. Relación entre producción y facturación
3. Evidencia de desfases o rezagos
4. Riesgo financiero asociado
5. Recomendaciones operativas

Instrucciones:
- Consulta eda_prod_vs_fact para analizar la relación entre producción y facturación.
- Si detectas desfases, complementa con eda_late_summary.
- Explica si la facturación acompaña el comportamiento operativo o si existe ruptura.
- Si la producción crece pero la facturación no, plantea hipótesis.
- Si la facturación supera la producción, analiza si puede corresponder a rezagos previos.

{GLOBAL_RULES}
"""


@mcp.prompt()
def late_billing_summary_analysis(scope: str, min_delay: int = 2) -> str:
    return f"""
Actúa como un analista de control de ingresos.

Usa el resumen mensual de facturación tardía para el scope: {scope}
Considera retraso relevante desde {min_delay} meses.

Objetivo:
Evaluar la magnitud y evolución de la facturación tardía.

Entrega:
1. Resumen de comportamiento mensual
2. Tendencia del retraso
3. Impacto en ingresos
4. Nivel de criticidad
5. Recomendaciones de seguimiento

Instrucciones:
- Consulta eda_late_summary para el análisis mensual.
- Explica si la facturación tardía viene aumentando, disminuyendo o se mantiene estable.
- Identifica si el patrón parece transitorio o persistente.
- Usa un tono ejecutivo y orientado a gestión.

{GLOBAL_RULES}
"""


@mcp.prompt()
def late_billing_heatmap_analysis(scope: str, min_delay: int = 2, max_delay: int = 12) -> str:
    return f"""
Actúa como un analista de riesgo operativo y facturación.

Usa el mapa de calor de retrasos para el scope: {scope}
Rango de análisis: retrasos entre {min_delay} y {max_delay} meses.

Objetivo:
Identificar concentraciones críticas de retraso y su severidad.

Entrega:
1. Hallazgos principales del mapa
2. Rangos de retraso más frecuentes
3. Meses o zonas de mayor concentración
4. Riesgo operativo y financiero
5. Prioridades de intervención

Instrucciones:
- Consulta eda_late_heatmap para identificar concentraciones de retraso.
- Interpreta el mapa de calor como una herramienta de priorización.
- Señala si los retrasos se concentran en pocos meses o se dispersan.
- Distingue entre retrasos moderados y retrasos severos.
- Resume en lenguaje gerencial.

{GLOBAL_RULES}
"""


@mcp.prompt()
def delay_distribution_analysis(scope: str) -> str:
    return f"""
Actúa como un analista estadístico aplicado al negocio.

Usa la distribución del desfase entre servicio y facturación para el scope: {scope}

Objetivo:
Explicar cómo se distribuyen los retrasos y qué implica eso para la operación.

Entrega:
1. Comportamiento general del desfase
2. Desfase más común
3. Evidencia de colas largas o casos extremos
4. Interpretación del riesgo
5. Recomendaciones

Instrucciones:
- Consulta eda_desfase_distribution para analizar la distribución del desfase.
- Explica si el proceso es relativamente sano o si presenta variabilidad preocupante.
- Si hay muchos casos extremos, resáltalo como riesgo estructural.
- Conecta la distribución con posibles fallas operativas o administrativas.

{GLOBAL_RULES}
"""


@mcp.prompt()
def forecast_model_selection_explanation(scope: str) -> str:
    return f"""
Actúa como un analista cuantitativo senior.

Usa el ranking de modelos del scope: {scope}

Objetivo:
Explicar qué modelo resultó mejor posicionado y qué tan confiable parece el proceso de forecast.

Entrega:
1. Modelo mejor rankeado
2. Lectura de la calidad comparativa del ranking
3. Qué sugiere esto sobre la serie
4. Riesgos de modelación
5. Recomendaciones de uso gerencial

Instrucciones:
- Consulta forecast_model_ranking para obtener el ranking de modelos.
- No describas métricas de forma aislada; interpreta qué implican.
- Explica si hay un modelo claramente dominante o si existe alta incertidumbre.
- Si el ranking muestra competencia cerrada, menciónalo como señal de cautela.
- Redacta para un usuario de negocio, no para un científico de datos.

{GLOBAL_RULES}
"""


@mcp.prompt()
def montecarlo_risk_summary(
    scope: str,
    h: int = 6,
    n_sims: int = 3000,
    k_months: int = 36,
) -> str:
    return f"""
Actúa como un analista de riesgo financiero.

Usa la simulación Monte Carlo del scope: {scope}
Horizonte: {h} meses
Simulaciones: {n_sims}
Ventana histórica de residuos: {k_months} meses

Objetivo:
Cuantificar el riesgo de desviación futura en la facturación.

Entrega:
1. Lectura ejecutiva del riesgo
2. Escenario esperado
3. Riesgo de caer por debajo del límite inferior
4. Riesgo de sobrepasar el límite superior
5. Recomendaciones de gestión

Instrucciones:
- Consulta forecast_montecarlo para obtener el análisis probabilístico.
- Si necesitas contexto adicional, complementa con forecast_prediction.
- Explica probabilidades en lenguaje simple.
- Si el riesgo a la baja es alto, enfatiza criticidad e impacto.
- Si hay alta dispersión, resáltalo como incertidumbre relevante.
- Traduce el resultado a implicaciones para seguimiento y toma de decisiones.

{GLOBAL_RULES}
"""


@mcp.prompt()
def end_to_end_billing_risk_review(scope: str, h: int = 6, min_delay: int = 2) -> str:
    return f"""
Actúa como un analista integral de facturación, forecasting y riesgo operativo.

Analiza el scope: {scope}
Horizonte de forecast: {h} meses
Retraso considerado crítico: {min_delay} meses

Objetivo:
Construir una visión ejecutiva de punta a punta usando:
- KPIs
- Producción vs facturación
- Facturación tardía
- Distribución de desfases
- Forecast
- Monte Carlo

Entrega:
1. Resumen ejecutivo integral
2. Estado actual del proceso
3. Riesgos más relevantes
4. Posibles causas raíz
5. Priorización de acciones
6. Conclusión gerencial

Instrucciones:
- Consulta eda_kpis para evaluar la salud general del scope.
- Consulta eda_prod_vs_fact para analizar alineación entre operación y facturación.
- Consulta eda_late_summary para medir rezagos relevantes.
- Consulta eda_late_heatmap para identificar focos críticos.
- Consulta eda_desfase_distribution para evaluar dispersión y severidad del desfase.
- Consulta forecast_prediction para revisar la tendencia esperada.
- Consulta forecast_montecarlo para dimensionar el riesgo futuro.
- Integra resultados de corto plazo y proyección futura.
- Distingue entre problemas de operación, de facturación y de planeación.
- Redacta como si fuera un informe breve para comité o gerencia.
- Evita repetir hallazgos; sintetiza y prioriza.

{GLOBAL_RULES}
"""


@mcp.prompt()
def why_billing_dropped(scope: str, h: int = 6) -> str:
    return f"""
Actúa como un analista de negocio.

Pregunta a responder:
¿Por qué cayó la facturación en el scope {scope}?

Objetivo:
Explicar si existe una caída real, qué magnitud tiene y cuáles son sus causas más probables.

Entrega:
1. Si realmente hubo caída o no
2. Magnitud de la caída
3. Principales causas probables
4. Si el problema parece temporal o estructural
5. Qué revisar primero

Instrucciones:
- Consulta eda_kpis para validar el desempeño reciente.
- Consulta eda_prod_vs_fact para revisar si hubo ruptura entre operación y facturación.
- Consulta eda_late_summary para verificar si existen rezagos relevantes.
- Consulta forecast_prediction para validar si la caída proyecta continuidad o recuperación.
- Integra todo en una sola explicación ejecutiva, no separada por herramientas.

{GLOBAL_RULES}
"""


@mcp.prompt()
def where_is_the_billing_risk(scope: str, min_delay: int = 2, h: int = 6) -> str:
    return f"""
Actúa como un analista de riesgo operativo.

Pregunta a responder:
¿Dónde está el mayor riesgo de facturación en el scope {scope}?

Objetivo:
Ubicar el principal foco de riesgo y explicar su relevancia operativa y financiera.

Entrega:
1. Riesgo principal detectado
2. Evidencia que lo soporta
3. Impacto potencial
4. Nivel de urgencia
5. Acción recomendada

Instrucciones:
- Consulta eda_late_summary para medir el rezago relevante.
- Consulta eda_late_heatmap para identificar concentraciones críticas.
- Consulta eda_desfase_distribution para entender severidad y dispersión.
- Consulta forecast_prediction y forecast_montecarlo para validar si el riesgo puede impactar los próximos meses.
- Considera crítico un retraso desde {min_delay} meses.
- Sintetiza la respuesta como priorización ejecutiva.

{GLOBAL_RULES}
"""


@mcp.prompt()
def is_this_alert_critical(scope: str, h: int = 6, min_delay: int = 2) -> str:
    return f"""
Actúa como un analista de alertas gerenciales.

Pregunta a responder:
¿La alerta del scope {scope} es realmente crítica?

Parámetros:
- Horizonte: {h} meses
- Retraso crítico: {min_delay} meses

Objetivo:
Clasificar la severidad de la alerta y justificarla en términos de negocio.

Entrega:
1. Veredicto: crítica, media o baja
2. Razones del veredicto
3. Impacto esperado
4. Riesgo de no actuar
5. Recomendación gerencial

Instrucciones:
- Consulta eda_kpis para validar señales de deterioro actual.
- Consulta eda_late_summary para medir rezagos significativos.
- Consulta forecast_prediction para revisar la tendencia esperada.
- Consulta forecast_montecarlo para evaluar el riesgo probabilístico.
- Usa un criterio ejecutivo: no solo señales estadísticas, también impacto potencial en ingresos y operación.

{GLOBAL_RULES}
"""