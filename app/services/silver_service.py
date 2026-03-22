import io
import re
import unicodedata
from datetime import datetime

import numpy as np
import pandas as pd

from app.core.config import settings
from app.services.s3_service import get_s3_client, object_exists

from rapidfuzz import fuzz, process

def _norm(s):
    if pd.isna(s):
        return ""
    s = str(s).upper().strip()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"\s+", " ", s).strip()
    return s


def classify_service(obs):
    t = _norm(obs)
    if not t:
        return ("SIN OBS", "SIN OBS")

    if re.match(r"^(DC|NP|NC|ND)-", t) or "ANULA FACTURA" in t or "NOTA CREDITO" in t or "NOTA DEBITO" in t or "SE APLICA NC" in t:
        if "DEVOLUC" in t:
            return ("AJUSTES", "DEVOLUCION / NOVEDAD")
        if "RADIC" in t:
            return ("AJUSTES", "RADICACION / ERROR")
        if "GUIA" in t or "NOVEDAD" in t:
            return ("AJUSTES", "NOVEDAD GUIA / OPERATIVA")
        return ("AJUSTES", "OTRO AJUSTE")

    if "CARGA LIVIANA" in t:
        return ("CARGA", "CARGA LIVIANA")

    if "ALMACEN" in t or "ALMACENAJE" in t or "BODEGAJE" in t:
        if "MENSAJERO AUXILIAR" in t or "MENSAJERO" in t:
            return ("ALMACENAMIENTO", "MENSAJERO AUXILIAR ALMACEN")
        if "BODEGAJE" in t:
            return ("ALMACENAMIENTO", "BODEGAJE")
        return ("ALMACENAMIENTO", "ALMACENAJE")

    if "PAQUETEO" in t or "PAQUETERIA" in t:
        if "DEDICADO" in t:
            return ("MENSAJERIA", "PAQUETEO DEDICADO")
        if "TRANSPORTE" in t:
            return ("TRANSPORTE", "PAQUETERIA")
        return ("MENSAJERIA", "PAQUETEO / PAQUETERIA")

    if ("MENSAJERIA" in t) and ("TRANSPORTE" in t):
        return ("MIXTO", "MENSAJERIA + TRANSPORTE")

    if "MENSAJERIA" in t:
        if "FIJA" in t:
            return ("MENSAJERIA", "FIJA")
        if "NACIONAL" in t:
            return ("MENSAJERIA", "NACIONAL")
        if "EXPRESA" in t:
            return ("MENSAJERIA", "EXPRESA")
        return ("MENSAJERIA", "OTRA")

    if "TRANSPORTE" in t or "TERRESTRE" in t:
        if "NACIONAL" in t:
            return ("TRANSPORTE", "NACIONAL")
        if "TERRESTRE" in t:
            return ("TRANSPORTE", "TERRESTRE")
        return ("TRANSPORTE", "OTRO")

    return ("OTROS", "REVISAR")


meses_full = {
    "ENERO": 1, "FEBRERO": 2, "MARZO": 3, "ABRIL": 4, "MAYO": 5, "JUNIO": 6,
    "JULIO": 7, "AGOSTO": 8, "SEPTIEMBRE": 9, "OCTUBRE": 10,
    "NOVIEMBRE": 11, "DICIEMBRE": 12,
}
meses_abrev = {
    "ENE": 1, "FEB": 2, "MAR": 3, "ABR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AGO": 8, "SEP": 9, "SET": 9,
    "OCT": 10, "NOV": 11, "DIC": 12,
}
MES_KEYS = list(meses_full.keys()) + list(meses_abrev.keys())

patron_anio = r"\b(20\d{2})\b"

patron_rango_full = (
    r"\b(ENERO|FEBRERO|MARZO|ABRIL|MAYO|JUNIO|JULIO|AGOSTO|SEPTIEMBRE|OCTUBRE|NOVIEMBRE|DICIEMBRE)"
    r"\s*[-/]\s*"
    r"(ENERO|FEBRERO|MARZO|ABRIL|MAYO|JUNIO|JULIO|AGOSTO|SEPTIEMBRE|OCTUBRE|NOVIEMBRE|DICIEMBRE)"
    r"\s*(20\d{2})\b"
)

patron_rango_abrev = (
    r"\b(ENE|FEB|MAR|ABR|MAY|JUN|JUL|AGO|SEP|SET|OCT|NOV|DIC)"
    r"\s*[-/]\s*"
    r"(ENE|FEB|MAR|ABR|MAY|JUN|JUL|AGO|SEP|SET|OCT|NOV|DIC)"
    r"\s*(20\d{2})\b"
)

patron_mes_full = r"\b(" + "|".join(meses_full.keys()) + r")\b"
patron_mes_abrev = r"\b(" + "|".join(meses_abrev.keys()) + r")\b"


def extraer_mes_anio(obs, threshold_fuzzy=85):
    if pd.isna(obs):
        return pd.Series([np.nan, np.nan])

    texto = str(obs).upper()

    # Guard: solo si hay contexto real de servicio/mes
    if not any(
        k in texto
        for k in [
            " MES ",
            "DEL MES",
            "DURANTE",
            "PRESTADO",
            "PRESTADOS",
            "SERVICIO",
            "PERIODO",
            "CORRESPONDIENTE",
        ]
    ):
        return pd.Series([np.nan, np.nan])

    # Rango FULL: DICIEMBRE-ENERO 2024 -> toma mes final + año
    m = re.search(patron_rango_full, texto)
    if m:
        _, mes_fin, anio = m.groups()
        return pd.Series([meses_full[mes_fin], int(anio)])

    # Rango ABREV: DIC-ENE 2024
    m = re.search(patron_rango_abrev, texto)
    if m:
        _, mes_fin, anio = m.groups()
        return pd.Series([meses_abrev[mes_fin], int(anio)])

    # Mes FULL exacto
    m = re.search(patron_mes_full, texto)
    if m:
        mes = meses_full[m.group(1)]
        anios = re.findall(patron_anio, texto)
        anio = int(anios[-1]) if anios else np.nan
        return pd.Series([mes, anio])

    # Mes ABREV exacto
    m = re.search(patron_mes_abrev, texto)
    if m:
        mes = meses_abrev[m.group(1)]
        anios = re.findall(patron_anio, texto)
        anio = int(anios[-1]) if anios else np.nan
        return pd.Series([mes, anio])

    # Fuzzy match como último recurso
    tokens = re.findall(r"[A-ZÁÉÍÓÚÑ]{4,}", texto)
    mejor_score = 0
    mes_num = np.nan

    for tok in tokens:
        match = process.extractOne(tok, MES_KEYS, scorer=fuzz.partial_ratio)
        if match:
            key, score, _ = match
            if score >= threshold_fuzzy and score > mejor_score:
                mejor_score = score
                mes_num = meses_full.get(key, meses_abrev.get(key))

    anios = re.findall(patron_anio, texto)
    anio = int(anios[-1]) if anios else np.nan

    return pd.Series([mes_num, anio])


def inferir_anio_servicio(row):
    if not pd.isna(row["ANIO_SERVICIO"]):
        return int(row["ANIO_SERVICIO"])

    if int(row["MES"]) == 1:
        return int(row["AÑO DOCUMENTO"]) - 1

    return int(row["AÑO DOCUMENTO"])


def build_silver() -> dict:
    bucket = settings.s3_bucket
    bronze_key = settings.bronze_key
    silver_key = settings.silver_key

    if not object_exists(bucket, bronze_key):
        raise FileNotFoundError(f"No existe bronze en s3://{bucket}/{bronze_key}")

    s3 = get_s3_client()
    response = s3.get_object(Bucket=bucket, Key=bronze_key)
    parquet_bytes = response["Body"].read()

    df = pd.read_parquet(io.BytesIO(parquet_bytes), engine="pyarrow")
    total_inicial = len(df)

    # -------------------------------------------------------------------
    # 🔁 Normalizar columnas numéricas con coma decimal
    # -------------------------------------------------------------------
    numeric_cols = [
        "VAL TOTAL",
        "VALOR NETO",
        "VALOR CON IVA",
        "IVA",
        "RETENCION  ICA",
        "RETEFUENTE",
        "% ICA",
        "por_iva",
        "%RETENCION",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(".", "", regex=False)   # quitar separadores de miles
                .str.replace(",", ".", regex=False)  # convertir coma en punto decimal
                .str.replace(" ", "", regex=False)
                .str.replace("$", "", regex=False)
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # -------------------------------------------------------------------
    # Continuar con la lógica existente (filtrado de 2025-10, clasificación, etc.)
    # -------------------------------------------------------------------
    cut_year = 2025
    cut_month = 10

    df["AÑO DOCUMENTO"] = pd.to_numeric(df["AÑO DOCUMENTO"], errors="coerce")
    df["MES"] = pd.to_numeric(df["MES"], errors="coerce")

    mask_oct_2025 = (df["AÑO DOCUMENTO"] == cut_year) & (df["MES"] == cut_month)
    df = df[~mask_oct_2025].copy()

    df[["FAMILIA_SERVICIO", "TIPO_SERVICIO"]] = (
        df["OBSERVACION"].apply(classify_service).apply(pd.Series)
    )

    required_cols = ["AÑO DOCUMENTO", "MES", "OBSERVACION"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas para ETL: {missing}")

    df[["MES_SERVICIO", "ANIO_SERVICIO"]] = df["OBSERVACION"].apply(extraer_mes_anio)
    mask_etl_ok = df["MES_SERVICIO"].notna()
    df.loc[mask_etl_ok, "ANIO_SERVICIO"] = df.loc[mask_etl_ok].apply(inferir_anio_servicio, axis=1)

    df_etl = df[mask_etl_ok].copy()

    df_etl["fecha_servicio"] = pd.to_datetime(
        dict(year=df_etl["ANIO_SERVICIO"], month=df_etl["MES_SERVICIO"], day=1),
        errors="coerce"
    ).dt.to_period("M").astype(str)

    df_etl["fecha_facturacion"] = pd.to_datetime(
        dict(year=df_etl["AÑO DOCUMENTO"], month=df_etl["MES"], day=1),
        errors="coerce"
    ).dt.to_period("M").astype(str)

    df_etl["diff_factura_servicio"] = (
        pd.to_datetime(df_etl["fecha_facturacion"]).dt.to_period("M").astype(int)
        - pd.to_datetime(df_etl["fecha_servicio"]).dt.to_period("M").astype(int)
    )

    mask_diff_ok = (
        (df_etl["diff_factura_servicio"] >= 0) &
        (df_etl["diff_factura_servicio"] < 12)
    )

    df_final = df_etl[mask_diff_ok].copy()
    total_val = df_final["VAL TOTAL"].sum()
    print(f"Suma de VAL TOTAL: {total_val:,.2f}")

    parquet_buffer = io.BytesIO()
    df_final.to_parquet(
        parquet_buffer,
        engine="pyarrow",
        compression="snappy",
        index=False,
    )
    parquet_buffer.seek(0)

    s3.put_object(
        Bucket=bucket,
        Key=silver_key,
        Body=parquet_buffer.getvalue(),
        ContentType="application/octet-stream",
    )

    excluidos_etl = total_inicial - len(df_etl)
    excluidos_diff = len(df_etl) - len(df_final)
    excluidos_total = total_inicial - len(df_final)

    return {
        "status": "success",
        "bucket": bucket,
        "bronze_key": bronze_key,
        "silver_key": silver_key,
        "rows_initial": int(total_inicial),
        "rows_final": int(len(df_final)),
        "excluidos_etl": int(excluidos_etl),
        "excluidos_diff": int(excluidos_diff),
        "excluidos_total": int(excluidos_total),
        "columns_final": int(df_final.shape[1]),
        "executed_at": datetime.utcnow().isoformat(),
        "message": "Capa silver creada correctamente.",
    }


def get_silver_preview(limit: int = 10) -> dict:
    bucket = settings.s3_bucket
    silver_key = settings.silver_key

    if not object_exists(bucket, silver_key):
        raise FileNotFoundError(f"No existe silver en s3://{bucket}/{silver_key}")

    s3 = get_s3_client()
    response = s3.get_object(Bucket=bucket, Key=silver_key)
    parquet_bytes = response["Body"].read()

    df = pd.read_parquet(io.BytesIO(parquet_bytes), engine="pyarrow")

    return {
        "status": "success",
        "rows_total": int(df.shape[0]),
        "columns_total": int(df.shape[1]),
        "preview_rows": df.head(limit).to_dict(orient="records"),
    }