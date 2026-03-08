import io
from typing import Tuple

import numpy as np
import pandas as pd

from app.core.config import settings
from app.services.s3_service import get_s3_client, object_exists

COL_CLIENTE = "ID_NOMBRE_CLIENTE"
COL_FAM = "FAMILIA_SERVICIO"
COL_VAL = "VAL TOTAL"


def _read_silver() -> pd.DataFrame:
    bucket = settings.s3_bucket
    key = settings.silver_key

    if not object_exists(bucket, key):
        raise FileNotFoundError(f"No existe silver en s3://{bucket}/{key}")

    s3 = get_s3_client()
    response = s3.get_object(Bucket=bucket, Key=key)
    parquet_bytes = response["Body"].read()
    return pd.read_parquet(io.BytesIO(parquet_bytes), engine="pyarrow")


def _ensure_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df[COL_CLIENTE] = df[COL_CLIENTE].fillna("SIN CLIENTE").astype(str)
    df[COL_FAM] = df[COL_FAM].fillna("SIN SERVICIO").astype(str)
    df[COL_VAL] = pd.to_numeric(df[COL_VAL], errors="coerce").fillna(0.0)

    df["fecha_servicio"] = pd.to_datetime(df["fecha_servicio"], errors="coerce")
    df["fecha_facturacion"] = pd.to_datetime(df["fecha_facturacion"], errors="coerce")
    df["diff_factura_servicio"] = pd.to_numeric(df["diff_factura_servicio"], errors="coerce")

    df["mes_serv"] = df["fecha_servicio"].dt.to_period("M").dt.to_timestamp()
    df["mes_fact"] = df["fecha_facturacion"].dt.to_period("M").dt.to_timestamp()

    return df


def _build_selector_options(df: pd.DataFrame):
    opt_empresa = ["EMPRESA | TOTAL"]

    familias = sorted(df[COL_FAM].dropna().unique().tolist())
    opt_emp_serv = [f"EMPRESA→SERVICIO | {f}" for f in familias]

    pairs = (
        df[[COL_FAM, COL_CLIENTE]]
        .drop_duplicates()
        .sort_values([COL_FAM, COL_CLIENTE])
    )
    opt_serv_cli = [f"SERVICIO→CLIENTE | {r[0]} | {r[1]}" for r in pairs.values]

    return opt_empresa + opt_emp_serv + opt_serv_cli


def _filter_scope(df: pd.DataFrame, sel: str) -> Tuple[pd.DataFrame, str]:
    if sel.startswith("EMPRESA |"):
        return df.copy(), "Empresa (Total)"

    if sel.startswith("EMPRESA→SERVICIO |"):
        fam_sel = sel.split("|", 1)[1].strip()
        out = df[df[COL_FAM] == fam_sel].copy()
        return out, f"Empresa → Servicio: {fam_sel}"

    parts = [p.strip() for p in sel.split("|")]
    fam_sel = parts[1]
    cli_sel = parts[2]
    out = df[(df[COL_FAM] == fam_sel) & (df[COL_CLIENTE] == cli_sel)].copy()
    return out, f"Servicio → Cliente: {fam_sel} / {cli_sel}"


def get_eda_options():
    df = _ensure_types(_read_silver())
    return {"options": _build_selector_options(df)}


def get_eda_kpis(scope: str):
    df = _ensure_types(_read_silver())
    df_scope, titulo = _filter_scope(df, scope)

    if df_scope.empty:
        return {"title": titulo, "empty": True}

    total_val = df_scope[COL_VAL].sum(skipna=True)
    total_val_mm = total_val / 1_000_000

    df_mes3 = df_scope[df_scope["diff_factura_servicio"] >= 2]
    val_mes3 = df_mes3[COL_VAL].sum(skipna=True)
    val_mes3_mm = val_mes3 / 1_000_000
    pct_mes3 = (val_mes3 / total_val * 100) if total_val else 0

    prod_total = df_scope.dropna(subset=["fecha_servicio"])[COL_VAL].sum()
    fact_total = df_scope.dropna(subset=["fecha_facturacion"])[COL_VAL].sum()
    gap_final_mm = (prod_total - fact_total) / 1_000_000

    ops = len(df_scope)

    return {
        "title": titulo,
        "empty": False,
        "total_val_mm": float(total_val_mm),
        "val_mes3_mm": float(val_mes3_mm),
        "pct_mes3": float(pct_mes3),
        "ops": int(ops),
        "gap_final_mm": float(gap_final_mm),
    }


def get_eda_desfase_distribution(scope: str):
    df = _ensure_types(_read_silver())
    df_scope, titulo = _filter_scope(df, scope)

    df_plot = (
        df_scope
        .dropna(subset=["diff_factura_servicio", COL_VAL])
        .groupby("diff_factura_servicio", as_index=False)
        .agg(
            cantidad=("diff_factura_servicio", "count"),
            valor_total=(COL_VAL, "sum"),
        )
        .sort_values("diff_factura_servicio")
    )

    if df_plot.empty:
        return {"title": titulo, "data": []}

    df_plot["valor_total_mm"] = df_plot["valor_total"] / 1_000_000

    return {
        "title": titulo,
        "data": df_plot.to_dict(orient="records"),
    }


def _build_hover_facturacion(df_scope: pd.DataFrame, top_n: int = 12) -> dict:
    mix = (
        df_scope.dropna(subset=["mes_fact", "mes_serv", COL_VAL])
        .groupby(["mes_fact", "mes_serv"], as_index=False)[COL_VAL]
        .sum()
        .rename(columns={COL_VAL: "valor"})
    )

    if mix.empty:
        return {}

    def build_hover_text(sub):
        sub = sub.sort_values("mes_serv")
        total = sub["valor"].sum()
        lines = [f"{str(r['mes_serv'].date())}: ${r['valor']:,.0f}" for _, r in sub.iterrows()]
        top = "<br>".join(lines[:top_n])
        extra = f"<br>... ({len(lines)-top_n} más)" if len(lines) > top_n else ""
        return f"<b>Factura compuesta por:</b><br>{top}{extra}<br><b>Total:</b> ${total:,.0f}"

    return mix.groupby("mes_fact").apply(build_hover_text).to_dict()


def get_eda_prod_vs_fact(scope: str):
    df = _ensure_types(_read_silver())
    df_scope, titulo = _filter_scope(df, scope)

    prod = (
        df_scope.dropna(subset=["mes_serv", COL_VAL])
        .groupby("mes_serv", as_index=False)[COL_VAL]
        .sum()
        .rename(columns={"mes_serv": "fecha", COL_VAL: "valor"})
    )

    fact = (
        df_scope.dropna(subset=["mes_fact", COL_VAL])
        .groupby("mes_fact", as_index=False)[COL_VAL]
        .sum()
        .rename(columns={"mes_fact": "fecha", COL_VAL: "valor"})
    )

    hover_map = _build_hover_facturacion(df_scope, top_n=12)
    fact["hover"] = fact["fecha"].map(hover_map).fillna(
        "<b>Factura compuesta por:</b><br>Sin desglose disponible"
    )

    prod["fecha"] = pd.to_datetime(prod["fecha"]).dt.strftime("%Y-%m-%d")
    fact["fecha"] = pd.to_datetime(fact["fecha"]).dt.strftime("%Y-%m-%d")

    return {
        "title": titulo,
        "produccion": prod.to_dict(orient="records"),
        "facturacion": fact.to_dict(orient="records"),
    }


def get_eda_late_summary(scope: str, min_delay: int = 2):
    df = _ensure_types(_read_silver())
    df_scope, titulo = _filter_scope(df, scope)

    total_mes = (
        df_scope.dropna(subset=["mes_fact", COL_VAL])
        .groupby("mes_fact", as_index=False)[COL_VAL]
        .sum()
        .rename(columns={COL_VAL: "total_facturado"})
    )

    late_mes = (
        df_scope.dropna(subset=["mes_fact", COL_VAL, "diff_factura_servicio"])
        .query("diff_factura_servicio >= @min_delay")
        .groupby("mes_fact", as_index=False)[COL_VAL]
        .sum()
        .rename(columns={COL_VAL: "facturado_tarde"})
    )

    df_late = total_mes.merge(late_mes, on="mes_fact", how="left").fillna(0)
    df_late["pct_tarde"] = (
        df_late["facturado_tarde"] / df_late["total_facturado"] * 100
    ).where(df_late["total_facturado"] > 0, 0)

    df_late["mes_fact"] = pd.to_datetime(df_late["mes_fact"]).dt.strftime("%Y-%m-%d")

    return {
        "title": titulo,
        "data": df_late.to_dict(orient="records"),
    }


def get_eda_late_heatmap(scope: str, min_delay: int = 2, max_delay: int = 12):
    df = _ensure_types(_read_silver())
    df_scope, titulo = _filter_scope(df, scope)

    df_delay = (
        df_scope.dropna(subset=["mes_fact", COL_VAL, "diff_factura_servicio"])
        .query("diff_factura_servicio >= @min_delay and diff_factura_servicio <= @max_delay")
        .groupby(["mes_fact", "diff_factura_servicio"], as_index=False)[COL_VAL]
        .sum()
        .rename(columns={"diff_factura_servicio": "meses_atraso", COL_VAL: "valor"})
    )

    if df_delay.empty:
        return {
            "title": titulo,
            "x_labels": [],
            "y_labels": [],
            "z": [],
            "text": [],
        }

    heat = (
        df_delay.pivot(index="mes_fact", columns="meses_atraso", values="valor")
        .fillna(0.0)
        .sort_index()
    )

    cols = list(range(min_delay, max_delay + 1))
    heat = heat.reindex(columns=cols, fill_value=0.0)

    z = heat.values.tolist()
    text = [
        [f"{v:,.0f}" if v > 0 else "" for v in row]
        for row in heat.values
    ]

    y_labels = [pd.to_datetime(i).strftime("%b %Y") for i in heat.index]
    x_labels = [str(c) for c in heat.columns]

    return {
        "title": titulo,
        "x_labels": x_labels,
        "y_labels": y_labels,
        "z": z,
        "text": text,
    }