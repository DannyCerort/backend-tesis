import io
from datetime import datetime

import numpy as np
import pandas as pd

from app.core.config import settings
from app.services.s3_service import get_s3_client, object_exists
from app.services.job_store import update_job, complete_job, fail_job
from app.services.forecast_models import (
    evaluate_models,
    fit_best_and_forecast,
    residual_quantile_bands,
    forecast_moving_average,
    forecast_naive,
)

HORIZON = 6
MIN_MONTHS = 12
ALPHA = 0.05

DATE_COL = "fecha_facturacion"
VALUE_COL = "VAL TOTAL"
CLIENTE_COL = "ID_NOMBRE_CLIENTE"
FAM_COL = "FAMILIA_SERVICIO"


def read_parquet_from_s3(bucket: str, key: str) -> pd.DataFrame:
    s3 = get_s3_client()
    response = s3.get_object(Bucket=bucket, Key=key)
    parquet_bytes = response["Body"].read()
    return pd.read_parquet(io.BytesIO(parquet_bytes), engine="pyarrow")


def write_parquet_to_s3(df: pd.DataFrame, bucket: str, key: str):
    s3 = get_s3_client()
    buffer = io.BytesIO()
    df.to_parquet(buffer, engine="pyarrow", compression="snappy", index=False)
    buffer.seek(0)

    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=buffer.getvalue(),
        ContentType="application/octet-stream",
    )


def build_series_hier_df(df: pd.DataFrame) -> pd.DataFrame:
    tmp = df[[DATE_COL, VALUE_COL, CLIENTE_COL, FAM_COL]].copy()
    tmp[DATE_COL] = pd.to_datetime(tmp[DATE_COL], errors="coerce")
    tmp[VALUE_COL] = pd.to_numeric(tmp[VALUE_COL], errors="coerce")
    tmp[CLIENTE_COL] = tmp[CLIENTE_COL].astype(str)
    tmp[FAM_COL] = tmp[FAM_COL].fillna("SIN CLASIFICAR").astype(str)
    tmp = tmp.dropna(subset=[DATE_COL, VALUE_COL])

    tmp["mes"] = tmp[DATE_COL].values.astype("datetime64[M]")
    tmp["mes"] = pd.to_datetime(tmp["mes"])

    base = (
        tmp.groupby(["mes", FAM_COL, CLIENTE_COL], as_index=False)[VALUE_COL]
        .sum()
        .rename(columns={FAM_COL: "familia", CLIENTE_COL: "cliente", VALUE_COL: "y"})
    )
    base["nivel"] = "familia_cliente"

    fam = base.groupby(["mes", "familia"], as_index=False)["y"].sum()
    fam["nivel"] = "familia"
    fam["cliente"] = "ALL"

    emp = base.groupby(["mes"], as_index=False)["y"].sum()
    emp["nivel"] = "empresa"
    emp["familia"] = "ALL"
    emp["cliente"] = "ALL"

    out = pd.concat([base, fam, emp], ignore_index=True)

    keys = out[["nivel", "familia", "cliente"]].drop_duplicates()

    series = []
    for nivel, familia, cliente in keys.values:
        sub = out[
            (out["nivel"] == nivel) &
            (out["familia"] == familia) &
            (out["cliente"] == cliente)
        ].copy()

        sub = sub.sort_values("mes")
        idx = pd.date_range(sub["mes"].min(), sub["mes"].max(), freq="MS")
        sub = sub.set_index("mes").reindex(idx)
        sub.index.name = "mes"
        sub = sub.reset_index()

        sub["nivel"] = nivel
        sub["familia"] = familia
        sub["cliente"] = cliente
        sub["y"] = pd.to_numeric(sub["y"], errors="coerce").fillna(0.0)

        series.append(sub)

    out2 = pd.concat(series, ignore_index=True)
    out2 = out2.sort_values(["nivel", "familia", "cliente", "mes"]).reset_index(drop=True)
    return out2


def _choose_initial_train(n: int) -> int:
    if n >= 36:
        initial_train = 24
    elif n >= 24:
        initial_train = 18
    elif n >= 18:
        initial_train = 12
    else:
        initial_train = max(6, int(0.6 * n))

    return min(initial_train, max(6, n - 2))


def forecast_one_series(y: pd.Series):
    y = y.asfreq("MS").fillna(0.0).astype(float)

    seasonality_info = {"is_seasonal": False, "seasonal_period": None, "seasonal_strength": 0.0}

    zero_ratio = float((y == 0).mean()) if len(y) else 1.0
    if y.std() == 0 or zero_ratio >= 0.6:
        res = forecast_moving_average(y, h=HORIZON, window=3) if len(y) >= 3 else forecast_naive(y, h=HORIZON)
        return res.y_hat, 0.0, 0.0, res.model_name, np.nan, "FALLBACK", seasonality_info

    initial_train = _choose_initial_train(len(y))

    ranking, s_info = evaluate_models(
        y,
        initial_train=initial_train,
        h=1,
        ma_window=3,
        arima_order=(1, 1, 1),
        sarima_order=(1, 1, 1),
        seasonal_candidates=(12, 6),
        seasonal_threshold=0.30,
        seasonal_min_cycles=2,
        ml_lags=12,
        ml_add_calendar=True,
        ml_min_obs=18,
    )

    best_model = str(ranking.iloc[0]["model"])
    mse = float(ranking.iloc[0]["mse"])

    res = fit_best_and_forecast(
        y,
        ranking,
        seasonality_info=s_info,
        h=HORIZON,
        ma_window=3,
        arima_order=(1, 1, 1),
        sarima_order=(1, 1, 1),
        ml_lags=12,
        ml_add_calendar=True,
    )

    bands = residual_quantile_bands(
        y=y,
        model_name=best_model,
        seasonality_info=s_info,
        alpha=ALPHA,
        ma_window=3,
        arima_order=(1, 1, 1),
        sarima_order=(1, 1, 1),
        ml_lags=12,
        ml_add_calendar=True,
    )

    q_low, q_high = float(bands["q_low"]), float(bands["q_high"])

    seasonality_info = {
        "is_seasonal": bool(s_info.get("is_seasonal", False)),
        "seasonal_period": int(s_info["seasonal_period"]) if pd.notna(s_info.get("seasonal_period", None)) else None,
        "seasonal_strength": float(s_info.get("strength", 0.0)),
    }

    return res.y_hat, q_low, q_high, best_model, mse, "OK", seasonality_info


def build_forecast_hier_df(df_series: pd.DataFrame, job_id: str | None = None) -> pd.DataFrame:
    df_series = df_series.copy()
    df_series["mes"] = pd.to_datetime(df_series["mes"])
    df_series["y"] = pd.to_numeric(df_series["y"], errors="coerce").fillna(0.0)

    keys = (
        df_series[["nivel", "familia", "cliente"]]
        .drop_duplicates()
        .sort_values(["nivel", "familia", "cliente"])
    )

    out_rows = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total = len(keys)

    for i, (nivel, familia, cliente) in enumerate(keys.values, start=1):
        if job_id:
            pct = 40 + int((i / total) * 50)
            update_job(
                job_id,
                pct,
                "forecasting_series",
                f"Procesando serie {i} de {total}: {nivel} | {familia} | {cliente}"
            )

        sub = df_series[
            (df_series["nivel"] == nivel) &
            (df_series["familia"] == familia) &
            (df_series["cliente"] == cliente)
        ].copy().sort_values("mes")

        y = sub.set_index("mes")["y"].asfreq("MS").fillna(0.0).astype(float)
        n_obs = int(len(y))
        train_size = n_obs

        if n_obs < MIN_MONTHS:
            continue

        train_end = y.index.max()

        try:
            yhat, q_low, q_high, model, mse, status, seasonality_info = forecast_one_series(y)

            df_fore = yhat.rename("forecast").reset_index().rename(columns={"index": "mes"})
            df_fore["mes"] = pd.to_datetime(df_fore["mes"], errors="coerce")

            df_fore["LL"] = (df_fore["forecast"] + q_low).clip(lower=0)
            df_fore["UL"] = (df_fore["forecast"] + q_high).clip(lower=0)

            df_fore["nivel"] = nivel
            df_fore["familia"] = familia
            df_fore["cliente"] = cliente
            df_fore["model"] = model
            df_fore["mse"] = mse
            df_fore["train_end"] = train_end
            df_fore["status"] = status
            df_fore["generated_at"] = now

            df_fore["n_obs"] = n_obs
            df_fore["train_size"] = train_size

            df_fore["is_seasonal"] = bool(seasonality_info.get("is_seasonal", False))
            df_fore["seasonal_period"] = seasonality_info.get("seasonal_period", None)
            df_fore["seasonal_strength"] = float(seasonality_info.get("seasonal_strength", 0.0))

            out_rows.append(df_fore)

        except Exception as e:
            df_err = pd.DataFrame({
                "nivel": [nivel],
                "familia": [familia],
                "cliente": [cliente],
                "mes": [pd.NaT],
                "forecast": [np.nan],
                "LL": [np.nan],
                "UL": [np.nan],
                "model": ["ERROR"],
                "mse": [np.nan],
                "is_seasonal": [False],
                "seasonal_period": [None],
                "seasonal_strength": [0.0],
                "n_obs": [n_obs],
                "train_size": [train_size],
                "train_end": [train_end],
                "status": [f"ERROR: {type(e).__name__}"],
                "generated_at": [now],
            })
            out_rows.append(df_err)

    if not out_rows:
        return pd.DataFrame(columns=[
            "nivel", "familia", "cliente", "mes",
            "forecast", "LL", "UL",
            "model", "mse",
            "is_seasonal", "seasonal_period", "seasonal_strength",
            "n_obs", "train_size",
            "train_end", "status", "generated_at",
        ])

    out = pd.concat(out_rows, ignore_index=True)

    return out[[
        "nivel", "familia", "cliente", "mes",
        "forecast", "LL", "UL",
        "model", "mse",
        "is_seasonal", "seasonal_period", "seasonal_strength",
        "n_obs", "train_size",
        "train_end", "status", "generated_at",
    ]]

def build_model_ranking_df(df_series: pd.DataFrame, job_id: str | None = None) -> pd.DataFrame:
    df_series = df_series.copy()
    df_series["mes"] = pd.to_datetime(df_series["mes"])
    df_series["y"] = pd.to_numeric(df_series["y"], errors="coerce").fillna(0.0)

    keys = (
        df_series[["nivel", "familia", "cliente"]]
        .drop_duplicates()
        .sort_values(["nivel", "familia", "cliente"])
    )

    out_rows = []
    total = len(keys)

    for i, (nivel, familia, cliente) in enumerate(keys.values, start=1):
        if job_id:
            pct = 55 + int((i / total) * 25)   # ranking entre 55% y 80%
            update_job(
                job_id,
                pct,
                "building_model_ranking",
                f"Calculando ranking {i} de {total}: {nivel} | {familia} | {cliente}"
            )

        sub = df_series[
            (df_series["nivel"] == nivel) &
            (df_series["familia"] == familia) &
            (df_series["cliente"] == cliente)
        ].copy().sort_values("mes")

        y = sub.set_index("mes")["y"].asfreq("MS").fillna(0.0).astype(float)
        n_obs = int(len(y))

        if n_obs < MIN_MONTHS:
            continue

        initial_train = _choose_initial_train(len(y))

        try:
            ranking, s_info = evaluate_models(
                y,
                initial_train=initial_train,
                h=1,
                ma_window=3,
                arima_order=(1, 1, 1),
                sarima_order=(1, 1, 1),
                seasonal_candidates=(12, 6),
                seasonal_threshold=0.30,
                seasonal_min_cycles=2,
                ml_lags=12,
                ml_add_calendar=True,
                ml_min_obs=18,
            )

            ranking = ranking.copy().reset_index(drop=True)
            ranking["rank"] = ranking.index + 1
            ranking["nivel"] = nivel
            ranking["familia"] = familia
            ranking["cliente"] = cliente
            ranking["n_obs"] = n_obs
            ranking["is_seasonal"] = bool(s_info.get("is_seasonal", False))
            ranking["seasonal_period"] = s_info.get("seasonal_period", None)
            ranking["seasonal_strength"] = float(s_info.get("strength", 0.0))

            out_rows.append(ranking)

        except Exception as e:
            out_rows.append(pd.DataFrame({
                "model": ["ERROR"],
                "mse": [np.nan],
                "rank": [1],
                "nivel": [nivel],
                "familia": [familia],
                "cliente": [cliente],
                "n_obs": [n_obs],
                "is_seasonal": [False],
                "seasonal_period": [None],
                "seasonal_strength": [0.0],
            }))

    if not out_rows:
        return pd.DataFrame(columns=[
            "model", "mse", "rank",
            "nivel", "familia", "cliente",
            "n_obs", "is_seasonal", "seasonal_period", "seasonal_strength"
        ])

    out = pd.concat(out_rows, ignore_index=True)

    out["mse"] = pd.to_numeric(out["mse"], errors="coerce")
    out.loc[np.isinf(out["mse"]), "mse"] = np.nan

    return out[[
        "nivel", "familia", "cliente",
        "rank", "model", "mse",
        "n_obs", "is_seasonal", "seasonal_period", "seasonal_strength"
    ]]

def run_gold_forecast_job(job_id: str):
    bucket = settings.s3_bucket
    silver_key = settings.silver_key
    series_key = settings.gold_series_key
    forecast_key = settings.gold_forecast_key
    ranking_key = settings.gold_ranking_key

    try:
        update_job(job_id, 5, "checking_silver", "Validando existencia de la capa silver")

        if not object_exists(bucket, silver_key):
            raise FileNotFoundError(f"No existe silver en s3://{bucket}/{silver_key}")

        update_job(job_id, 10, "loading_silver", "Leyendo silver desde S3")
        df = read_parquet_from_s3(bucket, silver_key)

        update_job(job_id, 20, "building_series", "Construyendo series jerárquicas")
        df_series = build_series_hier_df(df)

        update_job(job_id, 30, "saving_series", "Guardando series_hier en S3")
        write_parquet_to_s3(df_series, bucket, series_key)

        update_job(job_id, 40, "forecasting", "Calculando forecast jerárquico")
        df_fore = build_forecast_hier_df(df_series, job_id=job_id)

        update_job(job_id, 55, "saving_forecast", "Guardando forecast_hier en S3")
        write_parquet_to_s3(df_fore, bucket, forecast_key)

        update_job(job_id, 60, "building_ranking", "Construyendo ranking de modelos")
        df_rank = build_model_ranking_df(df_series, job_id=job_id)

        update_job(job_id, 85, "saving_ranking", "Guardando model_ranking en S3")
        write_parquet_to_s3(df_rank, bucket, ranking_key)

        result = {
            "bucket": bucket,
            "series_key": series_key,
            "forecast_key": forecast_key,
            "ranking_key": ranking_key,
            "series_rows": int(df_series.shape[0]),
            "forecast_rows": int(df_fore.shape[0]),
            "ranking_rows": int(df_rank.shape[0]),
        }
        complete_job(job_id, result=result)

    except Exception as e:
        fail_job(job_id, str(e))