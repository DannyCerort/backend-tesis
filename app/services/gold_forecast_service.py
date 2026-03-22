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
MIN_MONTHS = 18
ALPHA = 0.05

DATE_COL = "fecha_facturacion"
VALUE_COL = "VAL TOTAL"
CLIENTE_COL = "ID_NOMBRE_CLIENTE"
FAM_COL = "FAMILIA_SERVICIO"

ML_LAGS = 3
ML_MIN_OBS = 12

RF_TUNING_GRID = {
    "n_estimators": [200, 400],
    "max_depth": [5, None],
    "min_samples_leaf": [1, 2],
}

XGB_TUNING_GRID = {
    "n_estimators": [200, 500],
    "learning_rate": [0.05, 0.1],
    "max_depth": [3, 4],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
}


def read_parquet_from_s3(bucket: str, key: str) -> pd.DataFrame:
    print(f"[read_parquet_from_s3] Leyendo parquet desde s3://{bucket}/{key}")
    s3 = get_s3_client()
    response = s3.get_object(Bucket=bucket, Key=key)
    parquet_bytes = response["Body"].read()
    df = pd.read_parquet(io.BytesIO(parquet_bytes), engine="pyarrow")
    print(f"[read_parquet_from_s3] Filas leídas: {len(df)}")
    return df


def write_parquet_to_s3(df: pd.DataFrame, bucket: str, key: str):
    print(f"[write_parquet_to_s3] Escribiendo {len(df)} filas en s3://{bucket}/{key}")
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
    print(f"[write_parquet_to_s3] Escritura completada: s3://{bucket}/{key}")


def build_series_hier_df(df: pd.DataFrame) -> pd.DataFrame:
    print("[build_series_hier_df] Construyendo series jerárquicas...")
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
    print(f"[build_series_hier_df] Series únicas detectadas: {len(keys)}")

    series = []
    for j, (nivel, familia, cliente) in enumerate(keys.values, start=1):
        print(f"[build_series_hier_df] Reindexando serie {j}/{len(keys)} -> {nivel} | {familia} | {cliente}")

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
    print(f"[build_series_hier_df] Filas finales series_hier: {len(out2)}")
    return out2


def _choose_initial_train(n: int) -> int:
    if n >= 36:
        initial_train = 24
    elif n >= 24:
        initial_train = 18
    elif n >= 18:
        initial_train = 14
    else:
        initial_train = max(6, int(0.6 * n))

    return min(initial_train, max(6, n - 2))


def forecast_one_series(y: pd.Series):
    y = y.asfreq("MS").fillna(0.0).astype(float)

    seasonality_info = {"is_seasonal": False, "seasonal_period": None, "seasonal_strength": 0.0}

    zero_ratio = float((y == 0).mean()) if len(y) else 1.0
    print(
        f"[forecast_one_series] Serie con {len(y)} meses | "
        f"mean={y.mean():.2f} | std={y.std():.2f} | zero_ratio={zero_ratio:.2%}"
    )

    if y.std() == 0 or zero_ratio >= 0.6:
        print("[forecast_one_series] Aplicando fallback por serie constante o muy intermitente")
        res = forecast_moving_average(y, h=HORIZON, window=3) if len(y) >= 3 else forecast_naive(y, h=HORIZON)
        return res.y_hat, 0.0, 0.0, res.model_name, np.nan, "FALLBACK", seasonality_info

    initial_train = _choose_initial_train(len(y))
    print(
        f"[forecast_one_series] initial_train={initial_train} | "
        f"HORIZON={HORIZON} | ML_LAGS={ML_LAGS} | ML_MIN_OBS={ML_MIN_OBS}"
    )

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
        ml_lags=ML_LAGS,
        ml_add_calendar=True,
        ml_min_obs=ML_MIN_OBS,
        ml_tuning=True,
        rf_param_grid=RF_TUNING_GRID,
        xgb_param_grid=XGB_TUNING_GRID,
    )

    print("[forecast_one_series] Ranking top 5:")
    print(ranking.head(5).to_string(index=False))

    print("[forecast_one_series] Ranking completo:")
    print(ranking.to_string(index=False))

    ml_rows = ranking[ranking["model"].isin(["random_forest", "xgboost"])]
    if not ml_rows.empty:
        print("[forecast_one_series] Estado modelos ML:")
        print(ml_rows.to_string(index=False))
    else:
        print("[forecast_one_series] No aparecieron modelos ML en el ranking")

    best_model = str(ranking.iloc[0]["model"])
    mse = float(ranking.iloc[0]["mse"])
    best_params = ranking.iloc[0].get("best_params", None)
    print(f"[forecast_one_series] Mejor modelo: {best_model} | mse={mse:.4f} | best_params={best_params}")

    res = fit_best_and_forecast(
        y,
        ranking,
        seasonality_info=s_info,
        h=HORIZON,
        ma_window=3,
        arima_order=(1, 1, 1),
        sarima_order=(1, 1, 1),
        ml_lags=ML_LAGS,
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
        ml_lags=ML_LAGS,
        ml_add_calendar=True,
    )

    q_low, q_high = float(bands["q_low"]), float(bands["q_high"])
    print(f"[forecast_one_series] Bandas residuales -> q_low={q_low:.4f}, q_high={q_high:.4f}")

    seasonality_info = {
        "is_seasonal": bool(s_info.get("is_seasonal", False)),
        "seasonal_period": int(s_info["seasonal_period"]) if pd.notna(s_info.get("seasonal_period", None)) else None,
        "seasonal_strength": float(s_info.get("strength", 0.0)),
    }

    return res.y_hat, q_low, q_high, best_model, mse, "OK", seasonality_info


def build_forecast_hier_df(df_series: pd.DataFrame, job_id: str | None = None) -> pd.DataFrame:
    print("[build_forecast_hier_df] Iniciando forecast jerárquico...")
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
    print(f"[build_forecast_hier_df] Total series a procesar: {total}")

    for i, (nivel, familia, cliente) in enumerate(keys.values, start=1):
        msg = f"Procesando serie {i}/{total}: {nivel} | {familia} | {cliente}"
        print(f"[build_forecast_hier_df] {msg}")

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
            print(f"[build_forecast_hier_df] Serie omitida por n_obs={n_obs} < MIN_MONTHS={MIN_MONTHS}")
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
            print(
                f"[build_forecast_hier_df] OK -> modelo={model} | mse={mse} | "
                f"is_seasonal={df_fore['is_seasonal'].iloc[0]}"
            )

        except Exception as e:
            print(f"[build_forecast_hier_df] ERROR en serie {nivel} | {familia} | {cliente}: {type(e).__name__} - {e}")
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
        print("[build_forecast_hier_df] No se generaron filas de forecast")
        return pd.DataFrame(columns=[
            "nivel", "familia", "cliente", "mes",
            "forecast", "LL", "UL",
            "model", "mse",
            "is_seasonal", "seasonal_period", "seasonal_strength",
            "n_obs", "train_size",
            "train_end", "status", "generated_at",
        ])

    out = pd.concat(out_rows, ignore_index=True)
    print(f"[build_forecast_hier_df] Forecast final filas: {len(out)}")

    return out[[
        "nivel", "familia", "cliente", "mes",
        "forecast", "LL", "UL",
        "model", "mse",
        "is_seasonal", "seasonal_period", "seasonal_strength",
        "n_obs", "train_size",
        "train_end", "status", "generated_at",
    ]]


def build_model_ranking_df(df_series: pd.DataFrame, job_id: str | None = None) -> pd.DataFrame:
    print("[build_model_ranking_df] Iniciando construcción de ranking de modelos...")
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
    print(f"[build_model_ranking_df] Total series a rankear: {total}")

    for i, (nivel, familia, cliente) in enumerate(keys.values, start=1):
        print(f"[build_model_ranking_df] Ranking serie {i}/{total}: {nivel} | {familia} | {cliente}")

        if job_id:
            pct = 55 + int((i / total) * 25)
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
            print(f"[build_model_ranking_df] Serie omitida por n_obs={n_obs} < MIN_MONTHS={MIN_MONTHS}")
            continue

        initial_train = _choose_initial_train(len(y))
        print(
            f"[build_model_ranking_df] initial_train={initial_train} | "
            f"ML_LAGS={ML_LAGS} | ML_MIN_OBS={ML_MIN_OBS}"
        )

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
                ml_lags=ML_LAGS,
                ml_add_calendar=True,
                ml_min_obs=ML_MIN_OBS,
                ml_tuning=True,
                rf_param_grid=RF_TUNING_GRID,
                xgb_param_grid=XGB_TUNING_GRID,
            )

            print("[build_model_ranking_df] Top 5 ranking:")
            print(ranking.head(5).to_string(index=False))

            print("[build_model_ranking_df] Ranking completo:")
            print(ranking.to_string(index=False))

            ml_rows = ranking[ranking["model"].isin(["random_forest", "xgboost"])]
            if not ml_rows.empty:
                print("[build_model_ranking_df] Estado modelos ML:")
                print(ml_rows.to_string(index=False))
            else:
                print("[build_model_ranking_df] No aparecieron modelos ML en el ranking")

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
            print(f"[build_model_ranking_df] ERROR en ranking {nivel} | {familia} | {cliente}: {type(e).__name__} - {e}")
            out_rows.append(pd.DataFrame({
                "model": ["ERROR"],
                "mse": [np.nan],
                "best_params": [None],
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
        print("[build_model_ranking_df] No se generaron rankings")
        return pd.DataFrame(columns=[
            "model", "mse", "best_params", "rank",
            "nivel", "familia", "cliente",
            "n_obs", "is_seasonal", "seasonal_period", "seasonal_strength"
        ])

    out = pd.concat(out_rows, ignore_index=True)

    out["mse"] = pd.to_numeric(out["mse"], errors="coerce")
    out.loc[np.isinf(out["mse"]), "mse"] = np.nan

    print(f"[build_model_ranking_df] Filas finales ranking: {len(out)}")

    return out[[
        "nivel", "familia", "cliente",
        "rank", "model", "mse", "best_params",
        "n_obs", "is_seasonal", "seasonal_period", "seasonal_strength"
    ]]


def run_gold_forecast_job(job_id: str):
    bucket = settings.s3_bucket
    silver_key = settings.silver_key
    series_key = settings.gold_series_key
    forecast_key = settings.gold_forecast_key
    ranking_key = settings.gold_ranking_key

    try:
        print(f"[run_gold_forecast_job] Iniciando job_id={job_id}")
        print(f"[run_gold_forecast_job] bucket={bucket}")
        print(f"[run_gold_forecast_job] silver_key={silver_key}")
        print(f"[run_gold_forecast_job] series_key={series_key}")
        print(f"[run_gold_forecast_job] forecast_key={forecast_key}")
        print(f"[run_gold_forecast_job] ranking_key={ranking_key}")
        print(f"[run_gold_forecast_job] Config ML -> ML_LAGS={ML_LAGS}, ML_MIN_OBS={ML_MIN_OBS}")
        print(f"[run_gold_forecast_job] RF_TUNING_GRID={RF_TUNING_GRID}")
        print(f"[run_gold_forecast_job] XGB_TUNING_GRID={XGB_TUNING_GRID}")

        update_job(job_id, 5, "checking_silver", "Validando existencia de la capa silver")

        if not object_exists(bucket, silver_key):
            raise FileNotFoundError(f"No existe silver en s3://{bucket}/{silver_key}")

        print("[run_gold_forecast_job] Silver encontrada correctamente")

        update_job(job_id, 10, "loading_silver", "Leyendo silver desde S3")
        df = read_parquet_from_s3(bucket, silver_key)
        print(f"[run_gold_forecast_job] Silver cargada: {df.shape}")

        update_job(job_id, 20, "building_series", "Construyendo series jerárquicas")
        df_series = build_series_hier_df(df)
        print(f"[run_gold_forecast_job] df_series shape: {df_series.shape}")

        update_job(job_id, 30, "saving_series", "Guardando series_hier en S3")
        write_parquet_to_s3(df_series, bucket, series_key)

        update_job(job_id, 40, "forecasting", "Calculando forecast jerárquico")
        df_fore = build_forecast_hier_df(df_series, job_id=job_id)
        print(f"[run_gold_forecast_job] df_fore shape: {df_fore.shape}")

        update_job(job_id, 55, "saving_forecast", "Guardando forecast_hier en S3")
        write_parquet_to_s3(df_fore, bucket, forecast_key)

        update_job(job_id, 60, "building_ranking", "Construyendo ranking de modelos")
        df_rank = build_model_ranking_df(df_series, job_id=job_id)
        print(f"[run_gold_forecast_job] df_rank shape: {df_rank.shape}")

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

        print(f"[run_gold_forecast_job] Job completado con resultado: {result}")
        complete_job(job_id, result=result)

    except Exception as e:
        print(f"[run_gold_forecast_job] ERROR: {type(e).__name__} - {e}")
        fail_job(job_id, str(e))