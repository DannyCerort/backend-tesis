import io
import numpy as np
import pandas as pd

from app.core.config import settings
from app.services.s3_service import get_s3_client, object_exists
from app.services.forecast_models import evaluate_models, get_fitted_and_residuals


def _read_gold_series() -> pd.DataFrame:
    bucket = settings.s3_bucket
    key = settings.gold_series_key

    if not object_exists(bucket, key):
        raise FileNotFoundError(f"No existe gold series en s3://{bucket}/{key}")

    s3 = get_s3_client()
    response = s3.get_object(Bucket=bucket, Key=key)
    parquet_bytes = response["Body"].read()
    return pd.read_parquet(io.BytesIO(parquet_bytes), engine="pyarrow")


def _read_gold_forecast() -> pd.DataFrame:
    bucket = settings.s3_bucket
    key = settings.gold_forecast_key

    if not object_exists(bucket, key):
        raise FileNotFoundError(f"No existe gold forecast en s3://{bucket}/{key}")

    s3 = get_s3_client()
    response = s3.get_object(Bucket=bucket, Key=key)
    parquet_bytes = response["Body"].read()
    return pd.read_parquet(io.BytesIO(parquet_bytes), engine="pyarrow")

def _read_gold_ranking() -> pd.DataFrame:
    bucket = settings.s3_bucket
    key = settings.gold_ranking_key

    if not object_exists(bucket, key):
        raise FileNotFoundError(f"No existe gold ranking en s3://{bucket}/{key}")

    s3 = get_s3_client()
    response = s3.get_object(Bucket=bucket, Key=key)
    parquet_bytes = response["Body"].read()
    return pd.read_parquet(io.BytesIO(parquet_bytes), engine="pyarrow")

def _build_forecast_options(df_series: pd.DataFrame):
    opt_empresa = ["EMPRESA | TOTAL"]

    familias = (
        df_series[df_series["nivel"] == "familia"]["familia"]
        .dropna().unique().tolist()
    )
    familias = [f for f in familias if f != "ALL"]
    opt_emp_serv = [f"EMPRESA→SERVICIO | {f}" for f in sorted(familias)]

    pairs = (
        df_series[df_series["nivel"] == "familia_cliente"][["familia", "cliente"]]
        .dropna()
        .drop_duplicates()
    )
    pairs = pairs[(pairs["familia"] != "ALL") & (pairs["cliente"] != "ALL")]
    opt_serv_cli = [f"SERVICIO→CLIENTE | {r.familia} | {r.cliente}" for r in pairs.itertuples(index=False)]

    return opt_empresa + opt_emp_serv + sorted(opt_serv_cli)


def _parse_scope(sel: str):
    if sel.startswith("EMPRESA |"):
        return "empresa", "ALL", "ALL", "Empresa (Total)"
    elif sel.startswith("EMPRESA→SERVICIO |"):
        familia = sel.split("|", 1)[1].strip()
        return "familia", familia, "ALL", f"Empresa → Servicio: {familia}"
    else:
        parts = [p.strip() for p in sel.split("|")]
        familia = parts[1]
        cliente = parts[2]
        return "familia_cliente", familia, cliente, f"Servicio → Cliente: {familia} / {cliente}"


def get_forecast_options():
    df_series = _read_gold_series()
    return {"options": _build_forecast_options(df_series)}


def get_forecast_prediction(scope: str, h: int = 6):
    df_series = _read_gold_series()
    df_fore = _read_gold_forecast()

    nivel, familia, cliente, titulo = _parse_scope(scope)

    hist = df_series[
        (df_series["nivel"] == nivel) &
        (df_series["familia"] == familia) &
        (df_series["cliente"] == cliente)
    ].copy().sort_values("mes")

    fore = df_fore[
        (df_fore["nivel"] == nivel) &
        (df_fore["familia"] == familia) &
        (df_fore["cliente"] == cliente)
    ].copy().sort_values("mes")

    if hist.empty:
        return {"title": titulo, "empty": True}

    fore_h = fore[["mes", "forecast", "LL", "UL"]].dropna().copy().sort_values("mes").head(int(h))

    historical = hist[["mes", "y"]].copy()
    historical["mes"] = pd.to_datetime(historical["mes"]).dt.strftime("%Y-%m-%d")

    if not fore_h.empty:
        fore_h["mes"] = pd.to_datetime(fore_h["mes"]).dt.strftime("%Y-%m-%d")

    meta_cols = [c for c in [
        "model", "mse", "status", "train_end", "generated_at",
        "n_obs", "train_size", "is_seasonal", "seasonal_period", "seasonal_strength"
    ] if c in fore.columns]

    meta = {}
    if meta_cols and not fore.empty:
        row = fore[meta_cols].drop_duplicates().head(1).to_dict(orient="records")
        meta = row[0] if row else {}

    return {
        "title": titulo,
        "empty": False,
        "historical": historical.to_dict(orient="records"),
        "forecast": fore_h.to_dict(orient="records") if not fore_h.empty else [],
        "meta": meta,
    }


def get_forecast_model_ranking(scope: str):
    df_rank = _read_gold_ranking()

    nivel, familia, cliente, titulo = _parse_scope(scope)

    ranking = df_rank[
        (df_rank["nivel"] == nivel) &
        (df_rank["familia"] == familia) &
        (df_rank["cliente"] == cliente)
    ].copy().sort_values("rank")

    if ranking.empty:
        return {"title": titulo, "ranking": []}

    # Sanitizar mse para JSON
    ranking["mse"] = pd.to_numeric(ranking["mse"], errors="coerce")
    ranking.loc[np.isinf(ranking["mse"]), "mse"] = np.nan
    ranking["mse"] = ranking["mse"].round(2)

    # Convertir NaN a None
    ranking = ranking.replace({np.nan: None})

    seasonality_info = {}
    first = ranking.head(1)
    if not first.empty:
        sp = first["seasonal_period"].iloc[0]
        ss = first["seasonal_strength"].iloc[0]

        seasonality_info = {
            "is_seasonal": bool(first["is_seasonal"].iloc[0]) if first["is_seasonal"].iloc[0] is not None else False,
            "seasonal_period": int(sp) if sp is not None else None,
            "seasonal_strength": float(ss) if ss is not None else 0.0,
        }

    return {
        "title": titulo,
        "ranking": ranking[["rank", "model", "mse"]].to_dict(orient="records"),
        "seasonality_info": seasonality_info,
    }


def get_forecast_montecarlo(scope: str, h: int = 6, n_sims: int = 3000, seed: int = 42, k_months: int = 36):
    df_series = _read_gold_series()
    df_fore = _read_gold_forecast()

    nivel, familia, cliente, titulo = _parse_scope(scope)

    hist = df_series[
        (df_series["nivel"] == nivel) &
        (df_series["familia"] == familia) &
        (df_series["cliente"] == cliente)
    ].copy().sort_values("mes")

    fore = df_fore[
        (df_fore["nivel"] == nivel) &
        (df_fore["familia"] == familia) &
        (df_fore["cliente"] == cliente)
    ].copy().sort_values("mes")

    if hist.empty or fore.empty:
        return {"title": titulo, "empty": True}

    fore_h = fore[["mes", "forecast", "LL", "UL"]].dropna().copy().sort_values("mes").head(int(h))
    if fore_h.empty:
        return {"title": titulo, "empty": True}

    y = hist.set_index("mes")["y"].asfreq("MS").fillna(0.0).astype(float)
    best_model = str(fore["model"].iloc[0]) if "model" in fore.columns else "naive"

    def _build_seasonality_info_from_fore(fore_df: pd.DataFrame):
        needed = {"is_seasonal", "seasonal_period", "seasonal_strength"}
        if needed.issubset(set(fore_df.columns)):
            return {
                "is_seasonal": bool(fore_df["is_seasonal"].iloc[0]),
                "seasonal_period": (
                    int(fore_df["seasonal_period"].iloc[0])
                    if pd.notna(fore_df["seasonal_period"].iloc[0]) else None
                ),
                "seasonal_strength": (
                    float(fore_df["seasonal_strength"].iloc[0])
                    if pd.notna(fore_df["seasonal_strength"].iloc[0]) else 0.0
                ),
            }
        return None

    seasonality_info = _build_seasonality_info_from_fore(fore)

    if seasonality_info is None:
        _, seasonality_info = evaluate_models(
            y,
            initial_train=12 if len(y) >= 18 else max(6, int(len(y) * 0.6)),
            h=1,
            ma_window=3,
            arima_order=(1, 1, 1),
            sarima_order=(1, 1, 1),
            seasonal_candidates=(12, 6),
            seasonal_threshold=0.30,
        )

    try:
        _, resid = get_fitted_and_residuals(
            y=y,
            model_name=best_model,
            seasonality_info=seasonality_info,
            ma_window=3,
            arima_order=(1, 1, 1),
            sarima_order=(1, 1, 1),
        )
    except Exception:
        fitted = y.rolling(3).mean().shift(1)
        resid = (y - fitted).dropna()

    if resid is None or len(resid) < 6:
        return {"title": titulo, "empty": True, "message": "Muy pocos datos para Monte Carlo"}

    resid_tail = resid.tail(k_months).values
    if len(resid_tail) < 6:
        resid_tail = resid.values

    h_real = int(len(fore_h))
    rng = np.random.default_rng(int(seed))

    sigma = float(np.std(resid_tail))
    sigma = max(sigma, 1e-9)

    eps = rng.normal(loc=0.0, scale=sigma, size=(int(n_sims), h_real))
    f_vec = fore_h["forecast"].astype(float).values.reshape(1, -1)
    y_sim_mat = np.clip(f_vec + eps, 0, None)

    y_sum = y_sim_mat.sum(axis=1)

    f_sum = float(fore_h["forecast"].sum())
    ll_sum = float(fore_h["LL"].sum())
    ul_sum = float(fore_h["UL"].sum())

    p_baja = float(np.mean(y_sum < ll_sum))
    p_alta = float(np.mean(y_sum > ul_sum))
    p_in = float(np.mean((y_sum >= ll_sum) & (y_sum <= ul_sum)))
    p_alert = p_baja + p_alta

    return {
        "title": titulo,
        "empty": False,
        "h": h_real,
        "forecast_sum": f_sum,
        "ll_sum": ll_sum,
        "ul_sum": ul_sum,
        "p_baja": p_baja,
        "p_alta": p_alta,
        "p_in": p_in,
        "p_alert": p_alert,
        "distribution": y_sum.tolist(),
    }