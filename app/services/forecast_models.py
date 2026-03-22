"""
forecast_models.py

Modelos "suaves" (clásicos) + ML tabular para pronóstico univariado mensual.

Incluye:
- Modelos clásicos
- Random Forest / XGBoost
- Detección automática de estacionalidad
- Walk-forward con MSE
- Tuning tipo grilla para RF/XGB
- Bandas por cuantiles de residuales
- Fitted/residuales consistentes para Monte Carlo
- Prints de avance en consola
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple, List, Dict
from itertools import product
import warnings

import numpy as np
import pandas as pd

from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt, ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

from sklearn.ensemble import RandomForestRegressor

try:
    from xgboost import XGBRegressor
    _HAS_XGB = True
except Exception:
    XGBRegressor = None
    _HAS_XGB = False


# =========================
# FLAGS
# =========================
INCLUDE_NAIVE = True
INCLUDE_ARIMA = True
INCLUDE_SEASONAL_NAIVE = True
INCLUDE_ML = True
INCLUDE_XGBOOST = False   # <- por estabilidad, déjalo False mientras estabilizas

VERBOSE_FORECAST = True
VERBOSE_WALK_FORWARD = False   # <- evita imprimir cada split
VERBOSE_TUNING = True
VERBOSE_MODEL_FIT = False


# =========================
# Warnings: silenciar ruido repetitivo
# =========================
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings(
    "ignore",
    message="Too few observations to estimate starting parameters",
)
warnings.filterwarnings(
    "ignore",
    message="Maximum Likelihood optimization failed to converge",
)
warnings.filterwarnings(
    "ignore",
    message="invalid value encountered in divide",
)
warnings.filterwarnings(
    "ignore",
    message="`sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel`",
)


# =========================
# Grillas por defecto ML tuning (LIVIANAS)
# =========================
RF_PARAM_GRID = {
    "n_estimators": [200, 300, 400, 500, 800],
    "max_depth": [3, 4, 5, 6, 8, None],
    "min_samples_leaf": [1, 2, 3, 4],
    "max_features": ["sqrt", 0.5, 0.7, 1.0],
    "bootstrap": [True],
}

XGB_PARAM_GRID = {
    "n_estimators": [200, 400],
    "learning_rate": [0.05, 0.1],
    "max_depth": [2, 3],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
    "min_child_weight": [1],
}


def _vprint(msg: str):
    if VERBOSE_FORECAST:
        print(msg)


def _wvprint(msg: str):
    if VERBOSE_WALK_FORWARD:
        print(msg)


def _tvprint(msg: str):
    if VERBOSE_TUNING:
        print(msg)


def _mvprint(msg: str):
    if VERBOSE_MODEL_FIT:
        print(msg)


# =========================
# Tipos
# =========================
@dataclass
class ForecastResult:
    model_name: str
    y_hat: pd.Series
    fitted: Optional[pd.Series] = None
    details: Optional[dict] = None


# =========================
# Preparación de series
# =========================
def to_monthly_series(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    freq: str = "MS",
    fill_missing: str = "zero",
) -> pd.Series:
    tmp = df[[date_col, value_col]].copy()
    tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
    tmp[value_col] = pd.to_numeric(tmp[value_col], errors="coerce")
    tmp = tmp.dropna(subset=[date_col, value_col])

    if tmp.empty:
        return pd.Series(dtype=float)

    tmp["__month"] = tmp[date_col].values.astype("datetime64[M]")
    tmp["__month"] = pd.to_datetime(tmp["__month"])

    y = tmp.groupby("__month")[value_col].sum().sort_index()
    y.index = pd.DatetimeIndex(y.index)
    y = y.asfreq(freq)

    if fill_missing == "zero":
        y = y.fillna(0.0)

    return y.astype(float)


def check_min_length(y: pd.Series, min_len: int) -> None:
    if len(y) < min_len:
        raise ValueError(f"Serie insuficiente: requiere >= {min_len} puntos, tiene {len(y)}.")


# =========================
# Helpers ML
# =========================
def _make_supervised_features(
    y: pd.Series,
    lags: int = 12,
    add_calendar: bool = True,
) -> pd.DataFrame:
    y = y.asfreq("MS").astype(float)
    df = pd.DataFrame({"y": y})

    for i in range(1, lags + 1):
        df[f"lag_{i}"] = df["y"].shift(i)

    if add_calendar:
        m = df.index.month.astype(float)
        df["m_sin"] = np.sin(2 * np.pi * m / 12.0)
        df["m_cos"] = np.cos(2 * np.pi * m / 12.0)

    out = df.dropna()
    _mvprint(
        f"[_make_supervised_features] len_y={len(y)} | lags={lags} | add_calendar={add_calendar} | rows_out={len(out)}"
    )
    return out


def _recursive_forecast_ml(
    model,
    y_train: pd.Series,
    h: int,
    lags: int = 12,
    add_calendar: bool = True,
) -> pd.Series:
    y_train = y_train.asfreq("MS").fillna(0.0).astype(float)
    df_sup = _make_supervised_features(y_train, lags=lags, add_calendar=add_calendar)
    if df_sup.empty:
        raise ValueError("No hay suficientes datos para features ML (subserie muy corta).")

    X = df_sup.drop(columns=["y"]).values
    y_target = df_sup["y"].values.astype(float)
    _mvprint(f"[_recursive_forecast_ml] Entrenando {type(model).__name__} con X.shape={X.shape}")
    model.fit(X, y_target)

    history = y_train.values.astype(float).tolist()
    idx = pd.date_range(y_train.index[-1] + pd.offsets.MonthBegin(1), periods=h, freq="MS")

    preds = []
    for dt in idx:
        lag_vals = history[-lags:][::-1]
        if len(lag_vals) < lags:
            lag_vals = lag_vals + [0.0] * (lags - len(lag_vals))

        feats = list(lag_vals[:lags])

        if add_calendar:
            mm = float(dt.month)
            feats.append(float(np.sin(2 * np.pi * mm / 12.0)))
            feats.append(float(np.cos(2 * np.pi * mm / 12.0)))

        yhat = float(model.predict(np.array(feats, dtype=float).reshape(1, -1))[0])
        yhat = max(0.0, yhat)
        preds.append(yhat)
        history.append(yhat)

    return pd.Series(preds, index=idx, dtype=float)


def _one_step_ahead_fitted_ml(
    model_factory: Callable[[], object],
    y: pd.Series,
    lags: int = 12,
    add_calendar: bool = True,
    initial_train: int = 12,
) -> pd.Series:
    y = y.asfreq("MS").fillna(0.0).astype(float)
    fitted = pd.Series(index=y.index, dtype=float)

    start = max(initial_train, lags + 1)
    if len(y) <= start:
        _mvprint(f"[_one_step_ahead_fitted_ml] Serie muy corta para fitted ML: len={len(y)}, start={start}")
        return fitted

    for t in range(start, len(y)):
        y_train = y.iloc[:t]
        df_sup = _make_supervised_features(y_train, lags=lags, add_calendar=add_calendar)
        if df_sup.empty:
            continue

        X = df_sup.drop(columns=["y"]).values
        y_target = df_sup["y"].values.astype(float)

        model = model_factory()
        model.fit(X, y_target)

        dt = y.index[t]
        lag_vals = y.iloc[t - lags:t].values.astype(float)[::-1]
        feats = list(lag_vals)

        if add_calendar:
            mm = float(dt.month)
            feats.append(float(np.sin(2 * np.pi * mm / 12.0)))
            feats.append(float(np.cos(2 * np.pi * mm / 12.0)))

        fitted.iloc[t] = float(model.predict(np.array(feats, dtype=float).reshape(1, -1))[0])

    return fitted


# =========================
# Detección de estacionalidad
# =========================
def detect_seasonality_acf(
    y: pd.Series,
    seasonal_period: int = 12,
    threshold: float = 0.30,
    min_cycles: int = 2,
) -> Dict[str, float | bool]:
    y = y.dropna().astype(float)

    enough = len(y) >= (min_cycles * seasonal_period + 1)
    if not enough:
        return {"is_seasonal": False, "strength": 0.0, "seasonal_period": seasonal_period}

    strength = float(y.autocorr(lag=seasonal_period))
    is_seasonal = bool(np.isfinite(strength) and abs(strength) >= threshold)

    return {"is_seasonal": is_seasonal, "strength": strength, "seasonal_period": seasonal_period}


def detect_best_seasonality(
    y: pd.Series,
    candidate_periods: Tuple[int, ...] = (12, 6),
    threshold: float = 0.30,
    min_cycles: int = 2,
) -> Dict[str, float | bool | int]:
    best = {"is_seasonal": False, "seasonal_period": int(candidate_periods[0]), "strength": 0.0}

    for p in candidate_periods:
        r = detect_seasonality_acf(y, seasonal_period=p, threshold=threshold, min_cycles=min_cycles)
        if abs(float(r["strength"])) > abs(float(best["strength"])):
            best = {
                "is_seasonal": bool(r["is_seasonal"]),
                "seasonal_period": int(p),
                "strength": float(r["strength"]),
            }

    best["is_seasonal"] = bool(
        abs(float(best["strength"])) >= threshold
        and len(y.dropna()) >= (min_cycles * int(best["seasonal_period"]) + 1)
    )
    _vprint(f"[detect_best_seasonality] best={best}")
    return best


# =========================
# Forecasts clásicos
# =========================
def forecast_naive(y_train: pd.Series, h: int = 1) -> ForecastResult:
    last = float(y_train.iloc[-1])
    idx = pd.date_range(y_train.index[-1] + pd.offsets.MonthBegin(1), periods=h, freq="MS")
    fitted = y_train.shift(1)
    return ForecastResult("naive", pd.Series([last] * h, index=idx, dtype=float), fitted=fitted)


def forecast_seasonal_naive(y_train: pd.Series, h: int = 1, season_length: int = 12) -> ForecastResult:
    check_min_length(y_train, season_length + 1)
    idx = pd.date_range(y_train.index[-1] + pd.offsets.MonthBegin(1), periods=h, freq="MS")
    vals = [float(y_train.iloc[-season_length + (i % season_length)]) for i in range(h)]
    fitted = y_train.shift(season_length)
    return ForecastResult(
        "seasonal_naive",
        pd.Series(vals, index=idx, dtype=float),
        fitted=fitted,
        details={"season_length": season_length},
    )


def forecast_simple_average(y_train: pd.Series, h: int = 1) -> ForecastResult:
    avg = float(y_train.mean())
    idx = pd.date_range(y_train.index[-1] + pd.offsets.MonthBegin(1), periods=h, freq="MS")
    yhat = pd.Series([avg] * h, index=idx, dtype=float)
    fitted = y_train.expanding().mean().shift(1)
    return ForecastResult("simple_average", yhat, fitted=fitted, details={"mean": avg})


def forecast_moving_average(y_train: pd.Series, h: int = 1, window: int = 3) -> ForecastResult:
    check_min_length(y_train, min(window, len(y_train)))
    avg = float(y_train.tail(window).mean())
    idx = pd.date_range(y_train.index[-1] + pd.offsets.MonthBegin(1), periods=h, freq="MS")
    yhat = pd.Series([avg] * h, index=idx, dtype=float)
    fitted = y_train.rolling(window).mean().shift(1)
    return ForecastResult("moving_average", yhat, fitted=fitted, details={"window": window})


def forecast_ses(y_train: pd.Series, h: int = 1) -> ForecastResult:
    model = SimpleExpSmoothing(y_train, initialization_method="estimated")
    fit = model.fit(optimized=True)
    idx = pd.date_range(y_train.index[-1] + pd.offsets.MonthBegin(1), periods=h, freq="MS")
    yhat = pd.Series(fit.forecast(h), index=idx, dtype=float)
    fitted = pd.Series(fit.fittedvalues, index=y_train.index, dtype=float)
    return ForecastResult("ses", yhat, fitted=fitted, details={"params": dict(fit.params)})


def forecast_holt(y_train: pd.Series, h: int = 1, damped_trend: bool = False) -> ForecastResult:
    model = Holt(y_train, initialization_method="estimated", damped_trend=damped_trend)
    fit = model.fit(optimized=True)
    idx = pd.date_range(y_train.index[-1] + pd.offsets.MonthBegin(1), periods=h, freq="MS")
    yhat = pd.Series(fit.forecast(h), index=idx, dtype=float)
    fitted = pd.Series(fit.fittedvalues, index=y_train.index, dtype=float)
    return ForecastResult("holt", yhat, fitted=fitted, details={"params": dict(fit.params), "damped_trend": damped_trend})


def forecast_holt_winters(
    y_train: pd.Series,
    h: int = 1,
    season_length: int = 12,
    trend: str = "add",
    seasonal: str = "add",
    damped_trend: bool = False,
) -> ForecastResult:
    check_min_length(y_train, season_length * 2)
    model = ExponentialSmoothing(
        y_train,
        trend=trend,
        damped_trend=damped_trend,
        seasonal=seasonal,
        seasonal_periods=season_length,
        initialization_method="estimated",
    )
    fit = model.fit(optimized=True)
    idx = pd.date_range(y_train.index[-1] + pd.offsets.MonthBegin(1), periods=h, freq="MS")
    yhat = pd.Series(fit.forecast(h), index=idx, dtype=float)
    fitted = pd.Series(fit.fittedvalues, index=y_train.index, dtype=float)
    return ForecastResult(
        "holt_winters",
        yhat,
        fitted=fitted,
        details={
            "params": dict(fit.params),
            "season_length": season_length,
            "trend": trend,
            "seasonal": seasonal,
            "damped_trend": damped_trend,
        },
    )


def forecast_linear_regression(y_train: pd.Series, h: int = 1) -> ForecastResult:
    yv = y_train.values.astype(float)
    n = len(yv)
    check_min_length(y_train, 3)

    t = np.arange(1, n + 1, dtype=float)
    b, a = np.polyfit(t, yv, deg=1)

    fitted = pd.Series(a + b * t, index=y_train.index, dtype=float)

    t_future = np.arange(n + 1, n + h + 1, dtype=float)
    yhat_vals = a + b * t_future
    idx = pd.date_range(y_train.index[-1] + pd.offsets.MonthBegin(1), periods=h, freq="MS")
    yhat = pd.Series(yhat_vals, index=idx, dtype=float)

    return ForecastResult("linear_regression", yhat, fitted=fitted, details={"a": float(a), "b": float(b)})


def forecast_arima(y_train: pd.Series, h: int = 1, order: Tuple[int, int, int] = (1, 1, 1)) -> ForecastResult:
    model = SARIMAX(
        y_train,
        order=order,
        seasonal_order=(0, 0, 0, 0),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fit = model.fit(disp=False)
    idx = pd.date_range(y_train.index[-1] + pd.offsets.MonthBegin(1), periods=h, freq="MS")
    yhat = pd.Series(fit.forecast(steps=h), index=idx, dtype=float)
    fitted = pd.Series(fit.fittedvalues, index=y_train.index, dtype=float)
    return ForecastResult("arima", yhat, fitted=fitted, details={"order": order})


def forecast_sarima(
    y_train: pd.Series,
    h: int = 1,
    order: Tuple[int, int, int] = (1, 1, 1),
    seasonal_order: Tuple[int, int, int, int] = (0, 1, 1, 12),
) -> ForecastResult:
    check_min_length(y_train, seasonal_order[3] * 2)
    model = SARIMAX(
        y_train,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fit = model.fit(disp=False)
    idx = pd.date_range(y_train.index[-1] + pd.offsets.MonthBegin(1), periods=h, freq="MS")
    yhat = pd.Series(fit.forecast(steps=h), index=idx, dtype=float)
    fitted = pd.Series(fit.fittedvalues, index=y_train.index, dtype=float)
    return ForecastResult("sarima", yhat, fitted=fitted, details={"order": order, "seasonal_order": seasonal_order})


# =========================
# Forecasts ML
# =========================
def forecast_random_forest(
    y_train: pd.Series,
    h: int = 1,
    lags: int = 12,
    add_calendar: bool = True,
    n_estimators: int = 400,
    random_state: int = 42,
    max_depth: int | None = None,
    min_samples_leaf: int = 2,
    max_features: str | float | None = "sqrt",
    bootstrap: bool = True,
) -> ForecastResult:
    def _factory():
        return RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            bootstrap=bootstrap,
            n_jobs=1,   # <- clave: evitar explosión de hilos
        )

    _mvprint(
        "[forecast_random_forest] "
        f"h={h} | lags={lags} | n_estimators={n_estimators} | max_depth={max_depth} | "
        f"min_samples_leaf={min_samples_leaf} | max_features={max_features} | bootstrap={bootstrap}"
    )

    yhat = _recursive_forecast_ml(_factory(), y_train, h=h, lags=lags, add_calendar=add_calendar)
    fitted = _one_step_ahead_fitted_ml(_factory, y_train, lags=lags, add_calendar=add_calendar, initial_train=12)

    return ForecastResult(
        "random_forest",
        yhat,
        fitted=fitted,
        details={
            "lags": lags,
            "add_calendar": add_calendar,
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
            "max_features": max_features,
            "bootstrap": bootstrap,
        },
    )


def forecast_xgboost(
    y_train: pd.Series,
    h: int = 1,
    lags: int = 12,
    add_calendar: bool = True,
    random_state: int = 42,
    n_estimators: int = 700,
    learning_rate: float = 0.05,
    max_depth: int = 4,
    subsample: float = 0.9,
    colsample_bytree: float = 0.9,
    min_child_weight: float = 1.0,
) -> ForecastResult:
    if not _HAS_XGB:
        raise ImportError("xgboost no está instalado. Instala con: pip install xgboost")

    def _factory():
        return XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            min_child_weight=min_child_weight,
            objective="reg:squarederror",
            random_state=random_state,
            n_jobs=1,   # <- clave
            verbosity=0,
        )

    _mvprint(
        "[forecast_xgboost] "
        f"h={h} | lags={lags} | n_estimators={n_estimators} | learning_rate={learning_rate} | "
        f"max_depth={max_depth} | subsample={subsample} | colsample_bytree={colsample_bytree} | "
        f"min_child_weight={min_child_weight}"
    )

    yhat = _recursive_forecast_ml(_factory(), y_train, h=h, lags=lags, add_calendar=add_calendar)
    fitted = _one_step_ahead_fitted_ml(_factory, y_train, lags=lags, add_calendar=add_calendar, initial_train=12)

    return ForecastResult(
        "xgboost",
        yhat,
        fitted=fitted,
        details={
            "lags": lags,
            "add_calendar": add_calendar,
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "min_child_weight": min_child_weight,
        },
    )


# =========================
# Walk-forward
# =========================
def walk_forward_mse(
    y: pd.Series,
    forecaster: Callable[[pd.Series, int], ForecastResult],
    h: int = 1,
    initial_train: int = 12,
    model_name: str = "unknown_model",
) -> float:
    y = y.dropna().astype(float)
    check_min_length(y, initial_train + h)

    errors = []
    total_steps = len(y) - h - initial_train + 1
    _vprint(f"[walk_forward_mse] model={model_name} | steps={total_steps}")

    for k, t in enumerate(range(initial_train, len(y) - h + 1), start=1):
        y_train = y.iloc[:t]
        y_true = y.iloc[t:t + h].values.astype(float)

        try:
            _wvprint(f"[walk_forward_mse] model={model_name} | split={k}/{total_steps} | train_len={len(y_train)}")
            res = forecaster(y_train, h)
            y_pred = res.y_hat.values.astype(float)
            step_mse = np.mean((y_true - y_pred) ** 2)
            errors.append(step_mse)
        except Exception as e:
            _vprint(
                f"[walk_forward_mse] ERROR model={model_name} | split={k}/{total_steps} | "
                f"train_len={len(y_train)} | {type(e).__name__}: {e}"
            )
            return float("inf")

    final_mse = float(np.mean(errors)) if errors else float("inf")
    _vprint(f"[walk_forward_mse] model={model_name} | final_mse={final_mse:.6f}")
    return final_mse


def tune_ml_model_walk_forward(
    y: pd.Series,
    model_type: str,
    param_grid: dict,
    initial_train: int = 12,
    h: int = 1,
    lags: int = 12,
    add_calendar: bool = True,
) -> Tuple[dict, float]:
    y = y.dropna().astype(float)
    check_min_length(y, initial_train + h)

    if not param_grid:
        return {}, float("inf")

    keys = list(param_grid.keys())
    combos = list(product(*param_grid.values()))
    best_score = float("inf")
    best_params = {}

    _tvprint(
        f"[tune_ml_model_walk_forward] model_type={model_type} | combos={len(combos)} | "
        f"initial_train={initial_train} | h={h} | lags={lags}"
    )

    for i, values in enumerate(combos, start=1):
        params = dict(zip(keys, values))
        _tvprint(f"[tune_ml_model_walk_forward] model_type={model_type} | combo={i}/{len(combos)} | params={params}")

        try:
            if model_type == "rf":
                fn = lambda yt, hh, p=params: forecast_random_forest(
                    yt,
                    h=hh,
                    lags=lags,
                    add_calendar=add_calendar,
                    **p,
                )
                model_name = "random_forest_tuning"
            elif model_type == "xgb":
                if not _HAS_XGB:
                    continue
                fn = lambda yt, hh, p=params: forecast_xgboost(
                    yt,
                    h=hh,
                    lags=lags,
                    add_calendar=add_calendar,
                    **p,
                )
                model_name = "xgboost_tuning"
            else:
                raise ValueError(f"model_type no soportado: {model_type}")

            mse = walk_forward_mse(
                y,
                fn,
                h=h,
                initial_train=initial_train,
                model_name=model_name,
            )

            _tvprint(f"[tune_ml_model_walk_forward] model_type={model_type} | combo={i}/{len(combos)} | mse={mse:.6f}")

            if np.isfinite(mse) and mse < best_score:
                best_score = mse
                best_params = params.copy()
                _tvprint(
                    f"[tune_ml_model_walk_forward] NUEVO MEJOR -> model_type={model_type} | "
                    f"best_score={best_score:.6f} | best_params={best_params}"
                )

        except Exception as e:
            _tvprint(
                f"[tune_ml_model_walk_forward] ERROR model_type={model_type} | params={params} | "
                f"{type(e).__name__}: {e}"
            )
            continue

    _tvprint(
        f"[tune_ml_model_walk_forward] FIN model_type={model_type} | "
        f"best_score={best_score:.6f} | best_params={best_params}"
    )
    return best_params, best_score


def evaluate_models(
    y: pd.Series,
    initial_train: int = 12,
    h: int = 1,
    ma_window: int = 3,
    arima_order: Tuple[int, int, int] = (1, 1, 1),
    sarima_order: Tuple[int, int, int] = (1, 1, 1),
    seasonal_candidates: Tuple[int, ...] = (12, 6),
    seasonal_threshold: float = 0.30,
    seasonal_min_cycles: int = 2,
    ml_lags: int = 12,
    ml_add_calendar: bool = True,
    ml_min_obs: int = 18,
    ml_tuning: bool = False,
    rf_param_grid: Optional[dict] = None,
    xgb_param_grid: Optional[dict] = None,
) -> Tuple[pd.DataFrame, Dict[str, float | bool | int]]:
    y = y.dropna().astype(float)
    check_min_length(y, initial_train + h)

    _vprint(
        f"[evaluate_models] len_y={len(y)} | initial_train={initial_train} | h={h} | "
        f"ml_lags={ml_lags} | ml_min_obs={ml_min_obs} | ml_tuning={ml_tuning}"
    )

    s_info = detect_best_seasonality(
        y,
        candidate_periods=seasonal_candidates,
        threshold=seasonal_threshold,
        min_cycles=seasonal_min_cycles,
    )
    is_seasonal = bool(s_info["is_seasonal"])
    s = int(s_info["seasonal_period"])

    if len(y) < (2 * s):
        is_seasonal = False
        s_info["is_seasonal"] = False

    models: List[Tuple[str, Callable[[pd.Series, int], ForecastResult]]] = []

    models.append(("simple_average", lambda yt, hh: forecast_simple_average(yt, hh)))
    models.append(("moving_average", lambda yt, hh: forecast_moving_average(yt, hh, window=ma_window)))
    models.append(("ses", lambda yt, hh: forecast_ses(yt, hh)))
    models.append(("holt", lambda yt, hh: forecast_holt(yt, hh)))
    models.append(("linear_regression", lambda yt, hh: forecast_linear_regression(yt, hh)))

    if INCLUDE_NAIVE:
        models.append(("naive", lambda yt, hh: forecast_naive(yt, hh)))

    if INCLUDE_ARIMA:
        models.append(("arima", lambda yt, hh: forecast_arima(yt, hh, order=arima_order)))

    best_rf_params = {}
    best_xgb_params = {}

    if INCLUDE_ML and len(y) >= ml_min_obs:
        _vprint("[evaluate_models] ML habilitado para esta serie")

        if ml_tuning:
            rf_grid = rf_param_grid or RF_PARAM_GRID
            best_rf_params, _ = tune_ml_model_walk_forward(
                y=y,
                model_type="rf",
                param_grid=rf_grid,
                initial_train=initial_train,
                h=h,
                lags=ml_lags,
                add_calendar=ml_add_calendar,
            )

            if not best_rf_params:
                best_rf_params = {
                    "n_estimators": 200,
                    "random_state": 42,
                    "max_depth": 6,
                    "min_samples_leaf": 1,
                    "max_features": "sqrt",
                    "bootstrap": True,
                }
            else:
                best_rf_params = {
                    **best_rf_params,
                    "random_state": 42,
                }

            _vprint(f"[evaluate_models] best_rf_params={best_rf_params}")

            models.append((
                "random_forest",
                lambda yt, hh, params=best_rf_params: forecast_random_forest(
                    yt,
                    hh,
                    lags=ml_lags,
                    add_calendar=ml_add_calendar,
                    **params,
                )
            ))

            if INCLUDE_XGBOOST and _HAS_XGB:
                xgb_grid = xgb_param_grid or XGB_PARAM_GRID
                best_xgb_params, _ = tune_ml_model_walk_forward(
                    y=y,
                    model_type="xgb",
                    param_grid=xgb_grid,
                    initial_train=initial_train,
                    h=h,
                    lags=ml_lags,
                    add_calendar=ml_add_calendar,
                )

                if not best_xgb_params:
                    best_xgb_params = {
                        "random_state": 42,
                        "n_estimators": 400,
                        "learning_rate": 0.05,
                        "max_depth": 3,
                        "subsample": 0.8,
                        "colsample_bytree": 0.8,
                        "min_child_weight": 1.0,
                    }
                else:
                    best_xgb_params = {
                        **best_xgb_params,
                        "random_state": 42,
                    }

                _vprint(f"[evaluate_models] best_xgb_params={best_xgb_params}")

                models.append((
                    "xgboost",
                    lambda yt, hh, params=best_xgb_params: forecast_xgboost(
                        yt,
                        hh,
                        lags=ml_lags,
                        add_calendar=ml_add_calendar,
                        **params,
                    )
                ))
        else:
            models.append((
                "random_forest",
                lambda yt, hh: forecast_random_forest(
                    yt,
                    hh,
                    lags=ml_lags,
                    add_calendar=ml_add_calendar,
                )
            ))

            if INCLUDE_XGBOOST and _HAS_XGB:
                models.append((
                    "xgboost",
                    lambda yt, hh: forecast_xgboost(
                        yt,
                        hh,
                        lags=ml_lags,
                        add_calendar=ml_add_calendar,
                    )
                ))
    else:
        _vprint("[evaluate_models] ML NO habilitado para esta serie")

    if is_seasonal:
        _vprint(f"[evaluate_models] Modelos estacionales habilitados con s={s}")
        if INCLUDE_SEASONAL_NAIVE:
            models.append(("seasonal_naive", lambda yt, hh: forecast_seasonal_naive(yt, hh, season_length=s)))
        models.append(("holt_winters", lambda yt, hh: forecast_holt_winters(yt, hh, season_length=s)))
        models.append(("sarima", lambda yt, hh: forecast_sarima(yt, hh, order=sarima_order, seasonal_order=(0, 1, 1, s))))

    rows = []
    total_models = len(models)
    _vprint(f"[evaluate_models] Total modelos a evaluar: {total_models}")

    for i, (name, fn) in enumerate(models, start=1):
        _vprint(f"[evaluate_models] Evaluando modelo {i}/{total_models}: {name}")
        mse = walk_forward_mse(y, fn, h=h, initial_train=initial_train, model_name=name)

        params_used = None
        if name == "random_forest" and best_rf_params:
            params_used = best_rf_params
        elif name == "xgboost" and best_xgb_params:
            params_used = best_xgb_params

        rows.append({
            "model": name,
            "mse": mse,
            "best_params": params_used,
        })

    out = pd.DataFrame(rows).sort_values("mse", ascending=True).reset_index(drop=True)
    _vprint("[evaluate_models] Ranking final:")
    _vprint(out.to_string(index=False))
    return out, s_info


def fit_best_and_forecast(
    y: pd.Series,
    ranking: pd.DataFrame,
    seasonality_info: Dict[str, float | bool | int],
    h: int = 1,
    ma_window: int = 3,
    arima_order: Tuple[int, int, int] = (1, 1, 1),
    sarima_order: Tuple[int, int, int] = (1, 1, 1),
    ml_lags: int = 12,
    ml_add_calendar: bool = True,
) -> ForecastResult:
    y = y.dropna().astype(float)
    if y.empty:
        raise ValueError("Serie vacía.")

    best = str(ranking.iloc[0]["model"])
    best_params = ranking.iloc[0].get("best_params", None)
    if best_params is None or (isinstance(best_params, float) and pd.isna(best_params)):
        best_params = {}

    s = int(seasonality_info.get("seasonal_period", 12))
    is_seasonal = bool(seasonality_info.get("is_seasonal", False))

    _vprint(f"[fit_best_and_forecast] best={best} | best_params={best_params}")

    if best == "simple_average":
        return forecast_simple_average(y, h)
    if best == "moving_average":
        return forecast_moving_average(y, h, window=ma_window)
    if best == "ses":
        return forecast_ses(y, h)
    if best == "holt":
        return forecast_holt(y, h)
    if best == "holt_winters":
        if not is_seasonal or len(y) < (2 * s):
            return forecast_holt(y, h)
        return forecast_holt_winters(y, h, season_length=s)
    if best == "linear_regression":
        return forecast_linear_regression(y, h)

    if best == "seasonal_naive":
        if not is_seasonal or len(y) < (s + 1):
            return forecast_naive(y, h)
        return forecast_seasonal_naive(y, h, season_length=s)

    if best == "naive":
        return forecast_naive(y, h)

    if best == "arima":
        return forecast_arima(y, h, order=arima_order)

    if best == "sarima":
        if not is_seasonal or len(y) < (2 * s):
            return forecast_arima(y, h, order=arima_order)
        return forecast_sarima(y, h, order=sarima_order, seasonal_order=(0, 1, 1, s))

    if best == "random_forest":
        return forecast_random_forest(
            y,
            h,
            lags=ml_lags,
            add_calendar=ml_add_calendar,
            **best_params,
        )

    if best == "xgboost":
        if not _HAS_XGB:
            return forecast_random_forest(
                y,
                h,
                lags=ml_lags,
                add_calendar=ml_add_calendar,
            )
        return forecast_xgboost(
            y,
            h,
            lags=ml_lags,
            add_calendar=ml_add_calendar,
            **best_params,
        )

    raise ValueError(f"Modelo desconocido: {best}")


# =========================
# Bandas por residuales
# =========================
def residual_quantile_bands(
    y: pd.Series,
    model_name: str,
    seasonality_info: dict,
    alpha: float = 0.05,
    ma_window: int = 3,
    arima_order: Tuple[int, int, int] = (1, 1, 1),
    sarima_order: Tuple[int, int, int] = (1, 1, 1),
    ml_lags: int = 12,
    ml_add_calendar: bool = True,
) -> Dict[str, float]:
    y = y.dropna().astype(float)
    s = int(seasonality_info.get("seasonal_period", 12))
    is_seasonal = bool(seasonality_info.get("is_seasonal", False))

    fitted = None

    if model_name == "simple_average":
        fitted = y.expanding().mean().shift(1)

    elif model_name == "moving_average":
        fitted = y.rolling(ma_window).mean().shift(1)

    elif model_name == "ses":
        fit = SimpleExpSmoothing(y, initialization_method="estimated").fit(optimized=True)
        fitted = pd.Series(fit.fittedvalues, index=y.index)

    elif model_name == "holt":
        fit = Holt(y, initialization_method="estimated").fit(optimized=True)
        fitted = pd.Series(fit.fittedvalues, index=y.index)

    elif model_name == "holt_winters":
        if not is_seasonal or len(y) < (2 * s):
            fit = Holt(y, initialization_method="estimated").fit(optimized=True)
            fitted = pd.Series(fit.fittedvalues, index=y.index)
        else:
            fit = ExponentialSmoothing(
                y,
                trend="add",
                seasonal="add",
                seasonal_periods=s,
                initialization_method="estimated",
            ).fit(optimized=True)
            fitted = pd.Series(fit.fittedvalues, index=y.index)

    elif model_name == "linear_regression":
        n = len(y)
        t = np.arange(1, n + 1, dtype=float)
        b, a = np.polyfit(t, y.values.astype(float), deg=1)
        fitted = pd.Series(a + b * t, index=y.index, dtype=float)

    elif model_name == "seasonal_naive":
        fitted = y.shift(s)

    elif model_name == "naive":
        fitted = y.shift(1)

    elif model_name == "arima":
        fit = SARIMAX(
            y,
            order=arima_order,
            seasonal_order=(0, 0, 0, 0),
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False)
        fitted = pd.Series(fit.fittedvalues, index=y.index)

    elif model_name == "sarima":
        if not is_seasonal:
            fit = SARIMAX(
                y,
                order=arima_order,
                seasonal_order=(0, 0, 0, 0),
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(disp=False)
        else:
            fit = SARIMAX(
                y,
                order=sarima_order,
                seasonal_order=(0, 1, 1, s),
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(disp=False)
        fitted = pd.Series(fit.fittedvalues, index=y.index)

    elif model_name == "random_forest":
        def _rf_factory():
            return RandomForestRegressor(
                n_estimators=200,
                random_state=42,
                min_samples_leaf=2,
                max_features="sqrt",
                bootstrap=True,
                n_jobs=1,
            )
        fitted = _one_step_ahead_fitted_ml(_rf_factory, y, lags=ml_lags, add_calendar=ml_add_calendar, initial_train=12)

    elif model_name == "xgboost":
        if not _HAS_XGB:
            def _rf_factory():
                return RandomForestRegressor(
                    n_estimators=200,
                    random_state=42,
                    min_samples_leaf=2,
                    max_features="sqrt",
                    bootstrap=True,
                    n_jobs=1,
                )
            fitted = _one_step_ahead_fitted_ml(_rf_factory, y, lags=ml_lags, add_calendar=ml_add_calendar, initial_train=12)
        else:
            def _xgb_factory():
                return XGBRegressor(
                    n_estimators=400,
                    learning_rate=0.05,
                    max_depth=3,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    min_child_weight=1.0,
                    objective="reg:squarederror",
                    random_state=42,
                    n_jobs=1,
                    verbosity=0,
                )
            fitted = _one_step_ahead_fitted_ml(_xgb_factory, y, lags=ml_lags, add_calendar=ml_add_calendar, initial_train=12)

    else:
        raise ValueError(f"Modelo no soportado para residuales: {model_name}")

    e = (y - fitted).dropna()
    if e.empty:
        return {"q_low": 0.0, "q_high": 0.0}

    e_active = e[y.loc[e.index] > 0]
    min_active = 8
    e_use = e_active if len(e_active) >= min_active else e

    out = {"q_low": float(e_use.quantile(alpha)), "q_high": float(e_use.quantile(1 - alpha))}
    _vprint(f"[residual_quantile_bands] model={model_name} | q_low={out['q_low']:.6f} | q_high={out['q_high']:.6f}")
    return out


# =========================
# Fitted + residuales
# =========================
def get_fitted_and_residuals(
    y: pd.Series,
    model_name: str,
    seasonality_info: dict | None = None,
    ma_window: int = 3,
    arima_order: Tuple[int, int, int] = (1, 1, 1),
    sarima_order: Tuple[int, int, int] = (1, 1, 1),
    ml_lags: int = 12,
    ml_add_calendar: bool = True,
):
    y = y.asfreq("MS").fillna(0.0).astype(float)
    name = (model_name or "").lower().strip()

    s = 12
    is_seasonal = False
    if seasonality_info:
        s = int(seasonality_info.get("seasonal_period", 12) or 12)
        is_seasonal = bool(seasonality_info.get("is_seasonal", False))

    fitted = None

    if name == "simple_average":
        fitted = y.expanding().mean().shift(1)

    elif name == "moving_average":
        fitted = y.rolling(ma_window).mean().shift(1)

    elif name == "ses":
        fit = SimpleExpSmoothing(y, initialization_method="estimated").fit(optimized=True)
        fitted = pd.Series(fit.fittedvalues, index=y.index, dtype=float)

    elif name == "holt":
        fit = Holt(y, initialization_method="estimated").fit(optimized=True)
        fitted = pd.Series(fit.fittedvalues, index=y.index, dtype=float)

    elif name == "holt_winters":
        if not is_seasonal or len(y) < (2 * s):
            fit = Holt(y, initialization_method="estimated").fit(optimized=True)
            fitted = pd.Series(fit.fittedvalues, index=y.index, dtype=float)
        else:
            fit = ExponentialSmoothing(
                y,
                trend="add",
                seasonal="add",
                seasonal_periods=s,
                initialization_method="estimated",
            ).fit(optimized=True)
            fitted = pd.Series(fit.fittedvalues, index=y.index, dtype=float)

    elif name == "linear_regression":
        n = len(y)
        t = np.arange(1, n + 1, dtype=float)
        b, a = np.polyfit(t, y.values.astype(float), deg=1)
        fitted = pd.Series(a + b * t, index=y.index, dtype=float)

    elif name == "seasonal_naive":
        fitted = y.shift(s)

    elif name == "naive":
        fitted = y.shift(1)

    elif name == "arima":
        model = SARIMAX(
            y,
            order=arima_order,
            seasonal_order=(0, 0, 0, 0),
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False)
        fitted = pd.Series(model.fittedvalues, index=y.index, dtype=float)

    elif name == "sarima":
        if is_seasonal and s > 1:
            model = SARIMAX(
                y,
                order=sarima_order,
                seasonal_order=(0, 1, 1, s),
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(disp=False)
        else:
            model = SARIMAX(
                y,
                order=arima_order,
                seasonal_order=(0, 0, 0, 0),
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(disp=False)
        fitted = pd.Series(model.fittedvalues, index=y.index, dtype=float)

    elif name == "random_forest":
        def _rf_factory():
            return RandomForestRegressor(
                n_estimators=200,
                random_state=42,
                min_samples_leaf=2,
                max_features="sqrt",
                bootstrap=True,
                n_jobs=1,
            )
        fitted = _one_step_ahead_fitted_ml(_rf_factory, y, lags=ml_lags, add_calendar=ml_add_calendar, initial_train=12)

    elif name == "xgboost":
        if not _HAS_XGB:
            def _rf_factory():
                return RandomForestRegressor(
                    n_estimators=200,
                    random_state=42,
                    min_samples_leaf=2,
                    max_features="sqrt",
                    bootstrap=True,
                    n_jobs=1,
                )
            fitted = _one_step_ahead_fitted_ml(_rf_factory, y, lags=ml_lags, add_calendar=ml_add_calendar, initial_train=12)
        else:
            def _xgb_factory():
                return XGBRegressor(
                    n_estimators=400,
                    learning_rate=0.05,
                    max_depth=3,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    min_child_weight=1.0,
                    objective="reg:squarederror",
                    random_state=42,
                    n_jobs=1,
                    verbosity=0,
                )
            fitted = _one_step_ahead_fitted_ml(_xgb_factory, y, lags=ml_lags, add_calendar=ml_add_calendar, initial_train=12)

    else:
        fitted = y.shift(1)

    resid = (y - fitted).dropna()
    _vprint(f"[get_fitted_and_residuals] model={name} | resid_len={len(resid)}")
    return fitted, resid