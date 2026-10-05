from dataclasses import dataclass
from pathlib import Path
import pandas as pd

CONTEXT_LENGTH = 2048
PREDICTION_LENGTH = 24
SITES = (1,)
RAW_COVARIATES = ["AIRTEMP", "RH_AVG", "DEWPT", "WS", "GHI", "DNI", "DIFF"]


@dataclass
class TrackData:
    full: pd.DataFrame
    context: pd.DataFrame
    test: pd.DataFrame
    context_by_site: dict
    test_by_site: dict
    daily_by_site: dict
    covariates: list
    future_proxy_map: dict


def _daily(df):
    out = []
    for d, g in df.sort_values("timestamp").groupby(df["timestamp"].dt.floor("D")):
        if len(g) >= PREDICTION_LENGTH:
            out.append((d, g.iloc[:PREDICTION_LENGTH].copy()))
    return out


def load_data(data_dir="data"):
    p = Path(data_dir) / "solardata_modified.csv"
    if not p.exists():
        raise FileNotFoundError(f"Expected {p}")

    df = pd.read_csv(p)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if "item_id" not in df:
        df["item_id"] = 1

    raw_covariates = [c for c in RAW_COVARIATES if c in df.columns]
    if not raw_covariates:
        raise KeyError("None of the expected raw solar covariates were found.")

    future_proxy_map = {}
    for c in raw_covariates:
        lag_col = f"{c}_lag24"
        if lag_col not in df.columns:
            raise KeyError(
                f"Missing {lag_col}. The solar dataset must already contain the precomputed lag-24 "
                f"proxies for future {c}."
            )
        future_proxy_map[c] = lag_col

    keep = [
        "item_id", "timestamp", "target",
        *raw_covariates,
        *future_proxy_map.values(),
    ]
    # preserve order while avoiding duplicates
    keep = list(dict.fromkeys(keep))
    df = df[keep].sort_values(["item_id", "timestamp"]).reset_index(drop=True)

    cutoff = pd.Timestamp("2023-09-30 23:00")
    context = df[df.timestamp <= cutoff].copy()
    test = df[df.timestamp > cutoff].copy()

    ids = tuple(sorted(df.item_id.unique()))
    cb = {i: context[context.item_id == i].copy() for i in ids}
    tb = {i: test[test.item_id == i].copy() for i in ids}
    db = {i: _daily(tb[i]) for i in ids}

    return TrackData(
        df, context, test, cb, tb, db,
        raw_covariates, future_proxy_map,
    )


def future_covariates(future_df, data):
    """Return forecast-horizon covariates from the precomputed *_lag24 columns.

    Output columns use the ORIGINAL covariate names expected by model APIs,
    but values come from the corresponding *_lag24 columns.
    """
    out = future_df[["item_id", "timestamp"]].copy()
    for raw_col, proxy_col in data.future_proxy_map.items():
        out[raw_col] = future_df[proxy_col].to_numpy()
    return out


def future_covariate_array(future_df, data):
    """Array of precomputed future proxy covariates ordered as data.covariates."""
    proxy = future_covariates(future_df, data)
    return proxy[data.covariates].to_numpy()


def describe_data(d):
    return {
        "context_rows": len(d.context),
        "test_rows": len(d.test),
        "test_windows": sum(len(v) for v in d.daily_by_site.values()),
        "past_covariates": d.covariates,
        "future_covariate_proxies": d.future_proxy_map,
        "context_start": d.context.timestamp.min(),
        "context_end": d.context.timestamp.max(),
        "test_start": d.test.timestamp.min(),
        "test_end": d.test.timestamp.max(),
    }
