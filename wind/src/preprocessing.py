from dataclasses import dataclass
from pathlib import Path
import pandas as pd

NWP_FEATURES = ["WS", "WINDGUST", "RH2", "SST", "U", "V", "SLP"]
SITES = ("E05N", "E06")
CONTEXT_START = "2019-11-01 00:00"
CONTEXT_END = "2020-03-15 00:00"
TEST_START = "2020-03-15 00:00"
TEST_END = "2020-07-31 00:00"
CONTEXT_LENGTH = 2048
PREDICTION_LENGTH = 24
SEED = 42

@dataclass
class WindData:
    full: pd.DataFrame
    context: pd.DataFrame
    test: pd.DataFrame
    context_by_site: dict
    test_by_site: dict
    daily_by_site: dict

def _site_frame(targets, nwp, site, item_id):
    target = targets[["timestamp", site]].rename(columns={site: "target"}).copy()
    target["timestamp"] = pd.to_datetime(target["timestamp"])
    site_nwp = nwp.filter(regex=rf"^{site}_").copy()
    site_nwp = site_nwp.rename(columns=lambda c: c.removeprefix(f"{site}_"))
    if len(target) != len(nwp):
        raise ValueError(f"Target/NWP row mismatch: {len(target)} vs {len(nwp)}")
    frame = site_nwp.copy()
    frame.insert(0, "target", target["target"].to_numpy())
    frame.insert(0, "timestamp", pd.to_datetime(nwp["timestamp"]).to_numpy())
    frame.insert(0, "item_id", item_id)
    missing = [c for c in NWP_FEATURES if c not in frame.columns]
    if missing:
        raise KeyError(f"Missing {site} NWP columns: {missing}")
    return frame[["item_id", "timestamp", "target", *NWP_FEATURES]].sort_values("timestamp").reset_index(drop=True)

def _daily_windows(df):
    out = []
    for day, group in df.sort_values("timestamp").groupby(df["timestamp"].dt.floor("D")):
        if len(group) != 24:
            raise ValueError(f"Incomplete Wind day {day.date()}: {len(group)} rows")
        out.append((day, group.copy()))
    return out

def load_wind(data_dir="data"):
    data_dir = Path(data_dir)
    target_path = data_dir / "hub-height wind.csv"
    nwp_path = data_dir / "nwp_data.csv"
    if not target_path.exists() or not nwp_path.exists():
        raise FileNotFoundError(f"Expected {target_path} and {nwp_path}. See data/README.md.")
    targets = pd.read_csv(target_path)
    nwp = pd.read_csv(nwp_path)
    targets["timestamp"] = pd.to_datetime(targets["timestamp"])
    nwp["timestamp"] = pd.to_datetime(nwp["timestamp"])
    frames = {site: _site_frame(targets, nwp, site, i+1) for i, site in enumerate(SITES)}
    context_by_site = {site: df[(df.timestamp >= CONTEXT_START) & (df.timestamp < CONTEXT_END)].copy() for site, df in frames.items()}
    test_by_site = {site: df[(df.timestamp >= TEST_START) & (df.timestamp < TEST_END)].copy() for site, df in frames.items()}
    daily_by_site = {site: _daily_windows(test_by_site[site]) for site in SITES}
    context = pd.concat(context_by_site.values(), ignore_index=True)
    test = pd.concat(test_by_site.values(), ignore_index=True)
    full = pd.concat([context, test], ignore_index=True).sort_values(["item_id", "timestamp"]).reset_index(drop=True)
    return WindData(full, context, test, context_by_site, test_by_site, daily_by_site)

def describe_wind(data):
    return {
        "context_rows": len(data.context),
        "test_rows": len(data.test),
        "test_days_per_site": {s: len(data.daily_by_site[s]) for s in SITES},
        "context_start": data.context.timestamp.min(),
        "context_end": data.context.timestamp.max(),
        "test_start": data.test.timestamp.min(),
        "test_end": data.test.timestamp.max(),
    }
