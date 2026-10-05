from dataclasses import dataclass
from pathlib import Path
import pandas as pd

CONTEXT_LENGTH=2048
PREDICTION_LENGTH=24
FUTURE_COVARIATES=["Exogenous 1","Exogenous 2","Exogenous 3"]
PAST_COVARIATES=["Exogenous 4"]
ALL_COVARIATES=FUTURE_COVARIATES+PAST_COVARIATES
SITES=(1,)

@dataclass
class TrackData:
    full: pd.DataFrame
    context: pd.DataFrame
    test: pd.DataFrame
    context_by_site: dict
    test_by_site: dict
    daily_by_site: dict

def _daily(df):
    out=[]
    for d,g in df.sort_values("timestamp").groupby(df["timestamp"].dt.floor("D")):
        if len(g)>=PREDICTION_LENGTH: out.append((d,g.iloc[:PREDICTION_LENGTH].copy()))
    return out

def load_data(data_dir="data"):
    p=Path(data_dir)/"lmp_NP15_data.csv"
    if not p.exists(): raise FileNotFoundError(f"Expected {p}")
    df=pd.read_csv(p)
    df=df.rename(columns={"Price":"target","Unnamed: 0":"timestamp"})
    df["timestamp"]=pd.to_datetime(df["timestamp"])
    if "item_id" not in df: df["item_id"]=1
    need=["item_id","timestamp","target",*ALL_COVARIATES]
    miss=[c for c in need if c not in df.columns]
    if miss: raise KeyError(f"Missing columns: {miss}")
    df=df[need].sort_values(["item_id","timestamp"]).reset_index(drop=True)
    context=df[(df.timestamp>=pd.Timestamp("2022-01-01"))&(df.timestamp<pd.Timestamp("2025-01-01"))].copy()
    test=df[(df.timestamp>=pd.Timestamp("2025-01-01"))&(df.timestamp<pd.Timestamp("2026-01-01"))].copy()
    ids=tuple(sorted(df.item_id.unique()))
    cb={i:context[context.item_id==i].copy() for i in ids}; tb={i:test[test.item_id==i].copy() for i in ids}; db={i:_daily(tb[i]) for i in ids}
    return TrackData(df,context,test,cb,tb,db)

def describe_data(d):
    return {"context_rows":len(d.context),"test_rows":len(d.test),"test_windows":sum(len(v) for v in d.daily_by_site.values()),"context_start":d.context.timestamp.min(),"context_end":d.context.timestamp.max(),"test_start":d.test.timestamp.min(),"test_end":d.test.timestamp.max()}
