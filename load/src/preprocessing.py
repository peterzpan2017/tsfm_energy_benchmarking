from dataclasses import dataclass
from pathlib import Path
import pandas as pd

CONTEXT_LENGTH=2048
PREDICTION_LENGTH=24
SITES=(1,)
META={"Year","Month","Day","Hour","Load","target","item_id","timestamp"}

@dataclass
class TrackData:
    full: pd.DataFrame
    context: pd.DataFrame
    test: pd.DataFrame
    context_by_site: dict
    test_by_site: dict
    daily_by_site: dict
    covariates: list

def _stamp(df):
    out=df.copy()
    if "timestamp" not in out:
        base_year=2019
        out["timestamp"]=pd.to_datetime({"year":base_year+out["Year"],"month":out["Month"],"day":out["Day"],"hour":out["Hour"]-1},errors="raise")
    else: out["timestamp"]=pd.to_datetime(out["timestamp"])
    out=out.rename(columns={"Load":"target"})
    if "item_id" not in out: out["item_id"]=1
    return out

def _daily(df):
    out=[]
    for d,g in df.sort_values("timestamp").groupby(df["timestamp"].dt.floor("D")):
        if len(g)>=PREDICTION_LENGTH: out.append((d,g.iloc[:PREDICTION_LENGTH].copy()))
    return out

def load_data(data_dir="data"):
    data_dir=Path(data_dir); tr=data_dir/"Load-Train.xlsx"; te=data_dir/"Load-Test.xlsx"
    if tr.exists() and te.exists():
        context=_stamp(pd.read_excel(tr)); test=_stamp(pd.read_excel(te)); full=pd.concat([context,test],ignore_index=True)
    else:
        p=data_dir/"load_prepared.csv"
        if not p.exists(): raise FileNotFoundError(f"Expected {tr} and {te}, or {p}")
        full=_stamp(pd.read_csv(p)); cutoff=pd.Timestamp("2022-01-01"); context=full[full.timestamp<cutoff].copy(); test=full[full.timestamp>=cutoff].copy()
    cov=[c for c in full.columns if c not in META]
    keep=["item_id","timestamp","target",*cov]
    context=context[keep].sort_values(["item_id","timestamp"]).reset_index(drop=True); test=test[keep].sort_values(["item_id","timestamp"]).reset_index(drop=True); full=pd.concat([context,test],ignore_index=True)
    ids=tuple(sorted(full.item_id.unique())); cb={i:context[context.item_id==i].copy() for i in ids}; tb={i:test[test.item_id==i].copy() for i in ids}; db={i:_daily(tb[i]) for i in ids}
    return TrackData(full,context,test,cb,tb,db,cov)

def describe_data(d): return {"context_rows":len(d.context),"test_rows":len(d.test),"test_windows":sum(len(v) for v in d.daily_by_site.values()),"covariates":len(d.covariates),"context_start":d.context.timestamp.min(),"context_end":d.context.timestamp.max(),"test_start":d.test.timestamp.min(),"test_end":d.test.timestamp.max()}
