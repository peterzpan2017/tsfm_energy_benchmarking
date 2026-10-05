from pathlib import Path
import numpy as np
import pandas as pd

QUANTILES=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
GEFCOM_PRICE_ALL_DATES=pd.to_datetime([
    "2013-06-16","2013-06-17","2013-06-24","2013-07-04","2013-07-09",
    "2013-07-13","2013-07-16","2013-07-18","2013-07-19","2013-07-20",
    "2013-07-24","2013-07-25","2013-12-07","2013-12-08","2013-12-17"
])
GEFCOM_PRICE_EVAL_DATES=GEFCOM_PRICE_ALL_DATES[3:]
GEFCOM_PRICE_NAIVE_AQL=np.array([4.02875,7.97208,5.70395,12.15104,38.33541,44.22979,18.22395,31.56729,42.94958,2.85583,3.20395,22.38333],dtype=float)
PRICE_COVARIATES=["Forecasted Total Load","Forecasted Zonal Load"]

def load_gefcom_price(path):
    df=pd.read_csv(path)
    df=df.drop(columns=[c for c in df.columns if c.startswith("Unnamed:")],errors="ignore")
    if "datetime" in df.columns: df=df.rename(columns={"datetime":"timestamp"})
    df["timestamp"]=pd.to_datetime(df["timestamp"])
    if "item_id" not in df.columns: df["item_id"]=1
    keep=["item_id","timestamp","target",*PRICE_COVARIATES]
    return df[keep].sort_values(["item_id","timestamp"]).reset_index(drop=True)

def price_context_and_days(df):
    context=df[df.timestamp < pd.Timestamp("2013-06-16")].copy()
    test=df[(df.timestamp>=pd.Timestamp("2013-06-16")) & (df.timestamp<pd.Timestamp("2013-12-18"))].copy()
    days=[]
    for day,g in test.groupby(test.timestamp.dt.floor("D")):
        if len(g)==24: days.append((day,g.sort_values("timestamp").copy()))
    return context, test, days

def _row_aql(df, quantiles=QUANTILES):
    y=df["target"].to_numpy(float); losses=[]
    for q in quantiles:
        p=df[str(q)].to_numpy(float)
        losses.append(np.where(y<p,(1-q)*(p-y),q*(y-p)))
    return np.column_stack(losses).mean(axis=1)

def evaluate_gefcom_price(result, quantiles=QUANTILES):
    d=result.copy(); d["timestamp"]=pd.to_datetime(d["timestamp"]); d["task_date"]=d.timestamp.dt.floor("D")
    d["row_aql"]=_row_aql(d,quantiles)
    summary=d.groupby("task_date",as_index=False)["row_aql"].mean().rename(columns={"row_aql":"raw_aql"})
    summary=summary[summary.task_date.isin(GEFCOM_PRICE_EVAL_DATES)].sort_values("task_date").reset_index(drop=True)
    if len(summary)!=12: raise ValueError(f"Expected 12 evaluated price tasks, got {len(summary)}")
    summary["naive_aql"]=GEFCOM_PRICE_NAIVE_AQL
    summary["normalized_aql"]=1.0 - summary.raw_aql/summary.naive_aql
    return summary
