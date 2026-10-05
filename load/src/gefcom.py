from pathlib import Path
import numpy as np
import pandas as pd

QUANTILES=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
LOAD_WEATHER=[f"w{i}" for i in range(1,26)]
GEFCOM_LOAD_EVAL_TASKS=list(range(5,17))
GEFCOM_LOAD_NAIVE_AQL=np.array([18.7384,22.7585,13.2163,8.3626,10.9162,16.9937,13.4038,17.3151,13.8374,6.4237,10.9380,34.0685],dtype=float)

def load_gefcom_load(path):
    df=pd.read_csv(path)
    df["timestamp"]=pd.to_datetime(df["timestamp"])
    need=["item_id","timestamp","target","task_id",*LOAD_WEATHER]
    miss=[c for c in need if c not in df.columns]
    if miss: raise KeyError(f"Missing GEFCom load columns: {miss}")
    return df[need].sort_values(["task_id","timestamp"]).reset_index(drop=True)

def load_initial_and_tasks(df):
    context=df[df.task_id==1].copy()
    tasks=[(task_id,df[df.task_id==task_id].copy()) for task_id in range(2,17)]
    return context,tasks

def _row_aql(df, quantiles=QUANTILES):
    y=df["target"].to_numpy(float); losses=[]
    for q in quantiles:
        p=df[str(q)].to_numpy(float)
        losses.append(np.where(y<p,(1-q)*(p-y),q*(y-p)))
    return np.column_stack(losses).mean(axis=1)

def evaluate_gefcom_load(result, quantiles=QUANTILES):
    d=result.copy(); d["row_aql"]=_row_aql(d,quantiles)
    summary=d.groupby("task_id",as_index=False)["row_aql"].mean().rename(columns={"row_aql":"raw_aql"})
    summary=summary[summary.task_id.isin(GEFCOM_LOAD_EVAL_TASKS)].sort_values("task_id").reset_index(drop=True)
    if len(summary)!=12: raise ValueError(f"Expected 12 evaluated load tasks, got {len(summary)}")
    summary["naive_aql"]=GEFCOM_LOAD_NAIVE_AQL
    summary["normalized_aql"]=1.0 - summary.raw_aql/summary.naive_aql
    return summary
