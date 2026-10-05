import numpy as np
from sklearn.metrics import mean_absolute_error, mean_pinball_loss, root_mean_squared_error

DEFAULT_QUANTILES = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

def compute_metrics(df, quantiles=DEFAULT_QUANTILES):
    data = df.copy()
    if "prediction" not in data.columns:
        if "0.5" not in data.columns:
            raise ValueError("Need 'prediction' or median quantile column '0.5'.")
        data["prediction"] = data["0.5"]
    missing = [str(q) for q in quantiles if str(q) not in data.columns]
    if missing:
        raise ValueError(f"Missing quantile columns: {missing}")
    y = data["target"].to_numpy()
    yhat = data["prediction"].to_numpy()
    qloss = [mean_pinball_loss(y, data[str(q)].to_numpy(), alpha=q) for q in quantiles]
    return {"MAE": float(mean_absolute_error(y,yhat)), "RMSE": float(root_mean_squared_error(y,yhat)), "aQL": float(np.mean(qloss))}

def print_metrics(df, quantiles=DEFAULT_QUANTILES, label=None):
    m = compute_metrics(df, quantiles)
    if label: print(label)
    print(f"MAE:  {m['MAE']:.6f}")
    print(f"RMSE: {m['RMSE']:.6f}")
    print(f"aQL:  {m['aQL']:.6f}")
    return m
