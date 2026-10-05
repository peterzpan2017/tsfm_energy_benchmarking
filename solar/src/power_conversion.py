import numpy as np
import pandas as pd


def compute_t_cell(poa, temp_air, noct=45):
    """Compute cell temperature using the simple NOCT model."""
    return temp_air + (poa / 800.0) * (noct - 20.0)


def to_power(poa, temp_air, p_dc0=2400, gamma=-0.005, noct=45):
    """Convert POA irradiance to DC power using the PVWatts-style model."""
    poa = np.asarray(poa)
    temp_air = np.asarray(temp_air)
    T_cell = compute_t_cell(poa, temp_air, noct)
    P_dc = (poa / 1000.0) * p_dc0 * (1 + gamma * (T_cell - 25.0))
    return np.maximum(P_dc, 0)


def convert_solar_result_to_power(result, temperature_df, temperature_col="AIRTEMP"):
    """Convert target, median prediction, and available POA quantiles to DC power.

    The temperature series is used only for the deterministic POA-to-power conversion
    after forecasting. Rows are aligned by item_id and timestamp.
    """
    out = result.copy()
    temp = temperature_df[["item_id", "timestamp", temperature_col]].copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    temp["timestamp"] = pd.to_datetime(temp["timestamp"])

    # Use string keys internally so this also works with Moirai item IDs.
    out["_item_key"] = out["item_id"].astype(str)
    temp["_item_key"] = temp["item_id"].astype(str)
    temp = temp[["_item_key", "timestamp", temperature_col]].drop_duplicates(
        ["_item_key", "timestamp"]
    )
    out = out.merge(temp, on=["_item_key", "timestamp"], how="left", validate="many_to_one")
    if out[temperature_col].isna().any():
        raise ValueError("Missing air-temperature values for some solar forecast rows.")

    temp_air = out[temperature_col].to_numpy()
    cols = [c for c in ["target", "prediction"] if c in out.columns]
    cols += [c for c in out.columns if _is_quantile_column(c)]
    for col in dict.fromkeys(cols):
        out[col] = to_power(out[col].to_numpy(), temp_air)

    return out.drop(columns=["_item_key", temperature_col])


def _is_quantile_column(name):
    try:
        q = float(name)
    except (TypeError, ValueError):
        return False
    return 0 < q < 1
