import numpy as np
import pandas as pd


_INPUT_LEVELS = np.round(np.arange(0.1, 1.0, 0.1), 2)
_OUTPUT_LEVELS = np.array([
    0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5,
    0.6, 0.7, 0.8, 0.9, 0.95, 0.975, 0.99,
])


def compute_t_cell(poa, temp_air, noct=45):
    return temp_air + (poa / 800.0) * (noct - 20.0)


def to_power(poa, temp_air, p_dc0=2400, gamma=-0.005, noct=45):
    poa = np.asarray(poa)
    temp_air = np.asarray(temp_air)
    T_cell = compute_t_cell(poa, temp_air, noct)
    P_dc = (poa / 1000.0) * p_dc0 * (1 + gamma * (T_cell - 25.0))
    return np.maximum(P_dc, 0)


def _is_quantile_column(name):
    try:
        q = float(name)
    except (TypeError, ValueError):
        return False
    return 0 < q < 1


def _input_quantile_columns(df):
    available = {round(float(c), 6): c for c in df.columns if _is_quantile_column(c)}
    present = [round(float(q), 6) in available for q in _INPUT_LEVELS]
    if not any(present):
        return None
    if not all(present):
        raise ValueError("Distribution-based solar conversion requires all nine quantiles (0.1 to 0.9).")
    return [available[round(float(q), 6)] for q in _INPUT_LEVELS]


def convert_solar_result_to_power(result, temperature_df, temperature_col="AIRTEMP"):
    """Convert POA forecasts to solar power

    Air temperature is aligned by item_id/timestamp. If all nine POA
    quantiles exist, reconstruct the POA quantile function with bounded
    linear tails, transform 1,000 deterministic samples using each row's
    air temperature, then calculate 15 *power* quantiles. Existing
    quantile labels are maintained and tail labels follow their type.
    `prediction` is converted directly from the point POA forecast. The
    0.5 power quantile comes from the reconstructed power distribution.
    """
    out = result.copy()
    temp = temperature_df[["item_id", "timestamp", temperature_col]].copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    temp["timestamp"] = pd.to_datetime(temp["timestamp"])

    out["_item_key"] = out["item_id"].astype(str)
    temp["_item_key"] = temp["item_id"].astype(str)
    temp = temp[["_item_key", "timestamp", temperature_col]].drop_duplicates(
        ["_item_key", "timestamp"]
    )
    out = out.merge(temp, on=["_item_key", "timestamp"], how="left", validate="many_to_one")
    if out[temperature_col].isna().any():
        raise ValueError("Missing air-temperature values for some solar forecast rows.")

    temp_air = out[temperature_col].to_numpy(dtype=float)
    cols = _input_quantile_columns(out)
    if cols is not None:
        n_samples = 1000
        max_poa = 4000.0
        full_levels = np.r_[0.0, _INPUT_LEVELS, 1.0]
        u = (np.arange(n_samples) + 0.5) / n_samples
        poa_quantiles = out[cols].to_numpy(dtype=float)
        if not np.all(np.isfinite(poa_quantiles)):
            raise ValueError("POA quantiles contain NaN or Inf.")
        if np.any(np.diff(poa_quantiles, axis=1) < 0):
            raise ValueError("POA quantiles must be nondecreasing.")
        if np.any(poa_quantiles[:, 0] < 0) or np.any(poa_quantiles[:, -1] > max_poa):
            raise ValueError("POA quantiles outside the [0, 4000] W/m^2 support.")
        if not np.all(np.isfinite(temp_air)):
            raise ValueError("Air temperature contains NaN or Inf.")

        use_string_labels = isinstance(cols[0], str)
        level_to_existing = {round(float(c), 6): c for c in out.columns if _is_quantile_column(c)}
        power_quantiles = np.empty((len(out), len(_OUTPUT_LEVELS)))
        for i, (poa_q, air_t) in enumerate(zip(poa_quantiles, temp_air)):
            lower = max(0.0, 2 * poa_q[0] - poa_q[1])
            upper = min(max_poa, 2 * poa_q[-1] - poa_q[-2])
            poa_samples = np.interp(u, full_levels, np.r_[lower, poa_q, upper])
            power_quantiles[i, :] = np.quantile(
                to_power(poa_samples, air_t), _OUTPUT_LEVELS
            )

        for j, q in enumerate(_OUTPUT_LEVELS):
            q_key = round(float(q), 6)
            col = level_to_existing.get(q_key, f"{q:g}" if use_string_labels else float(q))
            out[col] = power_quantiles[:, j]
    # The point forecast is distinct from the median of the power distribution.
    if "prediction" in out.columns:
        out["prediction"] = to_power(out["prediction"].to_numpy(), temp_air)

    if "target" in out.columns:
        out["target"] = to_power(out["target"].to_numpy(), temp_air)

    return out.drop(columns=["_item_key", temperature_col])
