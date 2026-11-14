import numpy as np


# Quantile probabilities used by the forecasting models and downstream evaluation.
_INPUT_LEVELS = np.round(np.arange(0.1, 1.0, 0.1), 2)
_OUTPUT_LEVELS = np.array([
    0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5,
    0.6, 0.7, 0.8, 0.9, 0.95, 0.975, 0.99,
])


def ws2wp_func(ws):
    """Convert wind speed to power output using the existing turbine power curve."""
    wind_speeds = np.array([
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
        10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
        20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
        30, 31, 32, 33, 34, 35, 36, 37,
    ])
    power_percentages = np.array([
        0.00, 0.00, 0.00, 0.40, 3.00, 7.10,
        13.30, 21.80, 33.10, 47.20, 63.90,
        81.30, 93.60, 98.40, 100.00, 100.00,
        100.00, 100.00, 100.00, 100.00, 100.00,
        100.00, 100.00, 100.00, 98.00, 94.60,
        85.80, 73.00, 50.00, 29.60, 15.00,
        0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
    ])
    ws_array = np.asarray(ws)
    interpolated = np.interp(ws_array.ravel(), wind_speeds, power_percentages / 100)
    power = interpolated.reshape(ws_array.shape)
    return np.clip(power, 0, 15)


def _is_quantile_column(name):
    try:
        q = float(name)
    except (TypeError, ValueError):
        return False
    return 0 < q < 1


def _input_quantile_columns(df):
    """Find the nine input quantiles regardless of string/float column labels."""
    available = {round(float(c), 6): c for c in df.columns if _is_quantile_column(c)}
    present = [round(float(q), 6) in available for q in _INPUT_LEVELS]
    if not any(present):
        return None  # Keep compatibility with point-forecast-only notebooks.
    if not all(present):
        raise ValueError("Distribution-based wind conversion requires all nine quantiles (0.1 to 0.9).")
    return [available[round(float(q), 6)] for q in _INPUT_LEVELS]


def convert_wind_result_to_power(result):
    """Convert wind forecasts to power .

    With nine wind-speed quantiles, reconstruct the speed quantile function
    using linearly extrapolated, physically bounded tails; map 1,000
    deterministic midpoint-probability samples through the *nonmonotone*
    turbine power curve; and compute 15 power quantiles. Original quantile
    column labels (float or string) are preserved, and new tail columns use
    the same label convention. `prediction` is converted directly from the point wind-speed forecast;
    the 0.5 power quantile is obtained from the reconstructed distribution.

    """
    out = result.copy()
    cols = _input_quantile_columns(out)

    if cols is not None:
        n_samples = 1000
        max_ws = 37.0
        full_levels = np.r_[0.0, _INPUT_LEVELS, 1.0]
        u = (np.arange(n_samples) + 0.5) / n_samples
        speed_quantiles = out[cols].to_numpy(dtype=float)
        if not np.all(np.isfinite(speed_quantiles)):
            raise ValueError("Wind-speed quantiles contain NaN or Inf.")
        if np.any(np.diff(speed_quantiles, axis=1) < 0):
            raise ValueError("Wind-speed quantiles must be nondecreasing.")
        if np.any(speed_quantiles[:, 0] < 0) or np.any(speed_quantiles[:, -1] > max_ws):
            raise ValueError("Wind-speed quantiles outside the [0, 37] m/s support.")

        use_string_labels = isinstance(cols[0], str)
        level_to_existing = {round(float(c), 6): c for c in out.columns if _is_quantile_column(c)}
        power_quantiles = np.empty((len(out), len(_OUTPUT_LEVELS)))
        for i, speed_q in enumerate(speed_quantiles):
            lower = max(0.0, 2 * speed_q[0] - speed_q[1])
            upper = min(max_ws, 2 * speed_q[-1] - speed_q[-2])
            speed_samples = np.interp(u, full_levels, np.r_[lower, speed_q, upper])
            power_quantiles[i, :] = np.quantile(ws2wp_func(speed_samples), _OUTPUT_LEVELS)

        for j, q in enumerate(_OUTPUT_LEVELS):
            q_key = round(float(q), 6)
            col = level_to_existing.get(q_key, f"{q:g}" if use_string_labels else float(q))
            out[col] = power_quantiles[:, j]

    # The point forecast is distinct from the median of the power distribution.
    if "prediction" in out.columns:
        out["prediction"] = ws2wp_func(out["prediction"].to_numpy())

    if "target" in out.columns:
        out["target"] = ws2wp_func(out["target"].to_numpy())
    return out
