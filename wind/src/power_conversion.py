import numpy as np


def ws2wp_func(ws):
    """
    Convert wind speed to power output using interpolation of a power curve.

    Parameters
    ----------
    ws : array-like
        Wind speed input (can be scalar, list, or numpy array of any shape)

    Returns
    -------
    newPowerOutput : numpy.ndarray
        Power output with same shape as input
    """
    wind_speeds = np.array([
        0,1,2,3,4,5,6,7,8,9,
        10,11,12,13,14,15,16,17,18,19,
        20,21,22,23,24,25,26,27,28,29,
        30,31,32,33,34,35,36,37
    ])

    power_percentages = np.array([
        0.00,0.00,0.00,0.40,3.00,7.10,
        13.30,21.80,33.10,47.20,63.90,
        81.30,93.60,98.40,100.00,100.00,
        100.00,100.00,100.00,100.00,100.00,
        100.00,100.00,100.00,98.00,94.60,
        85.80,73.00,50.00,29.60,15.00,
        0.00,0.00,0.00,0.00,0.00,0.00,0.00
    ])

    ws_array = np.asarray(ws)
    ws_flat = ws_array.flatten()
    interpolated = np.interp(ws_flat, wind_speeds, power_percentages / 100)
    maximum_power = 1
    power_output_flat = interpolated * maximum_power
    newPowerOutput = power_output_flat.reshape(ws_array.shape)
    newPowerOutput = np.clip(newPowerOutput, 0, 15)
    return newPowerOutput


def convert_wind_result_to_power(result):
    """Convert target, median prediction, and available quantiles from wind speed to power."""
    out = result.copy()
    cols = [c for c in ["target", "prediction"] if c in out.columns]
    cols += [c for c in out.columns if _is_quantile_column(c)]
    for col in dict.fromkeys(cols):
        out[col] = ws2wp_func(out[col].to_numpy())
    return out


def _is_quantile_column(name):
    try:
        q = float(name)
    except (TypeError, ValueError):
        return False
    return 0 < q < 1
