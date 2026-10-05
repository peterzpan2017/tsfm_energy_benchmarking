# tsfm_energy_benchmarking


Compact paper-reproduction notebooks for **price, load, wind, and solar**. 

- Shared model environments: `requirements/`
- Shared evaluation: `src/metrics.py` (MAE, RMSE, aQL over 0.1–0.9)
- Track-specific data preparation: `<track>/src/preprocessing.py`
- Model notebooks: `<track>/notebooks/{tsfm,darts}/`

## Wind/solar power conversion

Wind and solar notebooks retain their original wind-speed/POA outputs and also write a second `*_power.csv` file, with conversion functions found in `wind/src/power_conversion.py` and `solar/src/power_conversion.py`. Power MAE, RMSE, and aQL are printed after the original speed/irradiance metrics.


## GEFCom sections
Price and load notebooks contain two dataset sections. GEFCom sections use the supplied cleaned CSVs directly, save full prediction CSVs, and report raw aQL and normalized improvement ratio for the final 12 evaluation tasks. 


### GEFCom evaluation
For the 12 evaluated GEFCom tasks, notebooks report raw aQL and competition-style normalized improvement ratio, defined as `1 - model_aQL / naive_aQL`. Full timestamp-level prediction CSVs are also saved under `outputs/predictions/gefcom/`.

## Quantile output convention

GEFCom2014 price/load sections keep the nine benchmark quantiles `0.1,...,0.9` for raw and normalized aQL evaluation.

For the non-GEFCom datasets, prediction CSVs use a common 15-quantile schema:
`0.01, 0.025, 0.05, 0.1, ..., 0.9, 0.95, 0.975, 0.99`.
Models only generate quantiles they natively support. Unsupported levels are saved as `NaN`. In particular, Chronos 2 omits 0.025/0.975 and TimesFM 2.5 uses only the nine quantiles.

## Darts checkpoint controls for price/load

The deep-learning Darts notebooks for price and load now keep independent controls and checkpoint folders for each dataset section:
- `GEF_LOAD_EXISTING` with `models/darts/gefcom/`
- `NON_GEF_LOAD_EXISTING` with `models/darts/gridstatus/` (price) or `models/darts/pg&e/` (load)
