# Solar forecasting track

reproduce main result in solar forecasting under MAE, RMSE, aQL

Dataset file: `solar/data/solardata_modified.csv`.

The CSV already contains both the original observed covariates and the precomputed lag-24 proxy columns.

Historical context uses the measured covariates:

- `AIRTEMP`
- `RH_AVG`
- `DEWPT`
- `WS`
- `GHI`
- `DNI`
- `DIFF`

Future Proxies:

- `AIRTEMP_lag24`
- `RH_AVG_lag24`
- `DEWPT_lag24`
- `WS_lag24`
- `GHI_lag24`
- `DNI_lag24`
- `DIFF_lag24`


