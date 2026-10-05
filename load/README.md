# Load

PG&E 2025 short-term load track. Notebooks reproduce the main rolling test result only: MAE, RMSE, and aQL.


### GEFCom evaluation
For the 12 evaluated GEFCom tasks, notebooks report raw aQL and competition-style normalized improvement ratio, defined as `1 - model_aQL / naive_aQL`. Full timestamp-level prediction CSVs are also saved under `outputs/predictions/gefcom/`.
