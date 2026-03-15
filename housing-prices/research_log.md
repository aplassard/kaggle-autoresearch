# Housing Prices autoresearch log

| timestamp_utc | branch | model_name | feature_set | cv_rmse_mean | cv_rmse_std | baseline_cv_rmse | delta_cv_rmse | threshold | decision | hypothesis | notes |
|---|---|---|---|---:|---:|---:|---:|---:|---|---|| 2026-03-15T16:05:49Z | unknown | ridge_v1 | baseline | 38309.10 | 11431.63 |  |  | -50 | no_baseline | Increasing Ridge alpha from 1.0 to 10.0 will reduce overfitting and improve cross-validation RMSE by applying stronger regularization | No approved baseline found; current run becomes the initial baseline candidate. |
| 2026-03-15T16:09:03Z | unknown | ridge_v1 | baseline | 37900.13 | 11311.28 | 38309.10 | -408.97 | -50 | accepted | Reduced RMSE by 408.97, meeting threshold of 50.00 improvement. |
