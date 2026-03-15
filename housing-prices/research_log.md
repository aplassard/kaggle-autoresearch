# Housing Prices autoresearch log

| timestamp_utc | branch | model_name | feature_set | cv_rmse_mean | cv_rmse_std | baseline_cv_rmse | delta_cv_rmse | threshold | decision | notes |
|---|---|---|---|---:|---:|---:|---:|---:|---|---|| 2026-03-15T16:00:41Z | unknown | ridge_v1 | baseline | 38309.10 | 11431.63 | 38381.28 | -72.18 | -50 | accepted | Reduced RMSE by 72.18, meeting threshold of 50.00 improvement. |
| 2026-03-15T16:02:45Z | unknown | ridge_v1 | baseline | 38317.96 | 11466.18 | 38309.10 | 8.86 | -50 | rejected | RMSE reduction -8.86 did not meet threshold of 50.00 improvement. |
