# Titanic autoresearch log

| timestamp_utc | branch | model_name | feature_set | cv_mean | cv_std | baseline_cv_mean | delta_cv_mean | threshold | decision | notes |
|---|---|---|---|---:|---:|---:|---:|---:|---|---|

| 2026-03-15T10:12:46Z | unknown | random_forest_v1 | engineered | 0.83388 | 0.01421 |  |  | 0.00200 | no_baseline | No approved baseline found; current run becomes the initial baseline candidate. |
| 2026-03-15T10:13:17Z | unknown | random_forest_v1 | engineered | 0.83388 | 0.01421 |  |  | 0.00200 | no_baseline | No approved baseline found; current run becomes the initial baseline candidate. |
| 2026-03-15T10:23:02Z | unknown | random_forest_v1 | engineered | 0.83164 | 0.01338 | 0.83388 | -0.00223 | 0.00200 | rejected | Improvement -0.00223 did not meet threshold 0.00200. |
| 2026-03-15T10:32:04Z | unknown | random_forest_v1 | engineered | 0.84061 | 0.01176 | 0.83388 | 0.00674 | 0.00200 | accepted | Improved CV mean by 0.00674, meeting threshold 0.00200. |
