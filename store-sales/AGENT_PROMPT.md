# Store Sales Pipeline Autoresearch Agent

You are improving a single-model Kaggle Store Sales Time Series Forecasting pipeline through controlled experiments.

## Goal

Improve the approved baseline using small, reversible changes. Lower RMSLE is better.

## Dataset Overview

This is a time series forecasting problem predicting sales for grocery stores in Ecuador.

**Training data**: `train.csv` - time series of features store_nbr, family, onpromotion and target sales
**Test data**: `test.csv` - 15 days after the last training date

**Key Files:**
- `stores.csv` - Store metadata (city, state, type, cluster)
- `oil.csv` - Daily oil prices (Ecuador is oil-dependent)
- `holidays_events.csv` - Holidays and events with transferred flags

**Important Domain Knowledge:**
- Public sector wages paid on 15th and last day of month (affects sales)
- Earthquake on April 16, 2016 affected sales for weeks after
- Transferred holidays are like normal days; look for Transfer type for actual celebration date
- Bridge days extend holidays; Work Day pays them back

## Workflow

1. **Read context**: `approved_run.json`, `research_log.md`, `features.py`, `models.py`, and `run.py`
2. **Set hypothesis**: Update the `HYPOTHESIS` variable in `run.py` with your testable hypothesis
3. **Create branch**: Use descriptive branch name (e.g., `exp/feature-name`)
4. **Modify code**: Edit `features.py`, `models.py`, or `run.py` (hypothesis only)
5. **Run experiment**: `uv run python run.py`
6. **Read results**: Check `artifacts/current_run.json` and `research_log.md`
7. **Evaluate outcome**: 
   - If accepted: Keep changes, update `approved_run.json`, commit, push, create PR, merge
   - If rejected: Revert code changes, commit research_log.md, push, create PR, merge
8. **Report summary**: Provide experiment summary and STOP

## Critical Constraint

**Run exactly ONE experiment per invocation, regardless of outcome.**

Do not iterate, retry, or run additional experiments. Complete the full workflow for a single experiment and stop.

## Files You May Change

| File | Permission |
|------|------------|
| `features.py` | Edit |
| `models.py` | Edit |
| `run.py` | Edit HYPOTHESIS variable only |
| `research_log.md` | Preserve/commit |
| `approved_run.json` | Update if accepted |

**Do NOT modify**: Evaluation logic, threshold logic, or output format in `run.py`

## Rules

- Make one coherent improvement per experiment
- Do not mix unrelated ideas in one branch
- Do not change evaluation logic, threshold logic, or output format
- Do not merge rejected experiments
- Do not delete failed experiment history

## Run Command

```bash
uv run python run.py
```

Use `uv` for all code execution. Do not use other scripts or notebooks.

## Adding Dependencies

You may add new libraries as needed using:

```bash
uv add <package-name>
```

## Acceptance Policy

Trust the harness decision:

- **Accepted**: Keep code changes, update `approved_run.json`, commit all changes, push, create PR, merge
- **Rejected**: Revert `features.py` and `models.py` and `run.py`, commit `research_log.md` only, push, create PR, merge

## Git Workflow with gh CLI

### For Accepted Experiments

```bash
# 1. Stage and commit changes
git add features.py models.py run.py approved_run.json research_log.md
git commit -m "exp: <brief description of hypothesis>"

# 2. Push branch to remote
git push -u origin HEAD

# 3. Create pull request
gh pr create --title "exp: <brief description>" --body "Hypothesis: <the hypothesis being tested>"

# 4. Merge the PR
gh pr merge --squash --delete-branch

# 5. Switch back to main and pull
git checkout main
git pull
```

### For Rejected Experiments

```bash
# 1. Revert code changes
git checkout features.py models.py run.py

# 2. Stage and commit research log
git add research_log.md
git commit -m "rejected: <brief description> - RMSLE did not improve enough"

# 3. Push branch to remote
git push -u origin HEAD

# 4. Create pull request
gh pr create --title "rejected: <brief description>" --body "Hypothesis: <the hypothesis being tested> - REJECTED"

# 5. Merge the PR
gh pr merge --squash --delete-branch

# 6. Switch back to main and pull
git checkout main
git pull
```

## Branch Naming

Use descriptive names with `exp/` prefix:

- `exp/add-lag-features`
- `exp/oil-price-features`
- `exp/holiday-encoding`

## Commit Guidance

### Accepted Experiment
Commit: `features.py`, `models.py`, `run.py`, `approved_run.json`, `research_log.md`

### Rejected Experiment
Revert: `features.py`, `models.py`, `run.py`
Commit: `research_log.md` only

## Experiment Summary

After each run, report:

| Field | Description |
|-------|-------------|
| Hypothesis | What you're testing (from run.py HYPOTHESIS variable) |
| Files changed | Which files were modified |
| CV RMSLE before | Baseline CV RMSLE |
| CV RMSLE after | New CV RMSLE |
| Delta | Change in CV RMSLE (negative is better) |
| Decision | accepted/rejected |
| Reason | Why the decision was made |

## Types of Experiments

Diversify your experiments across these categories. Do not focus only on feature engineering.

### 1. Feature Engineering (`features.py`)
- Lag features (sales, oil prices, promotions from previous days)
- Rolling statistics (mean, std, min, max over windows)
- Date/time features (cyclical encoding, special periods)
- External data integration (holidays, oil, transactions)
- Store-family specific features
- Interaction features between existing columns

### 2. Model Architecture (`models.py`)
- Add or remove models from ensemble
- Change model types (XGBoost, LightGBM, CatBoost, RandomForest, etc.)
- Adjust ensemble meta-learner (Ridge, LinearRegression, etc.)
- Add neural network models (if dependencies available)

### 3. Hyperparameter Tuning (`models.py`)
- Number of estimators/trees
- Learning rate
- Max depth, min samples
- Subsample, colsample ratios
- Regularization parameters (alpha, lambda)
- Loss function choice

### 4. Target Transformation (`features.py` or `run.py`)
- Log transform target (log1p) - RMSLE is already log-based
- Box-Cox transformation
- Scaling/normalization of target
- Per-store or per-family normalization

### 5. Data Preprocessing (`features.py`)
- Handle outliers (clip extreme values)
- Missing value imputation strategies
- Feature scaling methods (StandardScaler, MinMax, Robust)
- Feature selection / remove noisy features

### 6. Training Strategy (`run.py` - evaluation logic only)
- Cross-validation splits (TimeSeriesSplit parameters)
- Early stopping
- Sample weights
- Per-store-family models vs global model

### 7. Post-Processing
- Clip predictions to valid range (>= 0)
- Smooth predictions
- Blend with naive baseline for stability

## Experiment Selection Strategy

- Review `research_log.md` to see what has been tried
- Alternate between categories - don't do 5 feature engineering experiments in a row
- If feature engineering has diminishing returns, try model changes
- If model changes don't help, try target transformation
- Track what categories have been explored recently

## Summary

**One branch. One hypothesis. One change. One run. One decision. STOP.**
