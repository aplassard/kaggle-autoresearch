You are improving a single-model Kaggle Titanic pipeline through controlled experiments.

## Goal

Improve the approved baseline using small, reversible changes.

The workflow is:

1. Read `approved_run.json`, `research_log.md`, `features.py`, and `models.py`.
2. Propose 3 possible experiments.
3. Choose 1 clear hypothesis.
4. Create a branch for that experiment.
5. Modify only model or feature code.
6. Run `uv run python run.py`.
7. Read the result from `artifacts/current_run.json` and `research_log.md`.
8. If improvement is above threshold, keep the model change and update `approved_run.json`.
9. If improvement is below threshold, revert model changes and keep the log entry.
10. Commit the result.
11. Merge only accepted experiments.

## Files you may change

You may edit:
- `features.py`
- `models.py`

You may preserve or commit:
- `research_log.md`

You must not modify:
- `run.py`

Treat `run.py` as the fixed evaluation harness.

## Rules

- Make one coherent improvement at a time.
- Do not mix unrelated ideas in one branch.
- Do not change evaluation logic.
- Do not change threshold logic.
- Do not change artifact names or output format.
- Do not merge rejected experiments.
- Do not delete failed experiment history.

## Run command

Always evaluate with:

```bash
uv run python run.py
```
Do not use another script or notebook as the source of truth.

Acceptance policy
Trust the harness decision.

If accepted: keep code changes, update approved_run.json, commit, and merge.

If rejected: revert features.py and models.py, keep research_log.md, commit the log, and do not merge.

Branch naming
Use descriptive branch names such as:

exp/title-age-imputation

exp/sex-pclass-interaction

exp/rf-leaf-tuning

Commit guidance
Accepted experiment:

Commit features.py and/or models.py

Commit approved_run.json

Commit research_log.md

Rejected experiment:

Revert features.py and models.py

Commit only research_log.md

Experiment summary
After each run, report:

Hypothesis

Files changed

CV before

CV after

Delta

Decision

Reason

Next experiment

Default behavior
Operate in this mode by default:

one branch

one hypothesis

one change

one run

one decision

Prefer small, reversible improvements over broad rewrites.
