# Titanic Pipeline Autoresearch Agent

You are improving a single-model Kaggle Titanic pipeline through controlled experiments.

## Goal

Improve the approved baseline using small, reversible changes.

## Workflow

1. **Read context**: `approved_run.json`, `research_log.md`, `features.py`, and `models.py`
2. **Choose hypothesis**: Select 1 clear, testable hypothesis
3. **Create branch**: Use descriptive branch name (e.g., `exp/feature-name`)
4. **Modify code**: Edit only `features.py` or `models.py`
5. **Run experiment**: `uv run python run.py`
6. **Read results**: Check `artifacts/current_run.json` and `research_log.md`
7. **Evaluate outcome**: 
   - If accepted: Keep changes, update `approved_run.json`, commit, merge to main
   - If rejected: Revert code changes, keep log entry, commit log only
8. **Report summary**: Provide experiment summary and STOP

## Critical Constraint

**Run exactly ONE experiment per invocation, regardless of outcome.**

Do not iterate, retry, or run additional experiments. Complete the full workflow for a single experiment and stop.

## Files You May Change

| File | Permission |
|------|------------|
| `features.py` | Edit |
| `models.py` | Edit |
| `research_log.md` | Preserve/commit |
| `approved_run.json` | Update if accepted |

**Do NOT modify**: `run.py` (treat as fixed evaluation harness)

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

- **Accepted**: Keep code changes, update `approved_run.json`, commit all changes, merge to main
- **Rejected**: Revert `features.py` and `models.py`, commit only `research_log.md`, do NOT merge

## Branch Naming

Use descriptive names with `exp/` prefix:

- `exp/title-age-imputation`
- `exp/sex-pclass-interaction`
- `exp/rf-hyperparameter-tuning`

## Commit Guidance

### Accepted Experiment
Commit: `features.py`, `models.py`, `approved_run.json`, `research_log.md`

### Rejected Experiment
Revert: `features.py`, `models.py`
Commit: `research_log.md` only

## Experiment Summary

After each run, report:

| Field | Description |
|-------|-------------|
| Hypothesis | What you're testing |
| Files changed | Which files were modified |
| CV before | Baseline CV score |
| CV after | New CV score |
| Delta | Change in CV score |
| Decision | accepted/rejected |
| Reason | Why the decision was made |

## Summary

**One branch. One hypothesis. One change. One run. One decision. STOP.**
