# Housing Prices Pipeline Autoresearch Agent

You are improving a single-model Kaggle Housing Prices pipeline through controlled experiments.

## Goal

Improve the approved baseline using small, reversible changes. Lower RMSE is better.

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
git commit -m "rejected: <brief description> - RMSE did not improve enough"

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

- `exp/add-neighborhood-encoding`
- `exp/log-transform-target`
- `exp/ridge-alpha-tuning`

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
| CV RMSE before | Baseline CV RMSE |
| CV RMSE after | New CV RMSE |
| Delta | Change in CV RMSE (negative is better) |
| Decision | accepted/rejected |
| Reason | Why the decision was made |

## Summary

**One branch. One hypothesis. One change. One run. One decision. STOP.**
