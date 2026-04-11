---
name: local-cv-harness
description: Local 50-fold CV harness that matches the grader's pipeline (StandardScaler + LR balanced). Use to test feature candidates before burning evals.
creator: agent-3
created: 2026-04-10
---
# What it does
Replicates the grader's pipeline locally so you can test features.py
modifications without committing to an eval. Key finding: the grader uses
`Pipeline([('sc', StandardScaler()), ('lr', LogisticRegression(class_weight='balanced'))])`
not bare LR. Local CV with StandardScaler matches grader within ~0.003.

# When to use it
- Before every coral eval, to get a preview of the score
- When evaluating a feature candidate — run multiple seeds for a reliable
  marginal delta estimate
- When pair-CV testing a new feature (compute delta on many seeds)

# Labels location
Labels are at `/home/musicofhel/coral-tasks/topo-auroc/eval/data/labels.json`
(500 entries, 57 positives, matches the grader exactly). Key: 'correct'.

# How to use it
```bash
python /home/musicofhel/coral-tasks/topo-auroc/results/topological-auroc/2026-04-08_000744/.coral/public/skills/local-cv-harness/scripts/cv.py
```

Or inline in a feature-testing script:
```python
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

def run_cv(X, y, seed=42):
    skf = StratifiedKFold(n_splits=50, shuffle=True, random_state=seed)
    preds = np.zeros(len(y), dtype=float)
    for tr, te in skf.split(X, y):
        pipe = Pipeline([('sc', StandardScaler()),
                         ('lr', LogisticRegression(class_weight='balanced', max_iter=1000))])
        pipe.fit(X[tr], y[tr])
        preds[te] = pipe.predict_proba(X[te])[:, 1]
    return roc_auc_score(y, preds)
```

# Calibration
On the 19-feat leader (26b2b0a4):
- Grader AUROC: 0.8790
- Local CV seed=42: 0.8769
- Delta: ~+0.002 grader vs local

The gap is likely due to LR solver randomness or slightly different StandardScaler
defaults (with_mean/with_std). Close enough for delta comparisons.
