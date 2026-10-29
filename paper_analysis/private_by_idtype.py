"""Private score of every archived system, split by the two kinds of private match ID (paper v2, Sec. V-D).

Usage: python paper_analysis/private_by_idtype.py
Needs data/test_new.csv and test_answer_solution.csv in the repo root (gitignored). ActionOnly (v14g) is read
from the repo root file submission_v14g-aug-ovr_incl0.csv if present; otherwise it is skipped.

Why: private match IDs 340-354 ("pooled IDs") each pool rallies of several player pairings that also occur in
the reference data, whereas IDs 355-363 are nine new matches. This script scores each system on the two groups
separately. Metric code follows bootstrap_private.py (macro-F1 over labels present in the ground truth,
score = 0.4*F1_action + 0.4*F1_point + 0.2*AUC). Output: results/private_by_idtype.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "results"

SUBS = {
    "v1": "submission/1/submission_incl0.csv",
    "v2": "submission/2/submission_v2-player_cv0.341_incl0.csv",
    "v3": "submission/3/submission_v3-matchup_cvb0.350_incl0.csv",
    "v7": "submission/5/submission_v7-aug_incl0.csv",
    "v10": "submission/6/submission_v10-aug-ovr_incl0.csv",
    "v12": "submission/7/submission_v12-aug-ovr_incl0.csv",
    "v14": "submission/8/submission_v14-aug-ovr_incl0.csv",
    "v14g": "submission_v14g-aug-ovr_incl0.csv",
}
NAMES = {"v1": "Base", "v2": "Player", "v3": "Matchup", "v7": "Core", "v10": "Blend", "v12": "Filter",
         "v14": "Submitted", "v14g": "ActionOnly"}

te = pd.read_csv(ROOT / "data/test_new.csv")
ans = pd.read_csv(ROOT / "test_answer_solution.csv").set_index("rally_uid")
ans = ans[ans.index >= 20000].sort_index()
match = te.groupby("rally_uid").match.first().loc[ans.index]
group = np.where(match.between(340, 354), "pooled IDs",
                 np.where(match.between(355, 363), "new matches", "other"))
print("private match IDs:", sorted(match.unique()))
print("rallies per group:", pd.Series(group).value_counts().to_dict())


def mf1(y, p):
    return f1_score(y, p, labels=np.unique(y), average="macro", zero_division=0)


rows = []
for k, path in SUBS.items():
    p = ROOT / path
    if not p.exists():
        print(f"[skip] {k}: {path} not found")
        continue
    sub = pd.read_csv(p).set_index("rally_uid").loc[ans.index]
    for g in ("pooled IDs", "new matches", "all private"):
        m = np.ones(len(ans), bool) if g == "all private" else group == g
        fa = mf1(ans.actionId.to_numpy()[m], sub.actionId.to_numpy()[m])
        fp = mf1(ans.pointId.to_numpy()[m], sub.pointId.to_numpy()[m])
        auc = roc_auc_score(ans.serverGetPoint.to_numpy()[m], sub.serverGetPoint.to_numpy()[m])
        rows.append(dict(version=k, system=NAMES[k], group=g, n=int(m.sum()),
                         score=0.4 * fa + 0.4 * fp + 0.2 * auc, action_f1=fa, point_f1=fp, server_auc=auc))

res = pd.DataFrame(rows)
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)
print(res.pivot_table(index=["version", "system"], columns="group",
                      values=["score", "action_f1", "point_f1", "server_auc"], sort=False).round(3))
OUT.mkdir(exist_ok=True)
res.to_csv(OUT / "private_by_idtype.csv", index=False)
print(f"\nsaved {OUT / 'private_by_idtype.csv'}")
