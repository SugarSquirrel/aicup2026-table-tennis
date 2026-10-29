"""Paired bootstrap of private-score differences between milestones (paper Table V).

Usage: python paper_analysis/bootstrap_private.py
Needs test_answer_solution.csv in the repo root (released after the competition; gitignored).
Resamples the 609 private rallies (same indices for both submissions) B times.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "results"
B, SEED = 2000, 0

SUBS = {
    "v1": "1/submission_incl0.csv",
    "v2": "2/submission_v2-player_cv0.341_incl0.csv",
    "v3": "3/submission_v3-matchup_cvb0.350_incl0.csv",
    "v7": "5/submission_v7-aug_incl0.csv",
    "v10": "6/submission_v10-aug-ovr_incl0.csv",
    "v12": "7/submission_v12-aug-ovr_incl0.csv",
    "v14": "8/submission_v14-aug-ovr_incl0.csv",
}
# consecutive steps, then cumulative ones
PAIRS = [("v1", "v2"), ("v2", "v3"), ("v3", "v7"), ("v7", "v10"), ("v10", "v12"), ("v12", "v14"),
         ("v1", "v7"), ("v7", "v14"), ("v1", "v14"), ("v1", "v3")]

ans = pd.read_csv(ROOT / "test_answer_solution.csv")
ans = ans[ans.rally_uid >= 20000].reset_index(drop=True)
P = {k: ans[["rally_uid"]].merge(pd.read_csv(ROOT / "submission" / v), on="rally_uid") for k, v in SUBS.items()}
yA, yP, yR = ans.actionId.to_numpy(), ans.pointId.to_numpy(), ans.serverGetPoint.to_numpy()


def parts(sub, idx):
    a, p, r = yA[idx], yP[idx], yR[idx]
    fa = f1_score(a, sub.actionId.to_numpy()[idx], labels=np.unique(a), average="macro", zero_division=0)
    fp = f1_score(p, sub.pointId.to_numpy()[idx], labels=np.unique(p), average="macro", zero_division=0)
    auc = roc_auc_score(r, sub.serverGetPoint.to_numpy()[idx])
    return np.array([0.4 * fa + 0.4 * fp + 0.2 * auc, fa, fp, auc])


rng = np.random.default_rng(SEED)
n = len(ans)
idxs = [rng.integers(0, n, n) for _ in range(B)]
full = np.arange(n)

rows = []
print(f"{'pair':10s} {'metric':7s} {'delta':>8s} {'95% CI':>20s} {'P(delta>0)':>10s}")
for a, b in PAIRS:
    d0 = parts(P[b], full) - parts(P[a], full)
    ds = np.array([parts(P[b], i) - parts(P[a], i) for i in idxs])
    for j, name in enumerate(["score", "action_f1", "point_f1", "server_auc"]):
        lo, hi = np.percentile(ds[:, j], [2.5, 97.5])
        pos = np.mean(ds[:, j] > 0)
        rows.append(dict(pair=f"{a}->{b}", metric=name, delta=d0[j], ci_lo=lo, ci_hi=hi, p_pos=pos))
        print(f"{a + '->' + b:10s} {name[:7]:7s} {d0[j]:+8.4f} [{lo:+.4f}, {hi:+.4f}] {pos:10.3f}")

OUT.mkdir(exist_ok=True)
pd.DataFrame(rows).to_csv(OUT / "bootstrap_private.csv", index=False)
print(f"\nsaved {OUT / 'bootstrap_private.csv'}")
