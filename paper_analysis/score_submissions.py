"""Rescore every archived submission against the official test answers (paper Table IV).

Usage: python paper_analysis/score_submissions.py
Needs test_answer_solution.csv in the repo root (released after the competition; gitignored).

Score = 0.4*macroF1(action) + 0.4*macroF1(point) + 0.2*AUC(server).
Two macro-F1 variants are reported to confirm which one reproduces the official LB:
  'true'  = average over classes present in ground truth
  'union' = sklearn default (classes present in truth OR prediction)
Sanity check: submission/8 must give 0.434101 public / 0.389023 private (official 0.4341008 / 0.3890227).
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "results"
PUBLIC_MAX_UID = 19999  # public = 1236 reference rallies (uid <= 19404); private = 609 rallies (uid >= 20000)


def f1(y, p, mode):
    if mode == "true":
        return f1_score(y, p, labels=np.unique(y), average="macro", zero_division=0)
    return f1_score(y, p, average="macro", zero_division=0)


def score(df, mode):
    fa = f1(df.actionId_t, df.actionId_p, mode)
    fp = f1(df.pointId_t, df.pointId_p, mode)
    auc = roc_auc_score(df.serverGetPoint_t, df.serverGetPoint_p)
    return 0.4 * fa + 0.4 * fp + 0.2 * auc, fa, fp, auc


ans = pd.read_csv(ROOT / "test_answer_solution.csv")
rows = []
for csv in sorted(ROOT.glob("submission/*/*.csv")):
    sub = pd.read_csv(csv)
    m = ans.merge(sub, on="rally_uid", suffixes=("_t", "_p"), validate="one_to_one")
    assert len(m) == len(ans) == len(sub), (csv, len(m), len(sub))
    for split, part in [("public", m[m.rally_uid <= PUBLIC_MAX_UID]),
                        ("private", m[m.rally_uid > PUBLIC_MAX_UID])]:
        for mode in ("true", "union"):
            s, fa, fp, auc = score(part, mode)
            rows.append(dict(sub=csv.parent.name + "/" + csv.stem, split=split, f1mode=mode,
                             n=len(part), score=s, action_f1=fa, point_f1=fp, server_auc=auc))

out = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(out.pivot_table(index="sub", columns=["split", "f1mode"], values="score").round(7))
print()
print(out[out.f1mode == "true"].round(4).to_string(index=False))

pub, prv = ans[ans.rally_uid <= PUBLIC_MAX_UID], ans[ans.rally_uid > PUBLIC_MAX_UID]
print("\nserverGetPoint base rate public/private:", round(pub.serverGetPoint.mean(), 3), round(prv.serverGetPoint.mean(), 3))
print("pointId=0 (rally-ending) share public/private:", round((pub.pointId == 0).mean(), 3), round((prv.pointId == 0).mean(), 3))

OUT.mkdir(exist_ok=True)
out.to_csv(OUT / "scores.csv", index=False)
print(f"\nsaved {OUT / 'scores.csv'}")
