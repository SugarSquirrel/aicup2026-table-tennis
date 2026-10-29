"""Private rescued-stratum F1 split by the two kinds of private match ID (paper Sec. V-D).

Usage: python paper_analysis/private_strata_by_idtype.py
Needs data/train.csv, data/test_new.csv, data/Reference_Only_Old_Test_Data/test.csv and
test_answer_solution.csv in the repo root (all gitignored).

Why: most private rescued samples come from pooled IDs (match IDs 340-354), whose player pairings all occur in
the reference data, so a rescued-stratum gain could reflect pairing-specific information. This script reports
each stratum's macro-F1 separately for pooled IDs and the nine new matches (IDs 355-363).
Strata and metric follow private_strata.py exactly: next hitter = gamePlayerOtherId of the last observed
stroke; seen = in train, rescued = only in the reference data, cold = in neither; macro-F1 over the labels
present in the ground truth (zero_division=0).
Output: results/private_strata_by_idtype.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "results"

SUBS = {
    "v1": "1/submission_incl0.csv",
    "v2": "2/submission_v2-player_cv0.341_incl0.csv",
    "v3": "3/submission_v3-matchup_cvb0.350_incl0.csv",
    "v7": "5/submission_v7-aug_incl0.csv",
    "v10": "6/submission_v10-aug-ovr_incl0.csv",
    "v12": "7/submission_v12-aug-ovr_incl0.csv",
    "v14": "8/submission_v14-aug-ovr_incl0.csv",
}

tr = pd.read_csv(ROOT / "data/train.csv")
te = pd.read_csv(ROOT / "data/test_new.csv")
ref = pd.read_csv(ROOT / "data/Reference_Only_Old_Test_Data/test.csv")
ans = pd.read_csv(ROOT / "test_answer_solution.csv")

P_tr = set(tr.gamePlayerId) | set(tr.gamePlayerOtherId)
P_ref = set(ref.gamePlayerId) | set(ref.gamePlayerOtherId)
prv = te[~te.rally_uid.isin(set(ref.rally_uid))]
last = prv.sort_values("strikeNumber").groupby("rally_uid").tail(1).set_index("rally_uid")
nh = last.gamePlayerOtherId
stratum = pd.Series(np.where(nh.isin(P_tr), "seen", np.where(nh.isin(P_ref), "rescued", "cold")), index=nh.index)
match = prv.groupby("rally_uid").match.first().loc[stratum.index]
idtype = pd.Series(np.where(match.between(340, 354), "pooled IDs",
                            np.where(match.between(355, 363), "new matches", "other")), index=stratum.index)
print("rallies per (id type, stratum):")
print(pd.crosstab(idtype, stratum))

a = ans.set_index("rally_uid").loc[stratum.index]


def mf1(y, p):
    return f1_score(y, p, labels=np.unique(y), average="macro", zero_division=0)


rows = []
for k, path in SUBS.items():
    sub = pd.read_csv(ROOT / "submission" / path).set_index("rally_uid").loc[stratum.index]
    for g in ("pooled IDs", "new matches"):
        for s in ("seen", "rescued", "cold"):
            m = ((idtype == g) & (stratum == s)).to_numpy()
            if m.sum() == 0:
                continue
            rows.append(dict(version=k, id_type=g, stratum=s, n=int(m.sum()),
                             action_f1=mf1(a.actionId.to_numpy()[m], sub.actionId.to_numpy()[m]),
                             point_f1=mf1(a.pointId.to_numpy()[m], sub.pointId.to_numpy()[m])))
res = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(res.pivot_table(index="version", columns=["id_type", "stratum"], values="action_f1", sort=False)
      .reindex(list(SUBS)).round(3))
OUT.mkdir(exist_ok=True)
res.to_csv(OUT / "private_strata_by_idtype.csv", index=False)
print(f"saved {OUT / 'private_strata_by_idtype.csv'}")
