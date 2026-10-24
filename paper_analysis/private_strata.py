"""Per-stratum private scores of every archived version (paper Sec. V-D).

Usage: python paper_analysis/private_strata.py
Strata follow the next hitter of each private rally (opponent of the last observed stroke):
seen = in train, rescued = only in the reference data, cold = in neither.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

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

tr = pd.read_csv(ROOT / "data/train.csv")
te = pd.read_csv(ROOT / "data/test_new.csv")
ref = pd.read_csv(ROOT / "data/Reference_Only_Old_Test_Data/test.csv")
ans = pd.read_csv(ROOT / "test_answer_solution.csv")

P_tr = set(tr.gamePlayerId) | set(tr.gamePlayerOtherId)
P_ref = set(ref.gamePlayerId) | set(ref.gamePlayerOtherId)
prv = te[~te.rally_uid.isin(set(ref.rally_uid))]
nh = prv.sort_values("strikeNumber").groupby("rally_uid").tail(1).set_index("rally_uid").gamePlayerOtherId
stratum = pd.Series(np.where(nh.isin(P_tr), "seen", np.where(nh.isin(P_ref), "rescued", "cold")), index=nh.index)

a = ans.set_index("rally_uid").loc[stratum.index]
S = stratum.to_numpy()
print("private strata sizes:", stratum.value_counts().to_dict())


def mf1(y, p):
    return f1_score(y, p, labels=np.unique(y), average="macro", zero_division=0)


rows = []
for k, path in SUBS.items():
    sub = pd.read_csv(ROOT / "submission" / path).set_index("rally_uid").loc[stratum.index]
    for s in ("seen", "rescued", "cold"):
        m = S == s
        rows.append(dict(version=k, stratum=s, n=int(m.sum()),
                         action_f1=mf1(a.actionId.to_numpy()[m], sub.actionId.to_numpy()[m]),
                         point_f1=mf1(a.pointId.to_numpy()[m], sub.pointId.to_numpy()[m])))
res = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(res.pivot_table(index="version", columns="stratum", values=["action_f1", "point_f1"], sort=False)
      .reindex(list(SUBS)).round(3))

# paired bootstrap within each stratum for the steps the paper discusses
rng = np.random.default_rng(SEED)
P = {k: pd.read_csv(ROOT / "submission" / v).set_index("rally_uid").loc[stratum.index] for k, v in SUBS.items()}
print(f"\npaired bootstrap within stratum ({B} resamples): delta [95% CI]")
brows = []
for a_, b_ in [("v3", "v7"), ("v7", "v10"), ("v12", "v14"), ("v7", "v14")]:
    for s in ("seen", "rescued", "cold"):
        idx0 = np.flatnonzero(S == s)
        for col, name in [("actionId", "action_f1"), ("pointId", "point_f1")]:
            y = a[col].to_numpy()
            pa, pb = P[a_][col].to_numpy(), P[b_][col].to_numpy()
            d0 = mf1(y[idx0], pb[idx0]) - mf1(y[idx0], pa[idx0])
            ds = []
            for _ in range(B):
                i = rng.choice(idx0, len(idx0))
                ds.append(mf1(y[i], pb[i]) - mf1(y[i], pa[i]))
            lo, hi = np.percentile(ds, [2.5, 97.5])
            brows.append(dict(pair=f"{a_}->{b_}", stratum=s, metric=name, delta=d0, ci_lo=lo, ci_hi=hi))
            print(f"{a_ + '->' + b_:8s} {s:8s} {name:9s} {d0:+.3f} [{lo:+.3f}, {hi:+.3f}]")

OUT.mkdir(exist_ok=True)
res.to_csv(OUT / "private_strata.csv", index=False)
pd.DataFrame(brows).to_csv(OUT / "private_strata_bootstrap.csv", index=False)
print(f"\nsaved {OUT / 'private_strata.csv'} and private_strata_bootstrap.csv")
