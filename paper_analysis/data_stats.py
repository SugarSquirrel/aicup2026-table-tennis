"""Dataset statistics and truncation analysis (paper Table I, Sec. III, Sec. V-C).

Usage: python paper_analysis/data_stats.py
Needs data/ (train.csv, test_new.csv, Reference_Only_Old_Test_Data/test.csv) and
test_answer_solution.csv in the repo root.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
tr = pd.read_csv(ROOT / "data/train.csv")
te = pd.read_csv(ROOT / "data/test_new.csv")
ref = pd.read_csv(ROOT / "data/Reference_Only_Old_Test_Data/test.csv")
ans = pd.read_csv(ROOT / "test_answer_solution.csv")

pub_uids = set(ref.rally_uid)
pub = te[te.rally_uid.isin(pub_uids)]
prv = te[~te.rally_uid.isin(pub_uids)]


def players(df):
    return set(df.gamePlayerId) | set(df.gamePlayerOtherId)


def last_strokes(df):
    return df.sort_values("strikeNumber").groupby("rally_uid").tail(1).set_index("rally_uid")


# ---- Table I ----
print("== Table I")
for name, df in [("train", tr), ("reference", ref), ("test", te), ("  public", pub), ("  private", prv)]:
    print(f"{name:10s} matches={df.match.nunique():4d} rallies={df.rally_uid.nunique():6,d} "
          f"strokes={len(df):7,d} players={len(players(df)):4d}")

# ---- consistency checks behind Sec. III claims ----
print("\n== Split structure")
print("reference rallies all in test:", pub_uids <= set(te.rally_uid))
cols = ["rally_uid", "strikeNumber", "actionId", "pointId"]
m = ref[cols].merge(pub[cols], on=["rally_uid", "strikeNumber"], how="outer", indicator=True, suffixes=("_r", "_t"))
same = (m._merge == "both") & (m.actionId_r == m.actionId_t) & (m.pointId_r == m.pointId_t)
print(f"reference prefixes identical to public test part: {same.all()} ({same.sum()}/{len(m)} strokes)")
print("private matches:", sorted(prv.match.unique())[:3], "...", prv.match.nunique())
print("private matches in train/ref:", len(set(prv.match) & (set(tr.match) | set(ref.match))))

P_tr, P_ref = players(tr), players(ref)
print(f"\ntest players {len(players(te))}, unseen in train {len(players(te) - P_tr)}")
print(f"private players {len(players(prv))}: in train {len(players(prv) & P_tr)}, in reference {len(players(prv) & P_ref)}")

# next hitter = opponent of the last observed stroke (players alternate)
print("\n== Next-hitter strata (seen = in train, rescued = reference only, cold = neither)")
for name, df in [("public", pub), ("private", prv)]:
    nh = last_strokes(df).gamePlayerOtherId
    seen = nh.isin(P_tr)
    resc = ~seen & nh.isin(P_ref)
    print(f"{name:8s} n={len(nh)} seen={seen.mean():.3f} rescued={resc.mean():.3f} cold={(~seen & ~resc).mean():.3f}")

# ---- truncation ----
print("\n== Prefix lengths")
L_tr = tr.groupby("rally_uid").size()
print(f"train full rallies: median={L_tr.median():.0f} mean={L_tr.mean():.2f}")
for name, df in [("test", te), ("public", pub), ("private", prv)]:
    L = df.groupby("rally_uid").size()
    print(f"{name:8s} prefixes: median={L.median():.0f} mean={L.mean():.2f} "
          f"len1={int((L == 1).sum())} ({(L == 1).mean():.1%})")

# is pointId 0 the rally-ending stroke? (train)
last = tr.sort_values("strikeNumber").groupby("rally_uid").tail(1)
nonlast = tr.drop(last.index)
print(f"\ntrain: P(pointId=0 | last stroke)={(last.pointId == 0).mean():.3f}  "
      f"P(pointId=0 | not last)={(nonlast.pointId == 0).mean():.3f}")
even = tr.groupby("rally_uid").size() % 2 == 0
sgp = tr.groupby("rally_uid").serverGetPoint.first()
print(f"train: P(server wins | even length)={sgp[even].mean():.3f}  P(server wins | odd)={sgp[~even].mean():.3f}")

# truncation mechanism on the test answers: if the target stroke is rally-ending (pointId 0),
# the rally length is prefix+1, so its parity fixes serverGetPoint.
print("\n== Truncation mechanism (test answers)")
L_te = te.groupby("rally_uid").size()
a = ans.set_index("rally_uid").join(L_te.rename("L"))
a["part"] = np.where(a.index.isin(pub_uids), "public", "private")
a["final_target"] = a.pointId == 0
a["parity_pred"] = ((a.L + 1) % 2 == 0).astype(int)   # even total length -> server wins
for part, g in a.groupby("part"):
    f = g[g.final_target]
    print(f"{part:8s} target is rally-ending: {g.final_target.mean():.3f}; "
          f"parity rule correct on those: {(f.parity_pred == f.serverGetPoint).mean():.3f}")
