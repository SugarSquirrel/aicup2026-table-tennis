"""Score the post-competition re-runs (paper Sec. V-D / VI-C).

Usage: python paper_analysis/score_reruns.py
Inputs (written by paper_analysis/run_experiments.sh into the repo root):
  submission_v7-aug_incl0.csv, submission_v14-aug-ovr_incl0.csv, submission_v14g-aug-ovr_incl0.csv
1) Reproduction: re-run v7 / v14 vs the archived submissions (prediction agreement and scores).
2) v14g (old samples for action models only) vs the re-run v14 on the same machine:
   private score, components, per-stratum F1 and paired-bootstrap intervals.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "results"
B, SEED = 2000, 0

ans = pd.read_csv(ROOT / "test_answer_solution.csv").set_index("rally_uid")
te = pd.read_csv(ROOT / "data/test_new.csv")
tr = pd.read_csv(ROOT / "data/train.csv")
ref = pd.read_csv(ROOT / "data/Reference_Only_Old_Test_Data/test.csv")
pub_uids = set(ref.rally_uid)
P_tr = set(tr.gamePlayerId) | set(tr.gamePlayerOtherId)
P_ref = set(ref.gamePlayerId) | set(ref.gamePlayerOtherId)
prv_uids = sorted(set(te.rally_uid) - pub_uids)
nh = (te[te.rally_uid.isin(prv_uids)].sort_values("strikeNumber").groupby("rally_uid").tail(1)
      .set_index("rally_uid").gamePlayerOtherId.loc[prv_uids])
S = np.where(nh.isin(P_tr), "seen", np.where(nh.isin(P_ref), "rescued", "cold"))


def load(path):
    return pd.read_csv(path).set_index("rally_uid").loc[ans.index]


def mf1(y, p):
    return f1_score(y, p, labels=np.unique(y), average="macro", zero_division=0)


def parts(sub, uids):
    a, s = ans.loc[uids], sub.loc[uids]
    fa, fp = mf1(a.actionId, s.actionId), mf1(a.pointId, s.pointId)
    auc = roc_auc_score(a.serverGetPoint, s.serverGetPoint)
    return 0.4 * fa + 0.4 * fp + 0.2 * auc, fa, fp, auc


subs = {
    "v7 archived": ROOT / "submission/5/submission_v7-aug_incl0.csv",
    "v7 re-run": ROOT / "submission_v7-aug_incl0.csv",
    "v14 archived": ROOT / "submission/8/submission_v14-aug-ovr_incl0.csv",
    "v14 re-run": ROOT / "submission_v14-aug-ovr_incl0.csv",
    "v14g": ROOT / "submission_v14g-aug-ovr_incl0.csv",
}
subs = {k: load(v) for k, v in subs.items() if v.exists()}
print("loaded:", list(subs))

print("\n== Reproduction (agreement on all 1845 rallies)")
for v in ("v7", "v14"):
    a_, r_ = subs.get(f"{v} archived"), subs.get(f"{v} re-run")
    if a_ is None or r_ is None:
        continue
    print(f"{v}: action same {np.mean(a_.actionId == r_.actionId):.3f}, point same {np.mean(a_.pointId == r_.pointId):.3f}, "
          f"server |diff| max {np.abs(a_.serverGetPoint - r_.serverGetPoint).max():.4f}")

rows = []
print("\n== Scores")
for k, sub in subs.items():
    pu = parts(sub, sorted(pub_uids))
    pr = parts(sub, prv_uids)
    rows.append(dict(sub=k, public=pu[0], private=pr[0], prv_action_f1=pr[1], prv_point_f1=pr[2], prv_auc=pr[3]))
res = pd.DataFrame(rows)
print(res.round(4).to_string(index=False))

strata = []
print("\n== Private per-stratum F1")
for k, sub in subs.items():
    s = sub.loc[prv_uids]; a = ans.loc[prv_uids]
    row = dict(sub=k)
    for st in ("seen", "rescued", "cold"):
        m = S == st
        row[f"act_{st}"] = mf1(a.actionId.to_numpy()[m], s.actionId.to_numpy()[m])
        row[f"pt_{st}"] = mf1(a.pointId.to_numpy()[m], s.pointId.to_numpy()[m])
    strata.append(row)
strata = pd.DataFrame(strata)
print(strata.round(3).to_string(index=False))

boot = []
if "v14g" in subs and "v14 re-run" in subs:
    print("\n== v14 re-run -> v14g, paired bootstrap on private (95% CI)")
    rng = np.random.default_rng(SEED)
    base, new = subs["v14 re-run"].loc[prv_uids], subs["v14g"].loc[prv_uids]
    a = ans.loc[prv_uids]
    yA, yP, yR = a.actionId.to_numpy(), a.pointId.to_numpy(), a.serverGetPoint.to_numpy()

    def comp(sub, i):
        fa, fp = mf1(yA[i], sub.actionId.to_numpy()[i]), mf1(yP[i], sub.pointId.to_numpy()[i])
        auc = roc_auc_score(yR[i], sub.serverGetPoint.to_numpy()[i])
        return np.array([0.4 * fa + 0.4 * fp + 0.2 * auc, fa, fp, auc])

    n = len(prv_uids)
    full = np.arange(n)
    d0 = comp(new, full) - comp(base, full)
    ds = np.array([comp(new, i) - comp(base, i) for i in (rng.integers(0, n, n) for _ in range(B))])
    for j, name in enumerate(["score", "action_f1", "point_f1", "server_auc"]):
        lo, hi = np.percentile(ds[:, j], [2.5, 97.5])
        boot.append(dict(metric=name, delta=d0[j], ci_lo=lo, ci_hi=hi))
        print(f"{name:10s} {d0[j]:+.4f} [{lo:+.4f}, {hi:+.4f}]")
    for st in ("rescued", "cold"):
        idx0 = np.flatnonzero(S == st)
        for col, name in [("actionId", "action_f1"), ("pointId", "point_f1")]:
            y = a[col].to_numpy(); pb, pn = base[col].to_numpy(), new[col].to_numpy()
            d = mf1(y[idx0], pn[idx0]) - mf1(y[idx0], pb[idx0])
            bs = []
            for _ in range(B):
                i = rng.choice(idx0, len(idx0))
                bs.append(mf1(y[i], pn[i]) - mf1(y[i], pb[i]))
            lo, hi = np.percentile(bs, [2.5, 97.5])
            boot.append(dict(metric=f"{st}_{name}", delta=d, ci_lo=lo, ci_hi=hi))
            print(f"{st:8s} {name:9s} {d:+.3f} [{lo:+.3f}, {hi:+.3f}]")

OUT.mkdir(exist_ok=True)
res.to_csv(OUT / "reruns_scores.csv", index=False)
strata.to_csv(OUT / "reruns_strata.csv", index=False)
if boot:
    pd.DataFrame(boot).to_csv(OUT / "reruns_v14g_bootstrap.csv", index=False)
print(f"\nsaved results to {OUT}")
