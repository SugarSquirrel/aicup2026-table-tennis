"""Cluster (player-pairing) bootstrap of private-score differences (paper v2: Table V caption, Secs. V-C/V-D, VI, VII).

Usage: python paper_analysis/pair_bootstrap_private.py
Needs data/test_new.csv, data/train.csv, data/Reference_Only_Old_Test_Data/test.csv and
test_answer_solution.csv in the repo root (all gitignored). ActionOnly (v14g) is read from the repo root
file submission_v14g-aug-ovr_incl0.csv if present (produced by run_experiments.sh); otherwise it is skipped.

Why: rallies of the same pairing are not independent, and private match IDs 340-354 pool several pairings,
so bootstrap_private.py (which resamples rallies) can understate the uncertainty. Here the unit of resampling
is the player pairing.

Pairing of a rally = the unordered pair of distinct IDs in gamePlayerId and gamePlayerOtherId over all its
observed strokes in data/test_new.csv (players alternate; taking all strokes is robust to the few records that
list the hitter as their own opponent).
Each resample draws the private pairings with replacement and takes all rallies of every drawn pairing
(with multiplicity); both systems are scored on the same indices (paired).
Conventions otherwise follow bootstrap_private.py / private_strata.py: B=2000, SEED=0,
np.random.default_rng(SEED), 95% percentile intervals, macro-F1 over labels present in the ground truth,
score = 0.4*F1_action + 0.4*F1_point + 0.2*AUC; strata by the next hitter (gamePlayerOtherId of the last
observed stroke): seen = in train, rescued = only in the reference data, cold = in neither.
Outputs: results/pair_bootstrap_private.csv (overall) and results/pair_bootstrap_strata.csv (per stratum).
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score  # f1_score only used in the self-check below

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "results"
B, SEED = 2000, 0

SUBS = {
    "v1": "submission/1/submission_incl0.csv",
    "v2": "submission/2/submission_v2-player_cv0.341_incl0.csv",
    "v3": "submission/3/submission_v3-matchup_cvb0.350_incl0.csv",
    "v7": "submission/5/submission_v7-aug_incl0.csv",
    "v10": "submission/6/submission_v10-aug-ovr_incl0.csv",
    "v12": "submission/7/submission_v12-aug-ovr_incl0.csv",
    "v14": "submission/8/submission_v14-aug-ovr_incl0.csv",
    "v14g": "submission_v14g-aug-ovr_incl0.csv",  # ActionOnly, repo root (gitignored)
}
# same steps as bootstrap_private.py PAIRS, plus Submitted -> ActionOnly
PAIRS = [("v1", "v2"), ("v2", "v3"), ("v3", "v7"), ("v7", "v10"), ("v10", "v12"), ("v12", "v14"),
         ("v1", "v7"), ("v7", "v14"), ("v1", "v14"), ("v14", "v14g")]
# same steps as private_strata.py, plus Submitted -> ActionOnly
STRATA_PAIRS = [("v3", "v7"), ("v7", "v10"), ("v12", "v14"), ("v7", "v14"), ("v14", "v14g")]

te = pd.read_csv(ROOT / "data/test_new.csv")
tr = pd.read_csv(ROOT / "data/train.csv")
ref = pd.read_csv(ROOT / "data/Reference_Only_Old_Test_Data/test.csv")
ans = pd.read_csv(ROOT / "test_answer_solution.csv")
ans = ans[ans.rally_uid >= 20000].sort_values("rally_uid").reset_index(drop=True)  # 609 private rallies
uids = ans.rally_uid.to_numpy()

def rally_pairings(df):
    """Unordered pair of players of each rally, from all its strokes (robust to the few records that list the
    hitter as their own opponent). Rallies with other than two distinct IDs are reported."""
    ids = pd.concat([df[["rally_uid", "gamePlayerId"]].set_axis(["rally_uid", "p"], axis=1),
                     df[["rally_uid", "gamePlayerOtherId"]].set_axis(["rally_uid", "p"], axis=1)])
    pr = ids.groupby("rally_uid").p.agg(lambda s: tuple(sorted(set(int(x) for x in s))))
    bad = pr[pr.map(len) != 2]
    if len(bad):
        print(f"WARNING: {len(bad)} rallies without exactly two player IDs: {bad.head().to_dict()}")
    return pr


subs = {}
for k, path in SUBS.items():
    p = ROOT / path
    if not p.exists():
        print(f"[skip] {k}: {path} not found")
        continue
    subs[k] = pd.read_csv(p).set_index("rally_uid").loc[uids].reset_index()
PAIRS = [(a, b) for a, b in PAIRS if a in subs and b in subs]
STRATA_PAIRS = [(a, b) for a, b in STRATA_PAIRS if a in subs and b in subs]

# pairing and stratum of each private rally
pair_key = list(rally_pairings(te).loc[uids])
P_tr = set(tr.gamePlayerId) | set(tr.gamePlayerOtherId)
P_ref = set(ref.gamePlayerId) | set(ref.gamePlayerOtherId)
nh = te.sort_values("strikeNumber").groupby("rally_uid").tail(1).set_index("rally_uid").gamePlayerOtherId.loc[uids]
S = np.where(nh.isin(P_tr), "seen", np.where(nh.isin(P_ref), "rescued", "cold"))

clusters = {}
for i, p in enumerate(pair_key):
    clusters.setdefault(p, []).append(i)
keys = sorted(clusters)
match_of = te.groupby("rally_uid").match.first().loc[uids].to_numpy()
n_new = len({pair_key[i] for i in range(len(uids)) if 355 <= match_of[i] <= 363})
n_pool = len({pair_key[i] for i in range(len(uids)) if 340 <= match_of[i] <= 354})
print(f"private rallies {len(uids)}, pairings {len(keys)} "
      f"(pairings in match IDs 355-363: {n_new}, in 340-354: {n_pool})")

yA, yP, yR = ans.actionId.to_numpy(), ans.pointId.to_numpy(), ans.serverGetPoint.to_numpy()


def mf1(y, p):
    """Macro-F1 over the labels present in y (same as sklearn f1_score(labels=np.unique(y), average="macro",
    zero_division=0)); numpy version so that 2,000 cluster resamples x many steps run in a few minutes."""
    k = int(max(y.max(), p.max())) + 1
    tp = np.bincount(y[y == p], minlength=k).astype(float)
    t = np.bincount(y, minlength=k).astype(float)
    pr = np.bincount(p, minlength=k).astype(float)
    lab = t > 0
    denom = t[lab] + pr[lab]
    return float(np.mean(np.where(denom > 0, 2 * tp[lab] / np.where(denom > 0, denom, 1), 0.0)))


def parts(sub, idx):
    fa = mf1(yA[idx], sub.actionId.to_numpy()[idx])
    fp = mf1(yP[idx], sub.pointId.to_numpy()[idx])
    r = yR[idx]
    auc = roc_auc_score(r, sub.serverGetPoint.to_numpy()[idx]) if len(np.unique(r)) == 2 else np.nan
    return np.array([0.4 * fa + 0.4 * fp + 0.2 * auc, fa, fp, auc])


def draw(rng, cluster_keys, members):
    pick = rng.integers(0, len(cluster_keys), len(cluster_keys))
    return np.concatenate([members[cluster_keys[j]] for j in pick])


# self-check: the numpy macro-F1 equals sklearn's on the full private set
for k in subs:
    for col, y in [("actionId", yA), ("pointId", yP)]:
        q = subs[k][col].to_numpy()
        assert abs(mf1(y, q) - f1_score(y, q, labels=np.unique(y), average="macro", zero_division=0)) < 1e-12

rng = np.random.default_rng(SEED)
members = {k: np.array(v) for k, v in clusters.items()}
draws = [draw(rng, keys, members) for _ in range(B)]   # shared by all steps (paired)
full = np.arange(len(uids))

rows = []
print(f"\n== overall, resampling {len(keys)} pairings ({B} resamples)")
print(f"{'pair':10s} {'metric':7s} {'delta':>8s} {'95% CI':>20s} {'P(delta>0)':>10s}")
for a, b in PAIRS:
    d0 = parts(subs[b], full) - parts(subs[a], full)
    ds = np.array([parts(subs[b], i) - parts(subs[a], i) for i in draws])
    for j, name in enumerate(["score", "action_f1", "point_f1", "server_auc"]):
        lo, hi = np.nanpercentile(ds[:, j], [2.5, 97.5])
        pos = np.nanmean(ds[:, j] > 0)
        rows.append(dict(pair=f"{a}->{b}", metric=name, delta=d0[j], ci_lo=lo, ci_hi=hi, p_pos=pos,
                         excludes_zero=bool(lo > 0 or hi < 0), n_clusters=len(keys)))
        print(f"{a + '->' + b:10s} {name[:7]:7s} {d0[j]:+8.4f} [{lo:+.4f}, {hi:+.4f}] {pos:10.3f}")

# per stratum: clusters = pairings with at least one rally in the stratum; take only their stratum rallies
srows = []
print(f"\n== per stratum, resampling the pairings present in each stratum ({B} resamples)")
for s in ("seen", "rescued", "cold"):
    m = {}
    for i in np.flatnonzero(S == s):
        m.setdefault(pair_key[i], []).append(i)
    skeys = sorted(m)
    smembers = {k: np.array(v) for k, v in m.items()}
    sdraws = [draw(rng, skeys, smembers) for _ in range(B)]
    idx0 = np.flatnonzero(S == s)
    print(f"-- {s}: {len(idx0)} rallies, {len(skeys)} pairings")
    for a, b in STRATA_PAIRS:
        for col, y, name in [("actionId", yA, "action_f1"), ("pointId", yP, "point_f1")]:
            pa, pb = subs[a][col].to_numpy(), subs[b][col].to_numpy()
            d0 = mf1(y[idx0], pb[idx0]) - mf1(y[idx0], pa[idx0])
            ds = np.array([mf1(y[i], pb[i]) - mf1(y[i], pa[i]) for i in sdraws])
            lo, hi = np.percentile(ds, [2.5, 97.5])
            srows.append(dict(pair=f"{a}->{b}", stratum=s, metric=name, delta=d0, ci_lo=lo, ci_hi=hi,
                              excludes_zero=bool(lo > 0 or hi < 0), n_rallies=len(idx0), n_clusters=len(skeys)))
            print(f"   {a + '->' + b:9s} {name:9s} {d0:+.3f} [{lo:+.3f}, {hi:+.3f}]")

OUT.mkdir(exist_ok=True)
pd.DataFrame(rows).to_csv(OUT / "pair_bootstrap_private.csv", index=False)
pd.DataFrame(srows).to_csv(OUT / "pair_bootstrap_strata.csv", index=False)
print(f"\nsaved {OUT / 'pair_bootstrap_private.csv'} and {OUT / 'pair_bootstrap_strata.csv'}")
