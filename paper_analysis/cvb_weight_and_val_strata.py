"""CV-B weight, next-hitter seen share on the test set, and the pairings behind the validation rescued stratum
(paper v2: Sec. III-D, Sec. V-D, Sec. VII).

Usage: python paper_analysis/cvb_weight_and_val_strata.py
Needs data/train.csv, data/test_new.csv, data/Reference_Only_Old_Test_Data/test.csv (gitignored). CPU only.

1) CV-B weight: share of test_new rallies in which at least one of the two players of the rally
   appears in train.csv. The code uses the fixed constant 0.94; this recomputes where it came from.
2) Next-hitter seen share on the whole test set (next hitter = gamePlayerOtherId of the last observed stroke).
3) Validation rescued stratum of src/gen_submission_v7_actionaug.py, rebuilt without training any model:
   - the one-prefix-per-rally sample is regenerated with the same RNG calls as build(tr, "sampled", tld)
     (np.random.default_rng(42), rallies in groupby("rally_uid", sort=False) order, rallies with T<2 skipped,
     prefix length drawn from the test-length distribution restricted to L <= T-1, no RNG call when that set
     is empty);
   - folds = GroupKFold(5) over the sorted match IDs, exactly as in v7;
   - strata exactly as in v7's OOF loop: seen = next hitter is a next hitter of some all-prefix sample of the
     training folds; rescued = otherwise, and a next hitter of some all-prefix sample of the reference data.
   The script checks that it recovers v7's strata sizes (seen 11,723 / rescued 378 / cold 2,894, Table III). If it does not, the
   reconstruction is not exact and the pairing counts below must not be used.
   A sample's pairing is the unordered pair of distinct player IDs over all strokes of its rally; it is "in
   the reference data" if that pairing occurs among the reference rallies.
Note: GroupKFold's fold assignment can depend on the scikit-learn version; run in the pinned environment
(requirements.txt: scikit-learn==1.6.1, numpy==2.4.6).
Output: results/cvb_weight_and_val_strata.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "results"
SEED_V7 = 42
EXPECTED = {"seen": 11723, "rescued": 378, "cold": 2894}  # Table III sample counts printed by v7

tr = pd.read_csv(ROOT / "data/train.csv")
te = pd.read_csv(ROOT / "data/test_new.csv")
old = pd.read_csv(ROOT / "data/Reference_Only_Old_Test_Data/test.csv")
print(f"numpy {np.__version__}, scikit-learn {sklearn.__version__}")
rows = []


def rec(check, value, note=""):
    rows.append(dict(check=check, value=value, note=note))
    print(f"{check:48s} {value}  {note}")


def firsts(df):
    return df.sort_values("strikeNumber").groupby("rally_uid").head(1).set_index("rally_uid")


def pair_of(df):
    """Unordered pair of distinct player IDs over all strokes of each rally."""
    ids = pd.concat([df[["rally_uid", "gamePlayerId"]].set_axis(["rally_uid", "p"], axis=1),
                     df[["rally_uid", "gamePlayerOtherId"]].set_axis(["rally_uid", "p"], axis=1)])
    return ids.groupby("rally_uid").p.agg(lambda s: tuple(sorted(set(int(x) for x in s)))).to_dict()


P_tr = set(tr.gamePlayerId) | set(tr.gamePlayerOtherId)

# ---- 1) CV-B weight ----
pairs_te = pair_of(te)
any_seen = pd.Series({u: any(x in P_tr for x in pr) for u, pr in pairs_te.items()})
rec("test rallies with >=1 player in train", round(float(any_seen.mean()), 4), f"{int(any_seen.sum())}/{len(any_seen)}")

# ---- 2) next-hitter seen share on the test set ----
nh_te = te.sort_values("strikeNumber").groupby("rally_uid").tail(1).gamePlayerOtherId
rec("test next-hitter seen share", round(float(nh_te.isin(P_tr).mean()), 4), f"{int(nh_te.isin(P_tr).sum())}/{len(nh_te)}")

# ---- 3) rebuild v7's validation samples and strata ----
tld = te.groupby("rally_uid").size().value_counts().to_dict()
Ls = np.array(sorted(tld))
Ps = np.array([tld[l] for l in Ls], float)
Ps /= Ps.sum()


def rebuild(df, mode, seed=SEED_V7):
    """Mirror of build() in gen_submission_v7_actionaug.py, keeping only what the strata need."""
    rng = np.random.default_rng(seed)
    nh, lh, g, uid = [], [], [], []
    for rid, grp in df.groupby("rally_uid", sort=False):
        grp = grp.sort_values("strikeNumber")
        T = len(grp)
        if T < 2:
            continue
        gp, go = grp.gamePlayerId.to_numpy(), grp.gamePlayerOtherId.to_numpy()
        if mode == "all":
            Ll = range(1, T)
        else:
            ok = Ls <= T - 1
            Ll = [1] if ok.sum() == 0 else [int(rng.choice(Ls[ok], p=(Ps[ok] / Ps[ok].sum())))]
        for L in Ll:
            nh.append(int(go[L - 1])); lh.append(int(gp[L - 1])); g.append(int(grp.match.iloc[0])); uid.append(int(rid))
    return np.array(nh), np.array(lh), np.array(g), np.array(uid)


nha, lha, ga, _ = rebuild(tr, "all")
nhs, lhs, gs, uids = rebuild(tr, "sampled")
nhao, _, _, _ = rebuild(old, "all")
M = np.array(sorted(set(ga) | set(gs)))
fo = {}
for f, (_, vi) in enumerate(GroupKFold(5).split(M, groups=M)):
    for m in M[vi]:
        fo[m] = f
af = np.array([fo[m] for m in ga])
sf = np.array([fo[m] for m in gs])
old_players = set(nhao)
stratum = np.empty(len(nhs), dtype=object)
for f in range(5):
    fold_train_players = set(nha[af != f])
    ev = np.flatnonzero(sf == f)
    for i in ev:
        h = nhs[i]
        stratum[i] = "seen" if h in fold_train_players else ("rescued" if h in old_players else "cold")
    print(f"  fold {f}: eval n={len(ev)} seen={(stratum[ev] == 'seen').sum()} "
          f"rescued={(stratum[ev] == 'rescued').sum()} cold={(stratum[ev] == 'cold').sum()}")
for s in ("seen", "rescued", "cold"):
    rec(f"validation samples: {s}", int((stratum == s).sum()))
got = {s: int((stratum == s).sum()) for s in EXPECTED}
exact = got == EXPECTED
rec("reconstruction matches v7 strata sizes", exact,
    f"expected {EXPECTED}" if exact else
    f"expected {EXPECTED}, got {got} - NOT EXACT, do not use the pairing counts below for the paper")

pairs_tr = pair_of(tr)
pairs_ref = set(pair_of(old).values())
resc = np.flatnonzero(stratum == "rescued")
in_ref = np.array([pairs_tr[u] in pairs_ref for u in uids[resc]])
rec("rescued validation samples whose pairing is in reference", int(in_ref.sum()), f"of {len(resc)}")
rec("share", round(float(in_ref.mean()), 4) if len(resc) else np.nan)
rec("matches among rescued validation samples", len(set(gs[resc])))
rec("matches among those in-reference samples", len(set(gs[resc][in_ref])))

OUT.mkdir(exist_ok=True)
pd.DataFrame(rows).to_csv(OUT / "cvb_weight_and_val_strata.csv", index=False)
print(f"\nsaved {OUT / 'cvb_weight_and_val_strata.csv'}")
