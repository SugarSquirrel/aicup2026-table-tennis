"""Checks on the structure of the private test IDs and data consistency (paper v2: Table I, Sec. III-C, Sec. VII).

Usage: python paper_analysis/check_split_ids.py
Needs data/train.csv, data/test_new.csv, data/Reference_Only_Old_Test_Data/test.csv (gitignored).
This is the "check_a2_ids.py" referred to in the v2 TODO comments.

Definitions used below
- private part = test_new rallies whose rally_uid is not in the reference file; pooled IDs = private match IDs
  340-354, new matches = private match IDs 355-363 (the observed private match IDs are printed first).
- pairing = unordered pair of distinct IDs in gamePlayerId/gamePlayerOtherId over all strokes of a rally
  (robust to records that list the hitter as their own opponent).
- rally-number check: rally_id == scoreSelf + scoreOther + 1 on the first observed stroke (the score sum is
  constant within a rally because scoreSelf/scoreOther swap with the hitter).
- score-progression test (exact procedure): the oriented score state of a rally is
  (score of the lower player ID, score of the higher player ID), taken from its first observed stroke
  (scoreSelf belongs to gamePlayerId). A pooled rally is "consistent" with a set of reference games if its
  oriented state equals the oriented state of at least one rally in those games (game = match x numberGame).
  matched    = consistency with all reference games of the rally's own pairing;
  mismatched = consistency with all reference games of a different pooled pairing, assigned by a random
               derangement of the pooled pairings (np.random.default_rng(0), 20 repetitions; mean/min/max).
  The lower/higher-ID orientation is kept when the other pairing's games are used.
Output: results/split_checks.csv (one row per reported quantity).
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "results"
SEED, REPS = 0, 20

tr = pd.read_csv(ROOT / "data/train.csv")
te = pd.read_csv(ROOT / "data/test_new.csv")
ref = pd.read_csv(ROOT / "data/Reference_Only_Old_Test_Data/test.csv")
pub_uids = set(ref.rally_uid)
prv = te[~te.rally_uid.isin(pub_uids)]
print("private match IDs:", sorted(prv.match.unique()))
pooled = prv[prv.match.between(340, 354)]
new = prv[prv.match.between(355, 363)]
other = prv[~prv.match.between(340, 363)]
if len(other):
    print(f"WARNING: {other.rally_uid.nunique()} private rallies outside match IDs 340-363")

rows = []


def rec(check, group, value, note=""):
    rows.append(dict(check=check, group=group, value=value, note=note))
    print(f"{check:34s} {group:14s} {value}  {note}")


def firsts(df):
    return df.sort_values("strikeNumber").groupby("rally_uid").head(1).set_index("rally_uid")


def players(df):
    return set(df.gamePlayerId) | set(df.gamePlayerOtherId)


def pairings(df):
    ids = pd.concat([df[["rally_uid", "gamePlayerId"]].set_axis(["rally_uid", "p"], axis=1),
                     df[["rally_uid", "gamePlayerOtherId"]].set_axis(["rally_uid", "p"], axis=1)])
    pr = ids.groupby("rally_uid").p.agg(lambda s: tuple(sorted(set(int(x) for x in s))))
    bad = pr[pr.map(len) != 2]
    if len(bad):
        print(f"WARNING: {len(bad)} rallies without exactly two player IDs: {bad.head().to_dict()}")
    return pr


# ---- 1. rally number vs score ----
print("== rally number = scoreSelf + scoreOther + 1")
for name, df in [("train", tr), ("reference", ref), ("new matches", new), ("pooled IDs", pooled)]:
    f = firsts(df)
    ok = (f.rally_id == f.scoreSelf + f.scoreOther + 1)
    rec("rally_number_eq_score_sum_plus_1", name, round(float(ok.mean()), 4), f"{int(ok.sum())}/{len(ok)} rallies")

# ---- 2. pairings ----
print("\n== pairings")
pr_ref = set(pairings(ref))
pr_new, pr_pool = pairings(new), pairings(pooled)
rec("pairings", "new matches", len(set(pr_new)))
rec("pairings", "pooled IDs", len(set(pr_pool)))
rec("pairings", "private total", len(set(pr_new) | set(pr_pool)))
rec("pairings_in_reference", "new matches", len(set(pr_new) & pr_ref), f"of {len(set(pr_new))}")
rec("pairings_in_reference", "pooled IDs", len(set(pr_pool) & pr_ref), f"of {len(set(pr_pool))}")
per_id = pooled.assign(p=pooled.rally_uid.map(pr_pool)).groupby("match").p.nunique()
rec("pairings_per_pooled_ID", "pooled IDs", f"{per_id.min()}-{per_id.max()}", f"over {len(per_id)} IDs")
per_new = new.assign(p=new.rally_uid.map(pr_new)).groupby("match").p.nunique()
rec("pairings_per_new_match", "new matches", f"{per_new.min()}-{per_new.max()}", f"over {len(per_new)} IDs")

# ---- 3. score-progression consistency of pooled rallies ----
print("\n== score-progression test (see docstring)")


def oriented_state(f, pr):
    lo = np.array([pr[u][0] for u in f.index])
    s_lo = np.where(f.gamePlayerId.to_numpy() == lo, f.scoreSelf, f.scoreOther)
    s_hi = np.where(f.gamePlayerId.to_numpy() == lo, f.scoreOther, f.scoreSelf)
    return list(zip(s_lo.astype(int), s_hi.astype(int)))


fr = firsts(ref)
pr_ref_s = pairings(ref)
fr = fr.assign(p=pr_ref_s.loc[fr.index], state=oriented_state(fr, pr_ref_s))
states_by_pair = fr.groupby("p").state.apply(set).to_dict()
fp = firsts(pooled)
fp = fp.assign(p=pr_pool.loc[fp.index], state=oriented_state(fp, pr_pool))
matched = np.mean([s in states_by_pair.get(p, set()) for p, s in zip(fp.p, fp.state)])
rng = np.random.default_rng(SEED)
keys = sorted(set(fp.p))
mm = []
for _ in range(REPS):
    while True:  # derangement of pooled pairings
        perm = rng.permutation(len(keys))
        if not np.any(perm == np.arange(len(keys))):
            break
    to = {keys[i]: keys[perm[i]] for i in range(len(keys))}
    mm.append(np.mean([s in states_by_pair.get(to[p], set()) for p, s in zip(fp.p, fp.state)]))
rec("score_state_consistency", "matched", round(float(matched), 4), f"{len(fp)} pooled rallies")
rec("score_state_consistency", "mismatched", round(float(np.mean(mm)), 4),
    f"min {min(mm):.4f} max {max(mm):.4f} over {REPS} derangements")

# ---- 4. Table I rows ----
print("\n== Table I rows")
for name, df in [("new matches", new), ("pooled IDs", pooled), ("private", prv)]:
    rec("tableI_match_ids", name, df.match.nunique())
    rec("tableI_rallies", name, df.rally_uid.nunique())
    rec("tableI_strokes", name, len(df))
    rec("tableI_players", name, len(players(df)))

# ---- 5. private players ----
print("\n== private players")
P_prv, P_tr, P_ref = players(prv), players(tr), players(ref)
rec("private_players", "total", len(P_prv))
rec("private_players", "in reference", len(P_prv & P_ref))
rec("private_players", "in train", len(P_prv & P_tr))
neither = sorted(P_prv - P_ref - P_tr)
rec("private_players", "in neither", len(neither), "IDs " + ",".join(map(str, neither)))
rec("private_players_in_neither", "play in new matches", len(set(neither) & players(new)), f"of {len(neither)}")

# ---- 6. data inconsistencies ----
print("\n== data inconsistencies")
long = pd.concat([tr[["match", "gamePlayerId", "sex"]].rename(columns={"gamePlayerId": "player"}),
                  tr[["match", "gamePlayerOtherId", "sex"]].rename(columns={"gamePlayerOtherId": "player"})])
nsex = long.groupby("player").sex.nunique()
multi = nsex[nsex > 1].index.tolist()
for p in multi:
    detail = long[long.player == p].groupby("sex").match.unique().to_dict()
    rec("train_player_multiple_sex", f"player {p}", len(detail), str({int(k): sorted(map(int, v)) for k, v in detail.items()}))
rec("train_players_with_multiple_sex", "train", len(multi))
for name, df in [("train", tr), ("reference", ref), ("test", te)]:
    rec("hitter_equals_opponent_records", name, int((df.gamePlayerId == df.gamePlayerOtherId).sum()))


def sex_map(df):
    l = pd.concat([df[["gamePlayerId", "sex"]].rename(columns={"gamePlayerId": "p"}),
                   df[["gamePlayerOtherId", "sex"]].rename(columns={"gamePlayerOtherId": "p"})])
    return l.groupby("p").sex.agg(lambda s: frozenset(s))


for a_name, a_df, b_name, b_df in [("train", tr, "test", te), ("reference", ref, "test", te)]:
    ma, mb = sex_map(a_df), sex_map(b_df)
    shared = sorted(set(ma.index) & set(mb.index))
    agree = sum(ma[p] == mb[p] for p in shared)
    rec("shared_ids_same_sex", f"{a_name}-{b_name}", f"{agree}/{len(shared)}")

OUT.mkdir(exist_ok=True)
pd.DataFrame(rows).to_csv(OUT / "split_checks.csv", index=False)
print(f"\nsaved {OUT / 'split_checks.csv'}")
