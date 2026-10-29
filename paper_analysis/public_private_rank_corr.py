"""Rank correlation of public and private task-level F1 across the seven archived systems (paper Sec. V-C).

Usage: python paper_analysis/public_private_rank_corr.py
Reads results/scores.csv (written by score_submissions.py); needs no data files.

Why: label substitution changes the public serverGetPoint AUC but not the public action or point F1, so the
task-level F1 can be compared across all archived systems. Submission 4 is Core with label substitution
(identical action and point predictions to submission 5), so it is left out to keep one row per system.
Output: results/public_private_rank_corr.csv
"""
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

OUT = Path(__file__).resolve().parent / "results"
SYSTEMS = {"1": "Base", "2": "Player", "3": "Matchup", "5": "Core", "6": "Blend", "7": "Filter", "8": "Submitted"}

s = pd.read_csv(OUT / "scores.csv")
s = s[s.f1mode == "true"].assign(sys=lambda d: d["sub"].str.split("/").str[0])
p = s.pivot_table(index="sys", columns="split", values=["action_f1", "point_f1"]).loc[list(SYSTEMS)]
print(p.rename(index=SYSTEMS).round(4))

rows = []
for m in ("action_f1", "point_f1"):
    r = spearmanr(p[(m, "public")], p[(m, "private")])
    rows.append(dict(metric=m, n_systems=len(p), spearman_rho=r.statistic, p_value=r.pvalue))
    print(f"{m}: Spearman rho = {r.statistic:+.3f} (p = {r.pvalue:.3f}, n = {len(p)})")
pd.DataFrame(rows).to_csv(OUT / "public_private_rank_corr.csv", index=False)
print(f"saved {OUT / 'public_private_rank_corr.csv'}")
