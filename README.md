# AI CUP 2026 — 基於時序資料之桌球戰術與結果預測

每個 rally 用前 n-1 拍預測第 n 拍的 `actionId`(球種,19類)、`pointId`(落點,10類)，與 `serverGetPoint`(發球方得分機率)。
評分：`Final = 0.4·F1_action(macro) + 0.4·F1_point(macro) + 0.2·AUC`。

## 方法總覽
- 樣本：每個 rally 切「前綴(1..L) → 下一拍(L+1)」，保留終結拍(pointId=0)；驗證用比照 test 長度的單一前綴。
- 驗證：**GroupKFold by match**（跨場次，誠實估計；與公開榜校準：CV 0.315 ↔ LB 0.319）。
- 特徵：base(最後一拍+lag2/3+比分+parity) + **fold-safe transition** `P(next_*|last_action,last_point)` + **fold-safe 選手球種傾向** `P(next_action|next_hitter)`（player-marginal）。
- 模型：**LightGBM ⊕ TabPFN ⊕ 小型 GRU**，per-task OOF 權重搜尋 + macro-F1 的 prior-correction 決策。
- 已驗證**無效並剔除**：lag45/first-stroke/window/role 靜態特徵、two-stage/row-col 點位分解、stage-transition、選手發球得分率 prior（player 只在 action/point 有效、server 無效）。

## 版本 ↔ 分數 ↔ 程式碼 對照（最重要：用來回溯最高分版本）
| 版本 | CV Overall | 公開榜 | 主要差異 | 產生程式 | 提交檔 |
|---|---|---|---|---|---|
| v1-base | 0.315 | **0.3188** (rank 204/367) | LGBM+TabPFN+GRU | git tag `v1-base-lb0.319` 內的 `src/main.ipynb` | `submission/1/submission_incl0.csv` |
| **v2-player (目前 HEAD)** | **0.341** | 待提交 | + 選手球種傾向(player-marginal) | `src/main.ipynb`（現版）或 `src/gen_submission_player.py` | `submission_v2-player_cv0.341_incl0.csv` |

> `src/main.ipynb`（目前版本）= v2 最佳版。要回到 v1：`git checkout v1-base-lb0.319`。

> 比賽以**最後一次上傳**計分。若先前某版較高，務必確保**最後一次提交**回到該最高版（用下方 git tag 回溯對應程式碼即可重現）。

## 重現方式
1. 環境：conda `aicup2026`（python 3.11 + torch 2.5.1+cu121 + tabpfn 8.0.3 + tabpfn-extensions + lightgbm + matplotlib）。
2. 資料放 `data/`（從 AIdea 下載；本 repo 不含資料）。
3. TabPFN 首次需 `TABPFN_TOKEN`（見 priorlabs.ai）下載權重；之後本地推論免 token。**token 檔 `tabpfn_apikey.txt` 已被 .gitignore，切勿提交。**
4. 跑 `src/main.ipynb`（v1）或 `python src/gen_submission_player.py`（v2，於 `src/` 下執行）產生 submission。

## Git 版本工作流程（每個可提交版本都記錄）
- 每產出一個可提交版本：CSV 檔名標上 CV 分數（如 `submission_v2-player_cv0.341_incl0.csv`），commit 後打 tag：`git tag v2-player-cv0.341`。
- 要回到某版程式：`git checkout v2-player-cv0.341`。
- 已提交的檔案歸檔在 `submission/<次數>/`。
