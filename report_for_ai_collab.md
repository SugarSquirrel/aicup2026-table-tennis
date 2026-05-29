# AI CUP 2026 桌球預測 — 給協作 AI 的現況與策略 report

> 目的：與 ChatGPT / Gemini 三方腦力激盪，把公開榜分數從 **0.3511（rank 113/371）** 推向 **前 30（~0.44）**。
> 撰寫者：Claude（負責實作與 CV 驗證）。使用者負責在 3 個 AI 之間轉達。

## 1. 任務與資料（精簡事實，供未看過的 AI 參考）
- 每個 rally 用前 n-1 拍預測第 n 拍的 `actionId`(球種,19類,0–18)、`pointId`(落點,10類,0–9；1–9 是九宮格、0 是終結/無落點)、`serverGetPoint`(發球方是否得分,輸出機率)。
- 評分 `Final = 0.4·F1_action(macro) + 0.4·F1_point(macro) + 0.2·AUC`。
- **資料是離散類別碼**（沒有座標/球速/軌跡等連續物理量；故 physics-informed loss 不適用）。
- 14995 train rally / 1845 test rally。選手嚴格交替出拍，**下一拍擊球者 next_hitter = test 已知欄位 gamePlayerOtherId**（無洩漏）。
- **選手重疊**：train 166 人、test 71 人，40 人見過；**test 94% 的 rally 至少含一位見過的選手**，只有 6.1% 兩位都沒見過。**match（場次）完全不重疊。**
- macro-F1 在「真值出現的類別」上平均、**含 class 0**（已用提交驗證）。
- 官方：第 n 拍不一定是回合最後一拍（隨機刪除若干連續拍）。
- 記憶上限診斷：模型 in-sample 可達 action 0.96 / point 0.92 / serverAUC 0.99 → **這是泛化問題，不是標籤雜訊**。

## 2. 目前最佳做法（v2, 公開榜 0.3511）
- 樣本「前綴→下一拍」(保留終結拍)；驗證 **GroupKFold by match**；macro-F1 用 prior-correction 決策。
- 特徵 = base(最後一拍+lag2/3+比分+parity) + **fold-safe transition** `P(next_*|last_action,last_point)` + **fold-safe 選手球種傾向** `P(next_action|next_hitter)`、`P(next_point|next_hitter)`（只給 action/point）。
- 模型 = **LightGBM ⊕ TabPFN ⊕ 小型 GRU**，per-task OOF 權重 + 三方混合。
- 拆解：action F1≈0.354、point F1≈0.189、server AUC≈0.61。

## 3. ★最關鍵發現：公開榜「低估校正」
| 版本 | match-CV | 公開榜 | CV→LB |
|---|---|---|---|
| v1 (base, 無 player) | 0.315 | 0.319 | +0.004 |
| v2 (+ 選手球種傾向) | 0.341 | 0.351 | **+0.010** |
| **player 特徵增益** | **CV +0.026** | **LB +0.032** | player 在 LB 更有效 |

**結論**：公開測試集是「選手豐富 / train-like」，**我們嚴格的 match-CV 低估了所有「選手/已見資料」類特徵**（因為它把整場留出、人為製造未見選手）。
→ **凡是我們因「跨未見場次不泛化」而在 CV 上否決的『選手相關』方法，很可能在排行榜上其實有用，值得重測。**

## 4. 已驗證有效 / 已否決（含「該重新考慮」標記）
**有效**：transition 機率特徵(action/point)、選手球種傾向(player-marginal, action/point)、GRU 序列模型當第 3 成員、server 用 TabPFN-heavy。

**在 match-CV 否決，但★該重新考慮（因 LB 對選手特徵友善）★**：
- player×state 交互 `P(next_action|player,last_action,last_point)`（CV action −0.024，疑似 CV 低估）
- 選手發球得分率 prior 給 server（CV AUC −0.016；但 server 仍最弱、且 LB 選手豐富）
- **raw player ID embedding**（從未正式試；LB 選手豐富 → 最可能有用，但對 private 有過擬合風險）

**否決且維持否決（非選手類）**：點位幾何特徵 / two-stage / row×col 分解、stage-specific transition、額外靜態特徵(lag4-5/first-stroke/window/role)、physics-informed loss(無連續資料)、player×state(若 LB 也無效)。

## 5. 待嘗試的點子（依預期效益排序）
1. **更積極的選手建模**（LB 已證明這條最有效）：raw player ID embedding（GRU two-stream + LGBM categorical）、選手 point/spin/發球型態傾向、player×state、選手對戰(matchup)。
2. **更好的驗證**：建「**seen-player 加權 CV**」模擬 test 的 94%-見過選手分布，讓我們能離線驗證選手特徵、不必燒提交額度（目前 match-CV 會誤導我們否決有用的選手方法）。
3. **serverGetPoint（最弱 0.61）**：對手強度 / matchup；重新考慮選手 prior（給定 LB 選手豐富）。
4. 決策規則對齊實際 LB 的 macro-F1（prior-correction 強度微調）。

## 6. 給 ChatGPT / Gemini 的尖銳問題
1. 公開榜是「選手豐富」、match-CV 低估選手特徵 → 我們該多積極用 raw player ID / 選手專屬建模，才能衝 LB **又不會在 private 崩**（private 的選手重疊未知）？
2. 如何設計一個能「預測公開榜（選手豐富）表現」的離線 CV？（例如：留出場次但只在『其選手於 train-fold 出現過』的 rally 上評估？）
3. serverGetPoint 本質由「最終 rally 長度奇偶」決定、但 test 看不到最終長度（AUC 卡 0.61）。有沒有特徵能從截斷前綴預測「還會打幾拍/長度奇偶」？
4. 前段隊伍 0.44+ 最可能靠什麼？（我們推測是重度選手建模 / per-player 校準）

## 6b. ★seen-player CV 驗證結果（2026-05-29，重要新數據）
建了「已見選手子集」評估（test 實際 94% 是已見選手，此子集最能預測排行榜）。action macro-F1：
| 變體 | 全體 match-CV | **已見選手(≈LB)** | 未見選手 |
|---|---|---|---|
| player-marginal (v2) | 0.336 | 0.356 | 0.190 |
| + player×state | 0.320 | 0.348 | 0.183 |
| + **raw player ID** | 0.334 | **0.361** | 0.183 |
| both | 0.323 | 0.346 | 0.185 |

結論：(a) **已見 0.356 vs 未見 0.190**——巨大落差證實「LB≈已見選手表現」；(b) **raw player ID 在已見子集有效(+0.005)**、值得納入(傷未見但 test 幾乎全是已見)；(c) **player×state 連已見都更差 → 確定無用**。

## 6c. 本輪「試過但失敗」清單（請勿再建議這些，已用 CV/seen-CV 驗證無效）
- **transductive 同場次選手畫像**（用 test 該場觀測拍聚合「選手當下風格」, leave-one-rally-out）：action F1 0.336→0.256（**大幅變差**，計數稀疏+train/test 截斷不一致→過擬合）。
- player×state `P(action|player,last_action)`：seen 0.356→0.348（變差）。
- 選手 prior 給 serverGetPoint：seen AUC 0.593→0.580（變差；server 由長度奇偶決定、非選手技能）。
- raw player ID：seen action +0.005（小贏，唯一可加），但對 server/未見無益。
- 點位幾何/two-stage/row×col 分解、stage-specific transition、lag4-5/first-stroke/window/role、physics-loss：先前皆已否決。
- 更大的 GRU / 雙向 / attention / 多 seed：ensemble 無提升。

## ★中心謎題（最想請 ChatGPT/Gemini 解的）
**即使在「已見選手」上，action F1 也只有 0.356**，但 in-sample 記憶可達 0.96。代表：**同一位選手在不同對手/場次的打法會變，光靠『選手平均傾向』補不滿這個落差。** 前 30 名(0.44+) 必然在「已見選手」上做得遠比 0.356 好。**他們可能用什麼？** 候選假設：
- per-player × 對手(matchup) 條件模型（但 pair 重疊低、稀疏）
- 選手在「該場比賽內」的 self-adaptation（用 test 該 rally 已觀測的前幾拍即時校準該選手當下風格）→ test-time adaptation / in-context per-rally
- 更強的序列模型專門吃「選手 embedding × 當前序列」的交互
- 對 point/server 也做同等強度的選手建模（我們目前 point 只有弱 player 訊號、server 沒有）

## 7. 限制與紀律
- 每天提交有限（與組員共用），且**排名以最後一次上傳計**；故每次提交要是「更好」的版本。
- 最終名次看 **Private Leaderboard（6/3 公布）**——不要為公開榜過擬合到傷害 private。
- 所有選手統計必須 **fold-safe**（fold-train 統計、套到 val；test 用全 train）。
