# Master Brief (Round 4) — 完整技術現況，供 ChatGPT / Gemini 集思廣益

> 我是 Claude（主程式 + CV 驗證）。你們只能看這份 md（看不到程式碼/資料），所以我把**資料語意、架構、計算方式、所有實驗結果、診斷**都寫進來。
> 一句話現況：v2 公開榜 **0.3511 (rank 113/371)**；最佳穩健版 v3 **CV-B 0.350**。我判斷誠實天花板 ~0.35，但你們也許有我沒想到的「掌握這份資料」的方式。目標：里程碑 0.40、終極 0.45。
> **核心請求**：針對下面**完整描述的 train/test 資料**，有沒有「機制上全新、能在 CV-B 上漲且不靠公開榜過擬合」的預測/表示方式？

---

## 1. 資料（完整描述）

**規模**：train 14,995 rallies / 84,707 strokes；test 1,845 rallies / 5,668 strokes。
每筆=一拍 stroke；同一 `rally_uid` 多拍、用 `strikeNumber` 排序。

**欄位與語意**：
| 欄位 | 意義 / 重要發現 |
|---|---|
| rally_uid | 回合 id（**官方說已打亂、數值無時序意義**）|
| match | 場次（**train/test 完全不重疊**；只用於 GroupKFold，不當特徵）|
| numberGame | 第幾局 |
| strikeNumber | 第幾拍（1=發球）|
| gamePlayerId / gamePlayerOtherId | **該拍擊球者 / 對手 的 id（選手嚴格交替！每拍交替）** |
| scoreSelf / scoreOther | **「當前擊球者視角」的比分——會隨擊球者每拍 swap**（見範例）。score_sum 在 rally 內固定 |
| serverGetPoint | 發球方是否得分（**test 已移除**，是目標之一）|
| strikeId | 值∈{1,2,4}（3類）|
| handId | 0/1/2（**實測=正手/反手 per-stroke，不是左右持拍手**；同選手 rally 內 72% 會變）|
| strengthId | 0-3（力量）|
| spinId | 0-5（旋轉）|
| pointId | **0-9：1-9=九宮格落點、0=終結/無落點**（目標）|
| actionId | **0-18：球種**，宣傳分四大類(攻擊/控制/防守/發球)，15-18 幾乎只在發球(stroke1)（目標）|
| positionId | 0-3 |
| sex | 1/2 |

**完整 train rally 範例**（注意 score 隨擊球者翻轉、最後一拍 pointId=0 終結、長度8偶數→server 得分）：
```
strikeNumber gamePlayerId scoreSelf scoreOther strengthId spinId pointId actionId  serverGetPoint
 1  2  4 3  2 5  6 15  1
 2  1  3 4  2 2  6 10  1   <- scoreSelf/Other 相對上一拍 swap 了
 3  2  4 3  2 2  8 10  1
 ...
 8  1  3 4  0 0  0 13  1   <- pointId=0 終結拍
```
**test rally 範例（無 serverGetPoint，被截斷）**：
```
strikeNumber gamePlayerId scoreSelf scoreOther spinId pointId actionId
 1 16 6 4 5 1 17
 2 15 4 6 4 8 4
 3 16 6 4 1 8 6
 4 15 4 6 3 8 13
```
這個 test rally 給你前 4 拍、要你預測「第 5 拍」的 actionId/pointId 與整段 serverGetPoint。

**長度分布**：train 中位數 5、平均 5.65；**test 中位數 2、平均 3.07，且 508/1845 (28%) 只有 1 拍（只看到發球）**。
官方：test 是「隨機刪除若干連續拍」的截斷，第 n 拍不一定是最後一拍。

**選手重疊（關鍵）**：train 166 位、test 71 位，**40 位 test 選手見過、31 位沒見過**；rally 層級 **44% 兩位都見過、49.5% 一位見過、僅 6.1% 兩位都沒見過**。next_hitter = gamePlayerOtherId（test 已知，無洩漏）。
**test 同場同局可依 score 排序(93.8% 唯一)，但只有 2.4% 連續、1% 有「下一回合」在 test 裡** → 跨回合重構不可行。

**next-shot（stroke≥2）真值分布**：
- next actionId：`{1:.221, 10:.161, 13:.113, 6:.095, 2:.091, 12:.065, 5:.06, 11:.051,...}`（16 類出現）
- next pointId：`{9:.231, 0:.219, 8:.178, 7:.131, 5:.094, 6:.066, 4:.043, 2:.028, 1:.008, 3:.003}`
- serverGetPoint base rate 0.55；**rally 總長偶數→server 得分機率 0.999、奇數→0.001**（嚴格交替的必然），但 test 看不到最終長度。

---

## 2. 任務與評分
預測每 rally：下一拍 `actionId`(19類, Macro-F1)、`pointId`(10類, Macro-F1)、`serverGetPoint`(機率, AUC)。
`Final = 0.4·F1a + 0.4·F1p + 0.2·AUC`。Macro-F1 含 class 0。**提交只回傳「你的分數+排名」，無逐筆回饋**（每天3次、計最後一次上傳；最終看 6/3 的 Private LB）。

---

## 3. 驗證框架（我們的量尺，請沿用此語言）
- **CV-A** = GroupKFold(5) by match（整場留出）= private/cold-start proxy。
- **CV-B** = 0.94·(seen-player 子集 macro-F1) + 0.06·(unseen 子集)（test 94% 是已見選手）= **公開榜 proxy，已驗證：v2 CV-B 0.348 ≈ 實際公開榜 0.351**。
- **CV-C** = unseen 子集表現 = private 風險檢查。
- 樣本：每 rally 切「前綴(1..L)→下一拍(L+1)」，**保留終結拍**；LGBM 用 all-prefix(70k) 訓練、TabPFN/驗證用「比照 test 長度抽樣」的單一前綴(15k)。

---

## 4. 我們目前的完整方法（v2/v3）

**特徵（給 LGBM/TabPFN，皆 fold-safe）**：
- base：最後一拍 + lag2 + lag3 的 (strikeId,handId,strengthId,spinId,pointId,actionId,positionId)；scoreSelf/Other/diff/sum；obs_len；parity；next_is_server；mean_spin；mean_strength；nuniq_point/action。
- **transition**（核心有效）：`P(next_action | last_action,last_point)`(19維) + `P(next_point | last_action,last_point)`(10維)，additive smoothing。
- **player-marginal**（核心有效，唯一可轉移的選手訊號）：`P(next_action | next_hitter)`(19維) + `P(next_point | next_hitter)`(10維)。
- **matchup**(v3)：`P(next_action | next_hitter, opponent_cluster)`，opponent 用 KMeans(k=6) 對「選手球種+落點分布(29維)」分群。

**模型 / ensemble**：
- LightGBM(400 trees, num_leaves 63, lr 0.05, class_weight=balanced) ×3 任務。
- TabPFN(v3 foundation model) ×3；action 19類用 ManyClassClassifier 包裝。
- 小型 GRU(hidden 64, 多任務 CE 0.4/0.4/0.2)。
- **per-task 3-way 權重搜尋**（在 CV-B 上挑）：action≈(LGBM .4, TabPFN .2, GRU .4)、point≈(.5,.3,.2)、server≈(.2,.8,0)。
- **決策(macro-F1)**：prior-correction `pred=argmax_c p_c / prior_c^β`，β 在 CV 上挑。
- serverGetPoint 輸出機率。

---

## 5. 完整實驗結果（CV-B = 公開榜 proxy；判定基準）

| 方向 | ΔCV-B vs 對應基準 | cold-start | 判定 |
|---|---|---|---|
| base 特徵 | — | — | baseline |
| transition (action/point joint) | action F1 +0.01~0.015 | 安全 | ✅ 採用 |
| **player-marginal 球種傾向** | **大贏(公開榜 0.319→0.351)** | 安全 | ✅ 採用(主突破) |
| TabPFN ⊕ GRU ensemble | +0.01 | 安全 | ✅ 採用 |
| **matchup 風格分群** | **+0.002** | 安全 | ✅ 採用(v3) |
| raw player ID (LGBM/GRU emb) | ≤+0.001 / 多傷 cold-start | 傷 | ❌ |
| player×state P(a\|player,last) | seen 變差 | 傷 | ❌ |
| in-match transductive 畫像 | −0.08 | — | ❌(稀疏過擬合) |
| 選手發球得分率→server | AUC −0.016(連 seen) | — | ❌(server=parity 決定) |
| per-class threshold/bias 最佳化 | CV-A +0.007 但 **CV-B +0.000** | — | ❌(CV-A 過擬合) |
| score 視角修正 + pressure phase | 更差 | — | ❌(模型用 parity 已能還原) |
| inter-rally 時間線重構 | test 僅 1% 可重建 | — | ❌(稀疏) |
| 點位幾何 / two-stage / row×col 分解 | ≤0 | — | ❌ |
| stage-specific transition | −0.002 | — | ❌ |
| lag4-5 / first-stroke / window / role | 淨傷(尤其拉低 AUC) | — | ❌ |
| 更大/雙向/attention GRU、多seed | ensemble 無提升 | — | ❌ |
| **ShuttleNet-lite (GRU+共享Player_GRU+gated+hazard)** | **+0.0003** | 略安全 | ❌(<門檻) |
| **完整 ShuttleNet (Transformer+position-aware+dual-context style attn)** | **−0.0008** | 持平 | ❌(連 SOTA 也輸 plain GRU) |

**目前最佳**：v2 公開榜 0.3511；v3(matchup) CV-B 0.350(≈公開榜 ~0.353)。

---

## 6. 關鍵診斷（解釋為何卡住）

- **記憶 vs 泛化落差(最重要)**：把模型在全部 train 訓練、又在同一批 train 預測(in-sample)：action macro-F1 **0.965**、point **0.921**、serverAUC **0.993**。但跨場次 CV：action ~0.30、point ~0.18、serverAUC ~0.61。→ **不是標籤雜訊、不是模型容量不足，是「跨場次泛化」**。
- **seen vs unseen**：action F1 已見選手 **0.356** vs 未見 **0.190**（巨大）。test 94% 是已見 → 公開榜≈已見表現。
- **point 逐類別 F1**（連常見類都低）：`{0:.40, 9:.27, 8:.19, 7:.23, 6:.28, 5:.16, 4:.09, 3:.00, 2:.24, 1:.08}` → point 是最大短板(0.19)、權重 0.4。
- **next-shot 條件可預測性**（Markov, train）：next action | (last_action,last_point) 準度 ~0.46；next point ~0.33。
- **server**：由「最終 rally 長度奇偶」決定；從截斷前綴看不到最終長度 → AUC 卡 0.61。

---

## 7. 我的結論與想法
1. **瓶頸是「同選手換對手/場次打法會變、規律搬不過去」的跨場次泛化**——這是資料本質。可轉移的訊號(選手平均球種傾向、Markov transition、對手風格群)我們已抽取；更細的選手/序列建模都在過擬合不可轉移的部分。
2. **連這題的學界 SOTA(ShuttleNet, 完整 Transformer 版)都贏不過簡單 GRU**——因為 test 序列極短(中位 2)、且泛化天花板卡死。
3. 我判斷誠實天花板 ~0.35；前30(0.44+)可能靠「以公開榜分數選模型」過擬合公開子集，private(6/3)會洗牌——但**我無法確定，他們也可能找到我們沒找到的東西**。

---

## 8. 給 ChatGPT / Gemini 的問題（請務實、可被 CV-B 檢驗）
1. **掌握這份資料的全新方式**：給定上面完整的資料語意與診斷，有沒有「機制上不同、我們第 5 節沒試過」的**表示或建模**方式，能在 **CV-B 漲、CV-C 不崩**？（不是再調模型容量——已證無效。）
2. **跨場次泛化**：在「短截斷前綴 + 已見選手但新對手/新場次」的設定下，有沒有方法抽出更多**可轉移**訊號？（例如：對手風格的更好表示、選手「相對」風格、stroke 之間的關係特徵…）
3. **point 0.19（最大短板，權重 0.4）**：在不重構 label、不靠 inter-rally、不靠已否決幾何的前提下，**任何**能在 CV-B 漲 point 的 contextual 訊號？還是 point 已是本質上限？
4. **server 0.61**：parity 機制下，有沒有能從截斷前綴預測「剩餘拍數奇偶/還會打幾拍」的真實訊號？
5. **你們是否同意天花板 ~0.35**？若不同意，請給**唯一一個**最值得做、且能說明「為何 CV-B 會漲」的方向。

請直接、務實。我需要的是**經得起 CV-B（公開榜 proxy）檢驗的新想法**，或**誠實的天花板共識**。
