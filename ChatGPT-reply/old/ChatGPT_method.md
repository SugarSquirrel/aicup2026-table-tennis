````markdown
# AI CUP 桌球戰術與結果預測競賽：最大化預測能力的建議實作規格

## 0. 任務背景與核心目標

本競賽目標是根據每個 Rally 已觀測到的前 n-1 拍資料，預測：

1. `actionId`：下一拍球種
2. `pointId`：下一拍落點，九宮格分類
3. `serverGetPoint`：該 Rally 最終是否由發球者得分，輸出 0~1 機率值

官方評分方式為：

```text
Overall Score = 0.4 * actionId Macro F1
              + 0.4 * pointId Macro F1
              + 0.2 * serverGetPoint AUC-ROC
````

因此模型優先順序應該是：

```text
actionId 預測能力 ≈ pointId 預測能力 > serverGetPoint 預測能力
```

也就是說，主要優化目標不是 accuracy，而是：

* `actionId` 的 Macro F1
* `pointId` 的 Macro F1
* `serverGetPoint` 的 AUC-ROC

官方說明指出，此任務重點是讓模型理解擊球序列脈絡，並利用前 n-1 拍預測下一拍球種、落點與回合勝負；其中 `actionId` 與 `pointId` 採用 Macro F1，是因為球種與落點類別存在不平衡問題。([AIdea][1])

---

# 1. 最推薦的總體架構

## 1.1 最終建議

不要只做單一模型，也不要完全改成深度學習模型。

最推薦架構是：

```text
LightGBM 主模型
+ TabPFN 輔助模型
+ 序列特徵工程
+ Transition probability features
+ 合法類別 post-processing
+ OOF validation-based ensemble
```

整體流程：

```text
train.csv / test_new.csv
        ↓
依 rally_uid 分組，依 strikeNumber 排序
        ↓
建立 prefix samples：
    用前 L 拍預測第 L+1 拍
        ↓
特徵工程：
    lag features
    first-stroke features
    recent-window statistics
    transition probability features
    player/server behavior features
        ↓
三個任務分開建模：
    actionId model
    pointId model
    serverGetPoint model
        ↓
LightGBM 作為主模型
TabPFN 作為輔助模型
        ↓
OOF-based weighted ensemble
        ↓
post-processing：
    actionId 合法類別 mask
    Macro F1 class prior correction
        ↓
產生 submission.csv
```

---

# 2. 資料切分與樣本建立策略

## 2.1 Prefix sample 設計

對於每個 rally：

```text
rally = [shot_1, shot_2, shot_3, ..., shot_T]
```

建立訓練樣本：

```text
prefix = shot_1
target = shot_2

prefix = shot_1, shot_2
target = shot_3

prefix = shot_1, shot_2, shot_3
target = shot_4

...

prefix = shot_1 ... shot_{T-1}
target = shot_T
```

也就是：

```text
用前 L 拍預測第 L+1 拍
```

這樣可以最大化使用資料量。

---

## 2.2 LightGBM 使用 all-prefix training

LightGBM 可以吃完整的 all-prefix samples。

建議：

```text
每個 rally 產生所有 prefix samples
用於訓練 LightGBM
```

理由：

* LightGBM 訓練快
* 可以承受較多樣本
* 對 tabular feature engineering 很穩定
* 能有效利用全部序列資訊

---

## 2.3 TabPFN 使用 sampled-prefix training

TabPFN 不建議吃所有 prefix samples。

建議：

```text
每個 rally 抽 1~3 個 prefix samples
或者依照 test_new.csv 的 obs_len 分布抽樣
```

理由：

* TabPFN 適合小型到中型表格資料
* 全量 prefix samples 可能太多
* TabPFN 應作為輔助模型，不應作為主模型

---

# 3. Validation 設計

## 3.1 不要 random split

禁止使用 row-level random split。

錯誤做法：

```text
train rows 隨機切 80%
valid rows 隨機切 20%
```

原因：

同一個 rally 或同一場 match 內的資料高度相關，隨機切會造成資料洩漏，validation score 會虛高。

---

## 3.2 推薦使用 GroupKFold by match

目前附件資料中，train/test 的 `match` 不重疊。

因此：

```text
match 可以用來做 GroupKFold
但不應該直接當作模型特徵
```

推薦：

```text
GroupKFold(n_splits=5, groups=train["match"])
```

目的：

* 模擬 test 是未見過 match 的情況
* 避免同一場比賽資料同時出現在 train/valid
* 讓 validation score 更接近 private leaderboard

---

## 3.3 rally_uid 的使用原則

`rally_uid` 可以用來分組、產生 prefix samples、對應 submission。

但不要使用 `rally_uid` 的數值大小推論時間順序。

官方公告指出 `rally_uid` 是隨機打亂編號，數值連續與否不代表實際回合順序。([AIdea][1])

---

# 4. 特徵工程設計

這是整個方案最重要的部分。

模型要強，不是單純換模型，而是要把桌球序列的戰術資訊轉成 tabular features。

---

# 4.1 基礎特徵

每個 prefix sample 至少保留：

```text
obs_len
sex
score
server / receiver 相關欄位
目前 prefix 長度
目前 rally 已觀測拍數
```

如果有以下欄位，也應納入：

```text
gamePlayerId
gamePlayerOtherId
side
handId
positionId
strengthId
spinId
actionId
pointId
```

但注意：

```text
match 不建議當作特徵
rally_uid 不建議當作特徵
serverGetPoint 不可當作輸入特徵
```

---

# 4.2 Lag features

目前 notebook 只有 last、lag2、lag3。

建議擴充到：

```text
lag_1 ~ lag_5
```

對每個 lag k，取前綴中倒數第 k 拍的資訊：

```text
actionId_lag{k}
pointId_lag{k}
spinId_lag{k}
strengthId_lag{k}
handId_lag{k}
positionId_lag{k}
player_lag{k}
side_lag{k}
```

例如：

```text
last_actionId
lag2_actionId
lag3_actionId
lag4_actionId
lag5_actionId
```

若 prefix 長度不足，填入固定缺失值，例如 `-1`。

---

# 4.3 First-stroke features

第一拍通常是發球，對後續戰術影響很大。

應加入：

```text
first_actionId
first_pointId
first_spinId
first_strengthId
first_positionId
first_handId
first_player
first_side
```

直覺：

```text
短發球 + 下旋
可能導致下一拍擺短、搓球、控制落點

長發球 + 強旋轉
可能導致對方直接攻擊或被動回球
```

所以第一拍特徵對 `actionId`、`pointId`、`serverGetPoint` 都有幫助。

---

# 4.4 Recent-window statistics

除了單拍 lag，還要統計最近 k 拍的局部型態。

建議 window：

```text
last3
last5
```

對最近 3 拍、5 拍計算：

```text
last3_mean_strength
last3_std_strength
last3_mean_spin
last3_nunique_action
last3_nunique_point
last3_point0_count
last3_unique_player_count

last5_mean_strength
last5_std_strength
last5_mean_spin
last5_nunique_action
last5_nunique_point
last5_point0_count
last5_unique_player_count
```

如果 actionId 可以區分攻擊、防守、發球、控制類型，也可以建立：

```text
last3_attack_count
last3_defense_count
last3_serve_like_count
last3_control_count

last5_attack_count
last5_defense_count
last5_serve_like_count
last5_control_count
```

這部分可以幫助模型理解：

```text
最近是否進入攻防轉換
最近是否連續控制
是否已接近該 rally 的終結狀態
```

---

# 4.5 Rally-level cumulative statistics

針對目前 prefix 內所有已觀測拍數計算：

```text
rally_len_so_far
mean_strength_so_far
std_strength_so_far
mean_spin_so_far
nunique_action_so_far
nunique_point_so_far
point0_count_so_far
action_entropy_so_far
point_entropy_so_far
```

可再加上：

```text
server_shot_count_so_far
receiver_shot_count_so_far
server_attack_count_so_far
receiver_attack_count_so_far
server_point_distribution_so_far
receiver_point_distribution_so_far
```

這些特徵對 `serverGetPoint` 特別重要，因為勝負比較依賴整段 rally 的累積局勢，而不是單一拍。

---

# 4.6 Transition probability features

這是最重要的新增特徵。

## 4.6.1 核心想法

桌球下一拍不是隨機發生，而是具有狀態轉移結構：

```text
上一拍球種 + 上一拍落點
        ↓
下一拍球種 / 下一拍落點
```

因此應從 train 中統計條件機率：

```text
P(next_actionId | last_actionId)
P(next_actionId | last_pointId)
P(next_actionId | last_actionId, last_pointId)

P(next_pointId | last_actionId)
P(next_pointId | last_pointId)
P(next_pointId | last_actionId, last_pointId)

P(serverGetPoint | last_actionId, last_pointId)
```

再把這些機率當作特徵。

---

## 4.6.2 actionId transition features

針對 actionId 建立：

```text
trans_next_action_prob_by_last_action_{class}
trans_next_action_prob_by_last_point_{class}
trans_next_action_prob_by_last_action_point_{class}
```

例如如果 actionId 有 0~18，共 19 類，就建立：

```text
trans_next_action_prob_by_last_action_0
trans_next_action_prob_by_last_action_1
...
trans_next_action_prob_by_last_action_18
```

---

## 4.6.3 pointId transition features

針對 pointId 建立：

```text
trans_next_point_prob_by_last_action_{class}
trans_next_point_prob_by_last_point_{class}
trans_next_point_prob_by_last_action_point_{class}
```

例如 pointId 是 0~9 或實際資料中的所有類別，就建立對應類別的條件機率。

---

## 4.6.4 serverGetPoint transition features

針對勝負建立：

```text
trans_server_win_prob_by_last_action
trans_server_win_prob_by_last_point
trans_server_win_prob_by_last_action_point
trans_server_win_prob_by_first_action
trans_server_win_prob_by_first_action_first_point
```

這些特徵可幫助 binary model 預測發球方是否得分。

---

## 4.6.5 防止 transition feature leakage

這點非常重要。

在 cross validation 中，不能用整份 train 統計 transition probability 後再切 train/valid。

錯誤做法：

```text
用完整 train.csv 統計 transition probability
再做 GroupKFold
```

這會讓 valid 的 label 洩漏進 transition features。

正確做法：

```text
每一個 fold：
    只用 fold_train 統計 transition probability
    再套用到 fold_valid

最終 test 預測：
    才可以用完整 train.csv 統計 transition probability
    再套用到 test_new.csv
```

---

# 4.7 Frequency encoding / target encoding

對高基數欄位，例如：

```text
gamePlayerId
gamePlayerOtherId
server/player 欄位
```

不建議只做 raw categorical。

建議加入：

```text
player_count
player_action_distribution
player_point_distribution
player_server_win_rate
player_receiver_win_rate
player_last_action_next_action_distribution
player_last_point_next_point_distribution
```

但要注意：

```text
所有 target encoding 都必須 fold-safe
```

也就是：

```text
valid 的 encoding 只能由 fold_train 統計而來
test 的 encoding 才能由 full train 統計而來
```

---

# 5. 三個任務的模型策略

---

# 5.1 actionId 模型

## 任務特性

`actionId` 使用 Macro F1。

這代表模型不能只偏向常見類別，否則稀有球種的 F1 會很低。

## 推薦模型

主模型：

```text
LightGBM multiclass classifier
```

輔助模型：

```text
TabPFN classifier
```

## actionId 特徵重點

優先使用：

```text
lag_1 ~ lag_5 actionId
lag_1 ~ lag_5 pointId
first_actionId
first_pointId
recent-window action statistics
transition probability features
player action tendency
```

## actionId post-processing

根據資料觀察，部分發球類 action 幾乎不應出現在「下一拍」預測中。

建議測試三種策略：

```text
策略 A：不 mask
策略 B：mask actionId 17, 18
策略 C：mask actionId 15, 16, 17, 18
```

mask 方式：

```text
將指定類別機率設為 0 或極低
再重新 normalize
最後 argmax
```

應以 OOF Macro F1 決定採用哪一種。

---

# 5.2 pointId 模型

## 任務特性

`pointId` 使用 Macro F1。

落點分類不平衡，因此不能只預測常見落點。

## 推薦模型

主模型：

```text
LightGBM multiclass classifier
```

輔助模型：

```text
TabPFN classifier
```

## pointId 特徵重點

優先使用：

```text
last_pointId
lag_2 ~ lag_5 pointId
last_actionId
first_actionId
first_pointId
recent-window point statistics
P(next_pointId | last_pointId)
P(next_pointId | last_actionId)
P(next_pointId | last_actionId, last_pointId)
player point tendency
```

## pointId=0 注意事項

不要任意移除 `pointId=0`。

`pointId=0` 很可能代表無落點、終結、特殊狀態或未落入九宮格的事件。

如果訓練資料中 `pointId=0` 會作為 target 出現，就必須保留。

---

# 5.3 serverGetPoint 模型

## 任務特性

`serverGetPoint` 使用 AUC-ROC。

官方要求輸出 0~1 機率值，不是輸出 0/1 hard label。([AIdea][1])

## 推薦模型

主模型：

```text
LightGBM binary classifier
```

輔助模型：

```text
TabPFN binary classifier
```

## serverGetPoint 特徵重點

優先使用：

```text
rally_len_so_far
first_actionId
first_pointId
last_actionId
last_pointId
server/receiver shot count
server/receiver attack count
server/receiver point tendency
recent-window statistics
transition server win probability
```

## 嚴禁事項

不要把 `serverGetPoint` 當作輸入特徵。

官方公告曾提醒舊版 test.csv 的 `serverGetPoint` 欄位存在 leakage 風險，並建議訓練時移除 `serverGetPoint` 特徵以提升泛化。([AIdea][1])

---

# 6. Ensemble 策略

## 6.1 LightGBM 為主，TabPFN 為輔

不要讓 TabPFN 和 LightGBM 等權重。

原因：

```text
LightGBM 可以吃 all-prefix samples
TabPFN 通常只能吃 sampled-prefix samples
LightGBM 對大型 tabular feature engineering 更穩
TabPFN 適合作為補充模型
```

建議初始權重：

```text
actionId:
    0.85 * LightGBM + 0.15 * TabPFN

pointId:
    0.75 * LightGBM + 0.25 * TabPFN

serverGetPoint:
    0.90 * LightGBM + 0.10 * TabPFN
```

再用 OOF validation 搜尋最佳權重。

---

## 6.2 Ensemble 權重搜尋

對每個任務分開搜尋權重。

例如：

```text
actionId weight search:
    LightGBM weight = 0.50 ~ 1.00
    TabPFN weight   = 1 - LightGBM weight
    metric = Macro F1

pointId weight search:
    metric = Macro F1

serverGetPoint weight search:
    metric = AUC-ROC
```

最後以：

```text
Overall Score = 0.4 * action_F1 + 0.4 * point_F1 + 0.2 * server_AUC
```

選擇最終權重。

---

# 7. Macro F1 專用後處理

因為 `actionId` 和 `pointId` 都是 Macro F1，模型不能只學常見類別。

建議加入 class prior correction。

---

## 7.1 類別機率校正

對於 multiclass prediction probability：

```text
p(class_k)
```

加入類別調整係數：

```text
p_adjusted(class_k) = p(class_k) * alpha_k
```

其中：

```text
稀有類別 alpha_k 稍微提高
過度預測的常見類別 alpha_k 稍微降低
```

最後重新 normalize：

```text
p_adjusted = p_adjusted / sum(p_adjusted)
```

---

## 7.2 alpha_k 的選擇方式

不要手動看 Public LB 亂調。

應使用 OOF validation 搜尋：

```text
針對 actionId：
    找出每個類別 precision / recall / F1
    對 recall 過低的類別提高 alpha
    對 FP 過多的類別降低 alpha

針對 pointId：
    同樣依 OOF confusion matrix 調整
```

目標：

```text
提升 Macro F1
而不是提升 overall accuracy
```

---

# 8. 測試集預測方式

對 `test_new.csv`：

```text
每個 rally_uid 代表一筆 submission
使用該 rally 已觀測的全部拍數作為 prefix
預測下一拍 actionId
預測下一拍 pointId
預測該 rally serverGetPoint
```

因為 `sample_submission.csv` 可能只有表頭，所以不能依賴它提供 row 數。

應以：

```text
test_new.csv 中 unique rally_uid
```

作為 submission row 數。

輸出欄位：

```text
rally_uid
actionId
pointId
serverGetPoint
```

---

# 9. 模型優先開發順序

如果時間有限，請依照以下順序實作。

---

## Priority 1：修正 validation 與 submission pipeline

必做：

```text
GroupKFold by match
OOF metric calculation
test_new.csv unique rally_uid submission generation
serverGetPoint 輸出 probability
```

如果 validation 不正確，後面所有調參都不可信。

---

## Priority 2：加入 transition probability features

最可能提升分數。

先做：

```text
P(next_actionId | last_actionId, last_pointId)
P(next_pointId | last_actionId, last_pointId)
P(serverGetPoint | last_actionId, last_pointId)
```

再擴充：

```text
P(next_actionId | last_actionId)
P(next_actionId | last_pointId)
P(next_pointId | last_actionId)
P(next_pointId | last_pointId)
```

---

## Priority 3：擴充 lag + first-stroke + recent-window features

加入：

```text
lag_1 ~ lag_5
first-stroke features
last3 statistics
last5 statistics
rally cumulative statistics
```

---

## Priority 4：Macro F1 post-processing

針對 actionId / pointId：

```text
class prior correction
rare class alpha adjustment
actionId legal mask
```

---

## Priority 5：TabPFN ensemble

在 LightGBM pipeline 穩定後，再加 TabPFN。

不要先花太多時間在 TabPFN，因為最大增益通常來自特徵工程與 validation 修正。

---

# 10. 建議做的實驗清單

請按順序記錄每次 OOF 結果。

| Experiment | Description                              | Expected Effect                        |
| ---------- | ---------------------------------------- | -------------------------------------- |
| E0         | current notebook baseline                | baseline                               |
| E1         | GroupKFold by match + correct OOF metric | 建立可信分數                                 |
| E2         | lag 1~5                                  | 提升 actionId / pointId                  |
| E3         | first-stroke features                    | 提升 actionId / pointId / serverGetPoint |
| E4         | last3 / last5 statistics                 | 提升序列理解                                 |
| E5         | transition probability features          | 高機率顯著提升                                |
| E6         | player behavior features                 | 若 player ID 有泛化性，可能提升                  |
| E7         | actionId mask                            | 降低不合理類別錯誤                              |
| E8         | class prior correction                   | 提升 Macro F1                            |
| E9         | TabPFN ensemble                          | 小幅提升穩定性                                |
| E10        | weight search                            | 最大化 final score                        |

---

# 11. 不建議做的事

## 11.1 不建議直接上深度學習主模型

例如：

```text
LSTM
GRU
Transformer
TCN
```

不是不能做，而是不建議作為目前主線。

原因：

```text
資料量不算巨大
欄位是結構化 tabular sequence
比賽時間有限
LightGBM + 強特徵工程通常更穩
深度模型需要更多 tuning，且容易 overfit
```

除非 LightGBM pipeline 已經非常完整，再考慮加一個 sequence model 當 ensemble 成員。

---

## 11.2 不建議把 match / rally_uid 當特徵

原因：

```text
match 在 train/test 不重疊，當特徵容易 overfit
rally_uid 是隨機編號，不代表時間順序
```

`match` 適合用於 GroupKFold，不適合當模型輸入。

---

## 11.3 不建議使用舊版 test leakage

官方已提醒舊版 test 的 `serverGetPoint` 存在 leakage 風險，過度使用可能造成 overfitting。([AIdea][1])

請不要：

```text
使用 test serverGetPoint 當特徵
使用外部真實影片或賽果反推答案
人工修正 submission
```

官方規則也禁止使用測試集殘留資訊反向比對真實比賽影片或紀錄取得標籤。([AIdea][1])

---

# 12. 最終推薦版本

最終模型應該長這樣：

```text
Input:
    train.csv
    test_new.csv

Feature Builder:
    all-prefix samples for LightGBM
    sampled-prefix samples for TabPFN
    lag 1~5
    first-stroke features
    last3 / last5 statistics
    rally cumulative statistics
    fold-safe transition probability
    fold-safe player behavior encoding

Validation:
    5-fold GroupKFold by match
    metrics:
        actionId Macro F1
        pointId Macro F1
        serverGetPoint AUC
        overall score

Models:
    actionId:
        LightGBM multiclass main model
        TabPFN auxiliary model
        class prior correction
        legal action mask

    pointId:
        LightGBM multiclass main model
        TabPFN auxiliary model
        class prior correction

    serverGetPoint:
        LightGBM binary main model
        TabPFN auxiliary model
        probability output

Ensemble:
    actionId:
        start with 0.85 LGBM + 0.15 TabPFN

    pointId:
        start with 0.75 LGBM + 0.25 TabPFN

    serverGetPoint:
        start with 0.90 LGBM + 0.10 TabPFN

Post-processing:
    actionId mask:
        compare no mask vs mask [17,18] vs mask [15,16,17,18]

    class prior correction:
        tune alpha_k using OOF validation

Output:
    submission.csv with:
        rally_uid
        actionId
        pointId
        serverGetPoint
```

---

# 13. Claude Code 實作時的判斷原則

請 Claude Code 優先遵守以下原則：

1. 不要重寫成複雜深度學習主線。
2. 保留 LightGBM 為主模型。
3. 先修正 validation 與 submission pipeline。
4. 所有 target encoding / transition probability 必須 fold-safe。
5. `serverGetPoint` 不可當輸入特徵。
6. `match` 只用於 GroupKFold，不當模型特徵。
7. `rally_uid` 只用於分組與 submission，不當模型特徵。
8. `actionId` 和 `pointId` 優先優化 Macro F1。
9. `serverGetPoint` 輸出機率，不要輸出 0/1。
10. 每次實驗都要輸出：

    * actionId Macro F1
    * pointId Macro F1
    * serverGetPoint AUC
    * weighted overall score
11. 最終提交前，使用完整 train.csv 重新訓練模型。
12. 最終 transition features 可以用完整 train.csv 統計後套用到 test_new.csv。
13. 最終 submission 以 `test_new.csv` 的 unique `rally_uid` 產生，不依賴空的 sample_submission row。

```


::contentReference[oaicite:6]{index=6}
```

[1]: https://www.aidea-web.tw/topic/3f9662e8-9d18-4c6a-9332-103eded3a399 "AIdea"
