# AI CUP 2026 桌球競賽：重新評估 Player ID / 選手資訊是否應納入模型

> 給 Claude Code 的任務說明：  
> 目前我們需要重新檢查一個重要假設：**是否過度排除了 player identity / 選手資訊**。  
> 之前我們擔心 `gamePlayerId` / `gamePlayerOtherId` 會造成 overfitting，因此沒有納入模型。  
> 但現在使用者提出合理懷疑：如果 train/test 有部分選手重疊，完全排除 player id 可能也會丟掉重要訊號。

---

## 0. 當前結論先修正

之前的策略是：

```text
不要使用 player id，避免 overfit。
```

現在要修正成：

```text
不要盲目使用 raw player id；
但也不要完全排除 player identity。
```

更精準的判斷是：

```text
Player identity 可能對部分任務有用，尤其是 serverGetPoint。
但它必須經過嚴格消融、seen/unseen player 診斷、fold-safe encoding、smoothing/backoff。
```

因此這份任務不是要你直接把 player id 加進 final，而是要建立一套完整實驗，回答：

```text
1. Player identity 是否真的能提升 OOF score？
2. 它是提升所有玩家，還是只提升 seen players？
3. 它是否傷害 unseen players？
4. 它是否只適合用在 serverGetPoint，不適合 actionId / pointId？
5. 它是否能幫助目前卡住的 Overall ≈ 0.3038？
```

---

## 1. 競賽與資料語意回顧

本競賽是桌球 rally 時序資料預測：

```text
給定一個 rally 已觀測的前 n-1 拍，
預測下一拍 actionId、下一拍 pointId，以及該 rally 發球方是否得分。
```

三個任務：

```text
actionId:
    下一拍球種
    19 類，0–18
    Macro F1
    權重 0.4

pointId:
    下一拍落點
    10 類，0–9
    1–9 是九宮格，0 是終結 / 無落點 / 特殊狀態
    Macro F1
    權重 0.4

serverGetPoint:
    該 rally 最終是否由發球方得分
    binary probability
    AUC-ROC
    權重 0.2
```

總分：

```text
Overall = 0.4 * action_macro_f1
        + 0.4 * point_macro_f1
        + 0.2 * server_auc
```

---

## 2. 目前 pipeline 實際使用了哪些資訊

目前 `src/main.ipynb` 的 base features 大致使用：

```text
sex
obs_len
obs_parity
next_is_server

last / lag2 / lag3:
    strikeId
    handId
    strengthId
    spinId
    pointId
    actionId
    positionId

score:
    scoreSelf
    scoreOther
    score_diff
    score_sum

prefix aggregation:
    mean_spin
    mean_strength
    nuniq_point
    nuniq_action

fold-safe transition:
    P(next_action | last_action, last_point)
    P(next_point  | last_action, last_point)
```

目前沒有使用：

```text
gamePlayerId
gamePlayerOtherId
match as feature
rally_uid as feature
rally_id
numberGame
raw strikeNumber as feature
player-pair matchup
player-specific behavior
player-specific server / receiver tendency
```

`match` 只用於 GroupKFold，這是正確的；`rally_uid` 只用於分組與 submission，也正確。

但目前完全沒有使用 `gamePlayerId` / `gamePlayerOtherId`，這可能過度保守。

---

## 3. 為什麼 Player ID 可能有用

桌球不是純隨機下一拍預測，不同選手會有不同傾向：

```text
某些選手偏攻擊
某些選手擅長接發
某些選手發球得分率高
某些選手常打某些落點
某些選手對特定球種的回應模式不同
```

因此 player identity 可能幫助：

```text
actionId:
    選手常用球種 / 回球習慣

pointId:
    選手常打落點 / 站位偏好

serverGetPoint:
    發球方得分率 / 選手實力 / 接發能力
```

其中最可能有幫助的是：

```text
serverGetPoint
```

因為這個任務預測的是整個 rally 的勝負，選手能力與對戰關係可能比單拍特徵更重要。

---

## 4. 為什麼 Player ID 也有風險

不能直接把 player id 當成萬能特徵，因為它可能 overfit：

```text
模型記住 seen players 的習慣，
但對 unseen players 泛化失敗。
```

尤其現在 validation 是：

```text
GroupKFold by match
```

它能避免同一場 match 洩漏，但不能保證 player 不重疊。

也就是說，可能發生：

```text
某個 player 出現在 fold_train 的 match A
同一個 player 出現在 fold_valid 的 match B
```

這種情況下，player id feature 在 CV 中看起來有效，但 test 中遇到 unseen player 時可能失效。

因此不能只看 overall OOF，還必須看：

```text
seen-player metrics
unseen-player metrics
```

---

## 5. 第一階段：先做 Player overlap / cold-start 診斷

在做任何 player feature 前，請先輸出以下資料診斷。

### 5.1 全資料 train/test overlap

請計算：

```text
train gamePlayerId unique count
test gamePlayerId unique count
gamePlayerId overlap count
test gamePlayerId seen ratio
test gamePlayerId unseen ratio

train gamePlayerOtherId unique count
test gamePlayerOtherId unique count
gamePlayerOtherId overlap count
test gamePlayerOtherId seen ratio
test gamePlayerOtherId unseen ratio
```

再計算 player pair：

```text
player_pair = unordered or ordered pair(gamePlayerId, gamePlayerOtherId)

train player_pair unique count
test player_pair unique count
player_pair overlap count
test pair seen ratio
test pair unseen ratio
```

請同時輸出：

```text
test rallies where:
    both players seen in train
    only current player seen
    only other player seen
    both players unseen
```

---

### 5.2 Fold 內 seen/unseen 診斷

每個 GroupKFold fold 都要輸出：

```text
fold_train unique players
fold_valid unique players
fold_valid players seen in fold_train
fold_valid players unseen in fold_train
fold_valid seen-player rally ratio
fold_valid unseen-player rally ratio
```

針對 valid samples，建立 mask：

```text
current_player_seen_in_fold_train
other_player_seen_in_fold_train
both_players_seen
any_player_unseen
both_players_unseen
```

---

### 5.3 基準模型的 seen/unseen metrics

在目前 best baseline 或 current final model 上，請輸出：

```text
overall valid metrics
seen-player subset metrics
unseen-player subset metrics
both-seen subset metrics
any-unseen subset metrics
both-unseen subset metrics
```

指標：

```text
action_macro_f1
point_macro_f1
server_auc
overall
support count
```

這可以判斷目前模型是否本來就對 unseen player 表現很差。

---

## 6. 第二階段：Player features 消融實驗

請不要一次把所有 player features 加進 final。要分層測。

目前 current best baseline 是：

```text
Overall ≈ 0.3038
F1_action ≈ 0.284
F1_point ≈ 0.175
AUC_server ≈ 0.601
fold std ≈ 0.007
```

所有 player 實驗都要和這個 baseline 比較。

---

# P1：Raw Player ID Categorical Feature

## 6.1 目的

測試 raw player identity 是否本身有訊號。

## 6.2 特徵

加入：

```text
last_gamePlayerId
last_gamePlayerOtherId
lag2_gamePlayerId
lag2_gamePlayerOtherId
lag3_gamePlayerId
lag3_gamePlayerOtherId
```

如果資料中每一拍的 player 欄位不會變，也可直接用 rally-level：

```text
gamePlayerId
gamePlayerOtherId
```

## 6.3 注意事項

### 對 LightGBM

player id 不能被當成連續數字。

必須：

```text
1. consistent label encoding
2. unknown player = -1
3. 指定 categorical_feature
```

不要讓模型把：

```text
playerId 100 > playerId 20
```

這種無意義大小關係當成連續數值。

### 對 TabPFN

先不要把 raw player id 放進 TabPFN。

原因：

```text
TabPFN 對 high-cardinality raw ID 未必友善
它可能把 ID 當數值關係
```

P1 先只測 LGBM。

## 6.4 輸出

比較：

```text
baseline LGBM / ensemble
baseline + raw player categorical LGBM
```

輸出：

```text
action_macro_f1
point_macro_f1
server_auc
overall
fold mean/std
seen-player metrics
unseen-player metrics
```

如果 raw ID 只提升 seen players 但傷害 unseen players，不能直接進 final。

---

# P2：Player Seen / Count / Frequency Features

## 7.1 目的

這是低風險 player feature，不直接記住打法，而是告訴模型：

```text
這個 player 是否在 train 出現過？
出現過多少次？
這組 player pair 是否出現過？
```

## 7.2 特徵

加入：

```text
current_player_seen_in_train
other_player_seen_in_train
both_players_seen_in_train
any_player_unseen_in_train

current_player_train_count
other_player_train_count
min_player_train_count
max_player_train_count
sum_player_train_count

player_pair_seen_in_train
player_pair_train_count
```

在 CV 中必須 fold-safe：

```text
fold_valid 的 seen/count 只能根據 fold_train 計算。
```

test 才能用 full train 計算。

## 7.3 預期用途

這些特徵可能幫助：

```text
serverGetPoint probability calibration
uncertainty estimation
seen/unseen player behavior adjustment
```

但不一定會提升 action/point。

## 7.4 輸出

請比較：

```text
baseline
baseline + P2
```

並輸出 seen/unseen subset metrics。

---

# P3：Fold-safe Player Behavior Encoding

## 8.1 目的

測試選手歷史行為是否有用，但避免 leakage。

不是直接記住 player id，而是統計：

```text
這個 player 過去常打什麼球？
常打哪裡？
發球時是否容易得分？
接發時是否容易失分？
```

## 8.2 基礎 player priors

建立 fold-safe encoding：

```text
P(next_action | current_player)
P(next_point  | current_player)
P(serverGetPoint | current_player)

P(next_action | other_player)
P(next_point  | other_player)
P(serverGetPoint | other_player)
```

其中 `current_player` 和 `other_player` 的語意要明確定義。若資料的 `gamePlayerId` 表示當前擊球者或某一方球員，需要先檢查欄位語意。如果每一拍不同，請以最後一拍 / 下一拍角色推定建立。

---

## 8.3 role-specific player priors

如果能定義 server / receiver role，建立：

```text
P(serverGetPoint | server_player)
P(serverGetPoint | receiver_player)

P(next_action | next_hitter_player)
P(next_point  | next_hitter_player)
```

如果目前只能粗略由 `obs_len` parity 推下一拍是 server/receiver，也請先建立可檢查版本，不要硬假設一定正確。

---

## 8.4 player + state interaction

建立：

```text
P(next_action | current_player, last_action, last_point)
P(next_point  | current_player, last_action, last_point)
```

也可測：

```text
P(next_action | next_hitter_player, last_action, last_point)
P(next_point  | next_hitter_player, last_action, last_point)
```

但這類條件很稀疏，必須 smoothing/backoff。

---

## 8.5 Smoothing / backoff

所有 player behavior encoding 必須有 backoff：

```text
(player, last_action, last_point)
    -> (player, last_action)
    -> (player)
    -> global prior
```

或者：

```text
empirical_prob = count_class / total_count

smoothed_prob = (count_class + alpha * global_prior_class)
              / (total_count + alpha)
```

alpha 請測：

```text
alpha = 5, 10, 20, 50
```

如果 player_count 太少，直接 fallback 到 global。

---

## 8.6 fold-safe 要求

這點不能妥協。

每個 fold：

```text
fold_train:
    統計 player priors / encodings

fold_valid:
    只能使用 fold_train 統計出的 encoding
    不可看 fold_valid labels
```

final test：

```text
用 full train 統計 encoding
套用到 test
```

---

## 8.7 實驗拆分

請不要一次全部加，拆成：

```text
P3a: player prior only
    P(next_action | player)
    P(next_point | player)
    P(serverGetPoint | player)

P3b: player + last_action,last_point
    P(next_action | player,last_action,last_point)
    P(next_point | player,last_action,last_point)

P3c: server-only player encoding
    只把 player features 加入 serverGetPoint model
    actionId / pointId 不加

P3d: action/point-only player encoding
    只加到 action/point model
    server 不加
```

---

# 9. 第三階段：只在適合任務使用 player features

不要假設 player features 對三個任務都要使用。

可能結果：

```text
1. player features 只提升 serverGetPoint AUC
2. player features 提升 actionId，但傷害 pointId
3. player features 對 pointId 無效
4. player features 對 unseen players 傷害大
```

因此 final 可以是 task-specific：

```text
action model:
    no player features
    or only weak player behavior

point model:
    no player features
    or only geometry + current best

server model:
    player seen/count + player server prior
```

請允許這種不對稱設計。

---

## 10. 評估要求

每個 player 實驗都要輸出：

```text
official:
    action_macro_f1
    point_macro_f1
    server_auc
    overall

stability:
    fold mean
    fold std
    delta vs current best

seen/unseen:
    seen current player metrics
    unseen current player metrics
    both players seen metrics
    any player unseen metrics
    both players unseen metrics

per-task:
    action classification report
    point classification report
    server probability diagnostics
```

如果 player features 對 seen players 提升，但 unseen players 明顯下降，請標記：

```text
high overfit risk
```

---

## 11. 決策規則

Player features 可以納入 final 的條件：

```text
1. OOF overall 提升
2. fold std 沒明顯變大
3. unseen-player subset 沒有明顯下降
4. test player seen ratio 足夠支持使用
5. submission distribution 沒有異常
```

如果只對 serverGetPoint 有用：

```text
只納入 server model。
```

如果只對 seen players 有用，但 unseen players 傷害大：

```text
不要納入 final
或建立 gating：
    seen player -> use player-enhanced model
    unseen player -> use base model
```

---

## 12. Gating 版本：seen/unseen 混合模型

如果 P3 對 seen players 有效、對 unseen players 無效，可以嘗試 gating。

建立兩套 prediction：

```text
base_model_prob
player_model_prob
```

根據 test/fold sample 是否 seen：

```text
if both_players_seen:
    final_prob = w * player_model_prob + (1-w) * base_model_prob
elif one_player_seen:
    final_prob = w_small * player_model_prob + (1-w_small) * base_model_prob
else:
    final_prob = base_model_prob
```

權重用 OOF 搜尋：

```text
both_seen w ∈ {0.25, 0.5, 0.75, 1.0}
one_seen w ∈ {0.1, 0.25, 0.5}
```

這樣可以避免 player features 傷害 cold-start samples。

---

## 13. 立即執行順序

請依序執行：

```text
[ ] D0: player overlap diagnostics
[ ] D1: baseline seen/unseen metrics
[ ] P2: player seen/count/frequency features
[ ] P3c: server-only player encoding
[ ] P3a: player prior encoding for all tasks
[ ] P3b: player + state interaction encoding
[ ] P1: raw player categorical LGBM
[ ] G1: seen/unseen gating ensemble, only if P3 improves seen but hurts unseen
```

注意順序：

```text
先測低風險 P2 / P3c。
最後才測 raw player categorical P1。
```

---

## 14. 為什麼 raw player ID 放最後

Raw player ID 最容易 overfit，且對 TabPFN 不友善。

因此：

```text
不要先把 raw player ID 加進最終 pipeline。
先用 fold-safe behavior encoding 檢查 player identity 是否真的有訊號。
```

如果 P2/P3 都無效，raw player ID 多半也不值得冒險。

---

## 15. 最終輸出格式

請更新或新增一份表格：

```text
player_feature_ablation.csv
```

欄位：

```text
experiment_name
features
used_for_action
used_for_point
used_for_server
action_macro_f1
point_macro_f1
server_auc
overall
fold_std
seen_action_f1
unseen_action_f1
seen_point_f1
unseen_point_f1
seen_server_auc
unseen_server_auc
decision
notes
```

並在 console 輸出 summary：

```text
Best without player:
    overall / action / point / server

Best with player:
    overall / action / point / server

Player feature conclusion:
    useful for action? yes/no
    useful for point? yes/no
    useful for server? yes/no
    overfit risk? low/medium/high
    final decision
```

---

## 16. 最終目標

這一輪要回答：

```text
我們是否誤判了 player id 不該被訓練？
```

更精準的答案應該從實驗得出：

```text
A. 完全無用：維持目前不使用 player features
B. 只對 server 有用：只加入 server model
C. 對 seen players 有用但 cold-start 傷害：使用 seen/unseen gating
D. 對所有 subset 都穩定有效：納入 final
E. 只在 CV 有效但 unseen 崩壞：排除，避免 overfit
```

不要用直覺決定。請用 OOF、seen/unseen diagnostics、fold stability 決定。
