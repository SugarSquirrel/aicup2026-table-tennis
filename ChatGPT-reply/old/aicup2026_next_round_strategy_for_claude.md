# AI CUP 2026 桌球時序資料競賽：下一輪提升預測能力的完整實作規格

> 給 Claude Code 的任務說明：  
> 目前既有 pipeline 是一個乾淨、可提交的 flat tabular baseline，但不能把 `OOF Overall ≈ 0.3038` 視為競賽天花板。  
> 更精確的結論應該是：**目前 flat prefix tabular representation 的天花板約 0.30**。  
> 下一輪要驗證的是：分數是否被「資料表示方式」限制，而不是被模型本身限制。

---

## 0. 競賽問題定義

本競賽是「基於時序資料之桌球戰術與結果預測競賽」。資料來自真實桌球單打比賽，每一筆 rally 是一個連續來回，rally 內包含多拍 stroke，並以 `strikeNumber` 表示擊球順序。

核心任務是：

```text
給定某個 rally 的前 n-1 拍擊球序列，
預測第 n 拍的：
1. actionId：下一拍球種
2. pointId：下一拍落點
3. serverGetPoint：該 rally 最終是否由發球方得分
```

這不是一般 tabular classification，而是：

```text
Structured Event Sequence Prediction
= 結構化事件序列預測
```

因為每一拍 stroke 是一個事件，每個事件具有多個離散與數值屬性，例如：

```text
actionId
pointId
spinId
strengthId
handId
positionId
player role
score
strikeNumber
```

模型不應只把這些欄位當成普通獨立 tabular features，而應理解：

```text
rally 是由多個 stroke 組成的序列
不同 stroke 階段有不同戰術意義
pointId 是九宮格空間位置，不是無序類別
actionId 有球種階層：攻擊 / 控制 / 防守 / 發球
serverGetPoint 是整個 rally 的結果，不是下一拍即時結果
```

---

## 1. 預測目標與評分

每個 `rally_uid` 需要輸出三個欄位：

```text
rally_uid
actionId
pointId
serverGetPoint
```

### 1.1 actionId

```text
任務：預測下一拍球種
類型：多分類
類別：0–18，共 19 類
官方分數：Macro F1
權重：0.4
```

`actionId` 對應桌球球種，簡報中球種可分成四大類：

```text
ATTACK    攻擊
CONTROL   控制
DEFENSIVE 防守
SERVE     發球
```

注意：

```text
actionId 15–18 可能主要對應發球類動作。
但是否要 mask 需要由 OOF 決定。
目前實驗顯示 legal mask 幾乎無影響，因模型本來就不預測這些類別。
```

---

### 1.2 pointId

```text
任務：預測下一拍落點
類型：多分類
類別：0–9，共 10 類
官方分數：Macro F1
權重：0.4
```

`pointId` 的語意：

```text
1–9：九宮格落點
0：終結 / 無落點 / 特殊狀態
```

非常重要：

```text
pointId=0 不可直接移除。
因為官方說下一拍不一定直接有結果，但下一拍可能是終結拍。
因此 pointId=0 是合法 target，至少在訓練與本地驗證中必須保留。
```

目前最大的問題是：

```text
pointId 不應只被視為 0–9 的無序類別。
1–9 有九宮格幾何結構。
point 1 和 point 2 的關係，比 point 1 和 point 9 更接近。
目前 flat tabular model 沒有理解這件事。
```

---

### 1.3 serverGetPoint

```text
任務：預測該 rally 最終是否由發球方得分
類型：二分類機率
官方分數：AUC-ROC
權重：0.2
```

提交時：

```text
serverGetPoint 必須是 0 到 1 的 probability
不要輸出 hard label 0/1
```

不要把 `serverGetPoint` 當成模型輸入特徵，避免 leakage。

---

### 1.4 Overall Score

每次驗證都必須計算：

```text
Overall Score = 0.4 * action_macro_f1
              + 0.4 * point_macro_f1
              + 0.2 * server_auc
```

模型選擇必須看 Overall，不要只看 action 或 point 單一任務。

---

## 2. 目前 pipeline 的狀態與限制

目前 `src/main.ipynb` 已經做到：

```text
1. prefix -> next stroke 樣本建立
2. 保留 pointId=0
3. GroupKFold by match
4. LightGBM all-prefix training
5. TabPFN sampled-prefix training
6. prior correction
7. fold-safe transition features
8. LGBM + TabPFN per-task ensemble
9. 完整 evaluation diagnostics
10. submission validation
```

目前最佳結果：

```text
OOF Overall ≈ 0.3038
F1_action ≈ 0.284
F1_point ≈ 0.175
AUC_server ≈ 0.601
fold std ≈ 0.007
```

目前消融結果：

```text
lag4-5             淨傷害或無效
first-stroke       淨傷害，尤其拉低 AUC
last3/last5 window 淨傷害或無效
cumulative stats   淨傷害或無效
role features      淨傷害或無效
server transition  不穩定 / 傷 AUC
marginal transition 不穩定或無效
legal mask         幾乎無差異
```

目前有效成分：

```text
base features
+ P(next_action | last_action, last_point)
+ P(next_point  | last_action, last_point)
+ LGBM / TabPFN ensemble
+ OOF weight search
```

---

## 3. 對目前結果的正確解讀

請不要把目前結果解讀成：

```text
這個競賽的天花板就是 0.30
pointId 本質隨機
再做特徵工程沒用
```

更正確的解讀是：

```text
目前 flat prefix tabular representation 的天花板約 0.30。
```

目前方法把整個 prefix 壓縮成一列 summary features：

```text
last
lag2
lag3
mean_spin
mean_strength
nuniq_point
nuniq_action
score
sex
obs_len
transition probability
```

這種表示法丟失了大量結構資訊：

```text
1. pointId 的九宮格幾何結構
2. 左右手 / 持拍手造成的落點鏡像問題
3. 發球方 / 接發方 / 下一拍擊球者角色
4. rally 階段：接發、第三板、第四板、相持
5. pointId=0 的終結風險與非零落點的不同本質
6. 完整 stroke sequence 的時序依賴
```

因此目前 pointId 很低，不一定代表 pointId 不可預測，而可能代表：

```text
模型目前沒有看到正確的空間與角色表示。
```

下一輪目標不是再加普通 lag/window/cumulative，而是驗證：

```text
pointId 是否因為缺乏幾何 / 角色 / 階段 / 序列表示而低分。
```

---

## 4. 下一輪總體策略

下一輪不要再做一般 flat feature expansion。

不要再優先測：

```text
more lag
more window statistics
more cumulative mean/std
more raw player ID
more marginal transition
more server transition
```

下一輪應該測以下六個高結構化假設：

```text
E1. 修正 train/test prefix length distribution mismatch
E2. stage-specific transition
E3. pointId two-stage model
E4. point geometry features
E5. hand / side / perspective normalization
E6. stage-specific expert models
E7. small GRU sequence model
```

其中最優先：

```text
E1, E2, E3, E4
```

如果這些都無效，再做 E5/E6/E7。

---

## 5. E1：修正 prefix length distribution mismatch

### 5.1 問題

目前 LightGBM 使用 all-prefix training：

```text
一個長 rally 會產生多筆 prefix samples。
```

例如 rally 長度 T=10，會產生 9 筆樣本。

但 test 時：

```text
每個 rally_uid 只會有一個已觀測 prefix。
```

這會造成訓練分布與測試分布不一致：

```text
train all-prefix 過度放大長 rally
test one-prefix-per-rally 不一定符合 all-prefix 長度分布
```

目前很多靜態特徵傷害，可能不是特徵本身沒用，而是因為 prefix length distribution mismatch。

---

### 5.2 實驗 E1-A：all-prefix + obs_len sample weight

計算：

```text
P_test_obs_len(L)
P_train_all_prefix_len(L)
```

建立權重：

```text
weight(L) = P_test_obs_len(L) / P_train_all_prefix_len(L)
```

訓練 LGBM 時使用 sample weight。

目標：

```text
讓 train prefix length distribution 更接近 test。
```

請比較：

```text
current all-prefix LGBM
vs
all-prefix + obs_len reweighting
```

---

### 5.3 實驗 E1-B：sampled-prefix bagging

不要用單一 sampled training。

請做：

```text
sampled_train_seed_1
sampled_train_seed_2
...
sampled_train_seed_10
```

每一份 sampled training：

```text
依 test obs_len distribution 從每個 rally 抽 prefix
或讓 sampled prefix 長度分布接近 test
```

訓練 5–10 組 LGBM，最後 ensemble average。

比較：

```text
all-prefix LGBM
vs
weighted all-prefix LGBM
vs
sampled-prefix bagging LGBM
```

---

### 5.4 評估

每個版本都要輸出：

```text
action_macro_f1
point_macro_f1
server_auc
overall
fold mean/std
obs_len-wise metrics
```

新增 obs_len-wise metrics：

```text
L=1 的 action/point/server 表現
L=2 的 action/point/server 表現
L=3 的 action/point/server 表現
L>=4 的 action/point/server 表現
```

如果某方法總分沒提升，但 L=1/L=2 顯著提升，也要記錄。

---

## 6. E2：Stage-specific transition

### 6.1 問題

目前 transition 是全階段共用：

```text
P(next_action | last_action, last_point)
P(next_point  | last_action, last_point)
```

但桌球不同階段規律不同：

```text
L=1：接發球
L=2：第三板
L=3：第四板
L>=4：相持
```

同樣的 `(last_action, last_point)` 在不同 stage 的下一拍分布可能不同。

---

### 6.2 實作

新增：

```text
L_bucket:
    1
    2
    3
    4+
```

建立 fold-safe transition：

```text
P(next_action | L_bucket, last_action, last_point)
P(next_point  | L_bucket, last_action, last_point)
```

同時保留原本 global transition 作為 backoff：

```text
P(next | L_bucket, last_action, last_point)
    -> P(next | last_action, last_point)
    -> P(next | last_action)
    -> P(next | last_point)
    -> global prior
```

所有 transition 必須 fold-safe：

```text
每個 fold：
    只用 fold_train 統計
    套用到 fold_valid

final test：
    才用 full train 統計
```

---

### 6.3 比較組

請比較：

```text
T0 = current global joint transition
T1 = stage-specific action transition only
T2 = stage-specific point transition only
T3 = stage-specific action + point transition
T4 = stage-specific + global backoff
```

不要加入 server transition，因為之前已證明 server transition 傷 AUC。

---

## 7. E3：pointId two-stage model

### 7.1 問題

目前 pointId 是 direct 10-class：

```text
pointId ∈ {0,1,2,3,4,5,6,7,8,9}
```

但：

```text
pointId=0
```

和：

```text
pointId=1~9
```

本質不同。

`pointId=0` 更像：

```text
終結 / 無有效落點 / 特殊狀態
```

而 `1~9` 才是九宮格落點。

因此 direct 10-class model 同時要學：

```text
1. 下一拍是否終結
2. 若不終結，落在哪一格
```

這可能傷害 point Macro F1。

---

### 7.2 實作 two-stage point model

建立兩個模型：

#### Stage 1：point zero model

```text
target_zero = 1 if pointId == 0 else 0
```

模型輸出：

```text
P_zero = P(pointId = 0)
```

#### Stage 2：non-zero point model

只使用：

```text
pointId in 1..9
```

訓練 9-class model：

```text
P_nonzero(k) = P(pointId = k | pointId != 0)
```

#### 合成 10 類機率

```text
P(point=0) = P_zero
P(point=k) = (1 - P_zero) * P_nonzero(k), for k = 1..9
```

最後 argmax 得到 pointId。

---

### 7.3 比較

比較：

```text
P0 = direct 10-class point model
P1 = two-stage point model
P2 = ensemble(direct, two-stage)
```

ensemble 可以做：

```text
P_final = w * P_direct + (1-w) * P_two_stage
```

權重用 OOF 搜尋。

---

### 7.4 指標

除了 point Macro F1，請額外輸出：

```text
point_zero_auc
point_zero_f1
point_zero_precision
point_zero_recall
nonzero_point_macro_f1
point_macro_f1_all_10
```

這樣才能知道問題是：

```text
point=0 判斷不好
還是 1~9 九宮格判斷不好
```

---

## 8. E4：point geometry features

### 8.1 問題

目前 pointId 被當成無序類別，但 1~9 是九宮格，具有空間幾何。

應把 pointId 拆成：

```text
row
col
short/mid/deep
left/middle/right
corner/body
```

---

### 8.2 建立可調 mapping

先假設一般九宮格 mapping：

```text
1 2 3
4 5 6
7 8 9
```

對應：

```text
point_row:
    1,2,3 -> row 0
    4,5,6 -> row 1
    7,8,9 -> row 2

point_col:
    1,4,7 -> col 0
    2,5,8 -> col 1
    3,6,9 -> col 2
```

但請注意：

```text
官方簡報提到落點分類會根據右持拍手、左持拍手定義九宮格。
因此 mapping 必須寫成 config，不要 hard-code 死。
```

至少保留：

```text
MAPPING_A = [[1,2,3],[4,5,6],[7,8,9]]
MAPPING_B = [[7,8,9],[4,5,6],[1,2,3]]
MAPPING_C = horizontal mirror
MAPPING_D = vertical mirror
```

用 OOF 比較哪個 mapping 最合理。

---

### 8.3 幾何特徵

對 last / lag2 / lag3 pointId 建立：

```text
last_point_row
last_point_col
lag2_point_row
lag2_point_col
lag3_point_row
lag3_point_col
```

建立衍生特徵：

```text
last_point_is_left
last_point_is_middle_col
last_point_is_right
last_point_is_short
last_point_is_mid_depth
last_point_is_deep
last_point_is_corner
last_point_is_center
last_point_is_edge
```

建立相對變化：

```text
delta_row_last_lag2
delta_col_last_lag2
same_row_last_lag2
same_col_last_lag2
is_cross_court
is_line_change
short_to_deep
deep_to_short
left_to_right
right_to_left
```

---

### 8.4 幾何 target decomposition

除了直接 point 10-class，請額外訓練：

```text
point_row model
point_col model
```

對 non-zero point：

```text
row ∈ {0,1,2}
col ∈ {0,1,2}
```

合成：

```text
P(point=k) ≈ P(row=r) * P(col=c)
```

比較：

```text
direct 9-class nonzero point
vs
row/col decomposition
vs
ensemble(direct, rowcol)
```

這可能提升 point 的泛化，因為模型可以學到「深淺」和「左右」兩個較簡單任務。

---

## 9. E5：hand / side / perspective normalization

### 9.1 問題

簡報明確提到：

```text
落點分類根據不同持拍手來定義九宮格分布：右持拍手、左持拍手。
```

這代表 raw pointId 可能不是全域一致座標。

如果左手與右手的 pointId 定義存在鏡像關係，直接把 raw pointId 丟進模型會讓 pointId 規律被打散。

---

### 9.2 要先做資料檢查

請檢查：

```text
pointId distribution by handId
pointId distribution by actionId and handId
next_point distribution by last_point and handId
next_point distribution by last_action, last_point, handId
```

如果左右手的 pointId distribution 呈現鏡像，表示需要 normalization。

---

### 9.3 嘗試 mirrored / normalized point

建立：

```text
mirror_point_horizontal
mirror_point_vertical
mirror_point_by_hand
```

例如 horizontal mirror：

```text
1 <-> 3
4 <-> 6
7 <-> 9
2 -> 2
5 -> 5
8 -> 8
```

vertical mirror：

```text
1 <-> 7
2 <-> 8
3 <-> 9
4 -> 4
5 -> 5
6 -> 6
```

根據 handId 建立：

```text
normalized_pointId_by_hand
```

如果 handId 表示左手：

```text
normalized_point = mirror(raw_point)
```

如果 handId 表示右手：

```text
normalized_point = raw_point
```

但因為官方 mapping 未必如此，請保留多種 mapping 做 OOF 比較。

---

### 9.4 以 perspective 建立 point features

請建立：

```text
point_from_current_hitter_perspective
point_from_next_hitter_perspective
point_from_server_perspective
point_from_receiver_perspective
```

如果無法明確定義，至少建立不同版本並用 OOF 比較。

---

## 10. E6：stage-specific expert models

### 10.1 問題

目前只有 global model，然後把 obs_len 當 feature。

但不同 stage 是不同子問題：

```text
L=1：接發球
L=2：第三板
L=3：第四板
L>=4：相持
```

一個 global model 可能很難同時學好這些階段。

---

### 10.2 實作

建立四個 expert：

```text
expert_L1
expert_L2
expert_L3
expert_L4plus
```

每個 expert 分別訓練：

```text
actionId model
pointId model
serverGetPoint model
```

同時保留 global model。

推論時：

```text
final_prob = alpha * expert_stage_prob + (1-alpha) * global_prob
```

alpha 用 OOF 搜尋：

```text
alpha ∈ {0.25, 0.5, 0.75, 1.0}
```

如果某 stage 訓練樣本太少，就使用 global fallback。

---

### 10.3 評估

請輸出：

```text
overall
stage-wise action F1
stage-wise point F1
stage-wise server AUC
```

特別觀察：

```text
L=1 和 L=2 是否提升
```

因為 test 很可能集中在短 prefix。

---

## 11. E7：small GRU sequence model

### 11.1 為什麼要做 sequence model

目前 LGBM / TabPFN 都吃 flat features：

```text
prefix sequence -> manually summarized features -> model
```

但這題本質是：

```text
stroke sequence -> next stroke prediction
```

因此應至少建立一個小型 sequence model 作為 ensemble 成員。

不要一開始做大 Transformer。資料量不大，GRU 更穩。

---

### 11.2 模型架構

每一拍 stroke 輸入欄位：

```text
strikeId / strikeNumber
actionId
pointId
spinId
strengthId
handId
positionId
scoreSelf
scoreOther
server_or_receiver_role
sex
```

每個 categorical field 做 embedding：

```text
Embedding(actionId)
Embedding(pointId)
Embedding(spinId)
Embedding(strengthId)
Embedding(handId)
Embedding(positionId)
Embedding(role)
```

numeric features 做 normalization + linear projection：

```text
scoreSelf
scoreOther
obs_len
strikeNumber
```

每拍 representation：

```text
stroke_embedding = concat(all embeddings + numeric projection)
```

Sequence encoder：

```text
GRU hidden size 64 or 128
1–2 layers
dropout 0.1–0.3
```

取最後 hidden state：

```text
h_last
```

三個 prediction heads：

```text
action_head: 19-class softmax
point_head: 10-class softmax
server_head: binary sigmoid
```

---

### 11.3 Loss

Multi-task loss：

```text
loss = w_action * CE(action)
     + w_point  * CE(point)
     + w_server * BCE(server)
```

初始權重：

```text
w_action = 0.4
w_point  = 0.4
w_server = 0.2
```

也可以測：

```text
w_action = 1.0
w_point  = 1.0
w_server = 0.5
```

對 action / point 可使用 class weights 或 focal loss：

```text
class-weighted CE
或
focal loss gamma = 1 or 2
```

但 focal loss 要用 OOF 驗證，不要預設一定更好。

---

### 11.4 Training samples

與目前 prefix training 相同：

```text
prefix sequence -> next action / next point / serverGetPoint
```

需要 padding：

```text
max_len = max prefix length 或截斷到合理長度
mask padding positions
```

如果 rally 不長，可直接 pad 到 max_len。

---

### 11.5 Evaluation

必須使用：

```text
GroupKFold by match
```

指標：

```text
action_macro_f1
point_macro_f1
server_auc
overall
```

比較：

```text
current LGBM+TabPFN baseline
vs
GRU only
vs
baseline + GRU ensemble
```

GRU 不一定單獨贏，但可能作為 ensemble 成員提供不同 inductive bias。

---

## 12. 可考慮的新方法，但不是第一優先

### 12.1 TabM

TabM 是 2024 提出的 tabular deep learning 方法，重點是 parameter-efficient ensembling。可以作為目前 tabular ensemble 的第三個成員。

用途：

```text
LGBM + TabPFN + TabM
```

但它仍然吃 tabular features，因此如果 point geometry / stage / prefix weighting 沒做好，提升可能有限。

參考：
https://arxiv.org/abs/2410.24210

---

### 12.2 TabPFN-2.5

TabPFN-2.5 是 2025 的 tabular foundation model 進展，宣稱能支援更大資料與更多 features。

如果環境可用，可以測：

```text
TabPFN-2.5
```

或使用較新的 TabPFN 設定。

但請注意：

```text
TabPFN 不是 sequence model
它仍然依賴正確 feature representation
```

參考：
https://arxiv.org/abs/2511.08667

---

### 12.3 HT-Transformer / History Tokens 思路

2025 HT-Transformer 指出，對 event sequence classification，Transformer 可能缺少單一 compact history state，因此提出 history tokens 累積 prefix information。

這給本題的啟發是：

```text
不要只用普通 Transformer。
如果做 sequence model，可以先用 GRU 或加入 history token / CLS token 來聚合序列。
```

參考：
https://arxiv.org/abs/2508.01474

---

## 13. 這一輪的實驗優先順序

請依照以下順序做，不要一次全部做。

---

### Priority 1：修正訓練分布

```text
E1-A all-prefix + obs_len sample weight
E1-B sampled-prefix bagging
```

原因：

```text
成本低
可能修正 train/test mismatch
有機會讓既有特徵更有效
```

---

### Priority 2：pointId 任務重構

```text
E3 two-stage point model
E4 point geometry features
```

原因：

```text
目前 overall 上不去的最大瓶頸是 pointId。
要突破 0.30，point 必須提升。
```

---

### Priority 3：stage-specific

```text
E2 stage-specific transition
E6 stage-specific experts
```

原因：

```text
不同 prefix length 對應不同桌球戰術階段。
L=1/L=2/L=3/L>=4 不應完全共用同一組規律。
```

---

### Priority 4：hand / perspective normalization

```text
E5 hand/side point normalization
```

原因：

```text
簡報明確提到落點分類與左右持拍手有關。
這可能是目前 pointId 表示法最大的缺失。
```

---

### Priority 5：sequence model

```text
E7 small GRU multi-task model
```

原因：

```text
這題本質是 event sequence prediction。
如果 flat tabular representation 卡住，必須引入 sequence model。
```

---

## 14. 每次實驗必須輸出

每個實驗都要 append 到 `experiments_log.csv`：

```text
experiment_name
feature_set
model_type
action_macro_f1
point_macro_f1
server_auc
overall
fold_mean
fold_std
delta_vs_current_best
delta_vs_base
notes
```

每次都要輸出：

```text
official metrics:
    action_macro_f1
    point_macro_f1
    server_auc
    overall

diagnostics:
    action per-class precision/recall/f1/support
    point per-class precision/recall/f1/support
    point=0 binary metrics
    nonzero point macro_f1
    server logloss/brier/pr_auc
    obs_len-wise metrics
    stage-wise metrics
    submission distribution check
```

---

## 15. 決策規則

不要只看單一 metric。

採用新方法的條件：

```text
1. OOF overall 提升
2. fold std 沒有明顯變大
3. pointId 沒有被犧牲太多
4. server AUC 沒有被小特徵嚴重拉低
5. test prediction distribution 沒有異常
```

如果某方法：

```text
action +0.010
point -0.015
server +0.000
overall 下降
```

不要採用。

如果某方法：

```text
overall +0.002
但 fold std 從 0.007 變 0.020
```

標記為 unstable，不直接採用。

---

## 16. 最後要避免的事情

請不要再浪費時間在：

```text
1. 無條件增加 lag 數
2. 無條件增加 first-stroke raw features
3. 無條件增加 last3/last5 mean/std
4. raw player ID target encoding
5. server transition features
6. legal mask 當主要提升手段
7. 只調 TabPFN / LGBM 權重但不改資料表示
```

目前結果已證明這些方向要嘛無效，要嘛容易 overfit。

---

## 17. 最終目標

下一輪目標不是立刻追求 0.5，而是回答這個核心問題：

```text
目前 pointId 低分，是因為資料本質不可預測，
還是因為模型缺少幾何、左右手、stage、sequence representation？
```

如果 E1–E7 都測完仍然卡在 0.30 附近，才可以說：

```text
這份資料在誠實 match-held-out validation 下確實可預測性有限。
```

但在測完這些之前，不要把 0.30 視為競賽天花板。

---

## 18. 立即執行清單

請 Claude Code 依序執行：

```text
[ ] E1-A all-prefix + obs_len sample weighting
[ ] E1-B sampled-prefix bagging
[ ] E2 stage-specific transition
[ ] E3 point two-stage model
[ ] E4 point geometry features
[ ] E5 hand/perspective normalization
[ ] E6 stage-specific experts
[ ] E7 small GRU sequence model
```

每一步都要和目前 best baseline 比較：

```text
current best:
Overall 0.3038
F1_action 0.284
F1_point 0.175
AUC_server 0.601
fold std 0.007
```

這是下一輪所有實驗的比較基準。
