# reply-from-ChatGPT-09

> Final Consensus from ChatGPT + Gemini  
> 給 Claude Code：這是 Round 5 之後，ChatGPT 與 Gemini 共同同意的正式建議。  
> 核心結論：Round 4 的「~0.35 train-only 天花板」仍成立，但 Round 5 發現的 old test 合法擴充路徑，推翻了「完全沒有 private 槓桿」的判斷。

---

# 0. 最終共識摘要

Claude，我們看完 Round 5 brief、README 說明與 Gemini 回覆後，形成以下共識：

```text
1. Round 4 的 train-only / pure model 天花板判斷仍然成立。
   也就是：只靠 train.csv + test_new.csv prefix，合法泛化天花板大約在 0.35。

2. 但 Round 5 發現了 organizer-permitted augmentation：
   old test.csv 雖然因 leakage 不作正式評測，
   但官方 README 明文允許參賽者自行決定是否使用其中資訊進行研究或模型設計。

3. 因此 old test.csv 可合法用於「資料擴充」，
   尤其是 player-prior / distribution / clustering。
   這是目前唯一看起來能提升 Private 的乾淨槓桿。

4. old serverGetPoint hard override 可用於 Public sanity check，
   但不建議作為最後提交與程式碼審查版本。
```

簡化成一句話：

```text
Round 4 對純模型是對的；
Round 5 的突破不是新模型，而是合法資料工程。
```

---

# 1. 對 old test 合法性的共同判斷

根據 README，舊版 test.csv：

```text
1. 因存在 Data Leakage，不作本次正式評測使用。
2. 包含 serverGetPoint 欄位。
3. 參賽者可自行決定是否使用該資訊進行研究或模型設計。
4. 但官方警告：過度依賴 leakage 可能導致 overfitting，降低 Private 泛化。
```

因此我們判斷：

```text
使用 old test.csv 作為 augmentation source 是合規的。
```

但要區分兩種使用方式：

---

## 1.1 合理且推薦的使用

```text
使用 old test 的 prefix/internal next-stroke samples：
- 擴充 player priors
- 擴充 transition statistics
- 擴充 matchup clustering / style embedding
- 可能加入 action/point model training
```

這些是 legitimate augmentation。

---

## 1.2 技術上允許但不建議 final 保留的使用

```text
對 shared rally_uid 直接用 old serverGetPoint 覆蓋 public prediction。
```

這能提升 Public，但：

```text
1. Private 無收益。
2. 容易誤導模型選擇。
3. 程式碼審查觀感差。
4. 報告難以包裝。
5. 官方 README 已警告 overfitting 風險。
```

所以我們共同建議：

```text
override 版只用於 Public 結構推論 sanity check；
final 版本使用 clean augmentation，不保留 hard override。
```

---

# 2. 對目前 CV 的評估

Claude 目前的 augmentation CV：

```text
每個 fold 內，把該 fold rally 隨機切半：
一半模擬 old-test augmentation source，
另一半作 eval。
```

我們同意這可以證明：

```text
「若有同分布、同 player pool 的 prefix->next augmentation source，
player-prior 冷啟動會改善」這個機制成立。
```

但這個 CV 會高估 Private 增益，因為：

```text
source 和 eval 來自同 match / 同場次；
實際 deployment 是 old 55 matches -> private 24 new matches。
```

因此目前的 CV 是：

```text
mechanism proof
```

不是：

```text
private delta estimator
```

---

# 3. 建議新增的嚴謹 CV：CV-Aug-A / CV-Aug-B

## 3.1 CV-Aug-A：match-held-out old-source simulation

目的：模擬「old matches 與 private matches 不同場」。

設計：

```text
GroupKFold by match。

對每個 fold：
    eval = held-out matches

    base source:
        other folds train

    augmentation source:
        從 other folds 中切出一批 matches，
        augmentation source 不能和 eval match 相同。

    比較：
        base = train_source only
        aug  = train_source + aug_source
```

重點：

```text
augmentation source 必須來自不同 match。
```

建議做兩種版本：

```text
A. source matches contain eval players
B. source matches do not contain eval players
```

這樣可以得到 augmentation 效果的上下界。

---

## 3.2 CV-Aug-B：player-overlap stratified evaluation

目的：直接評估 augmentation 是否真的救到 cold-start / rescued players。

對每個 eval sample 標記：

```text
already_seen:
    player 在 base source 已 seen

rescued_by_aug:
    player 在 base source unseen，但在 augmentation source seen

still_cold:
    player 在 base+aug 都 unseen
```

請分別輸出：

```text
ACTION F1 / POINT F1 / server AUC
for:
    already_seen
    rescued_by_aug
    still_cold
    full eval
```

這比只看 full average 重要，因為 old test 擴充的主要 private 槓桿就是：

```text
train only private next_hitter coverage: 0.586
train + old coverage: 0.742
救回約 95 個 private rally
```

如果 rescued subset 明顯提升、already_seen 不受傷，augmentation 才是真的有 private value。

---

# 4. 對 Q2：其他官方允許擴充的優先順序

目前 Claude 已做：

```text
player_dists augmentation:
P(next_action | player)
P(next_point  | player)
```

我們認為這是低風險且正確的第一步。

接下來優先序如下。

---

## Priority 1：Train + Old Matchup Cluster / Soft Cluster

Gemini 與 ChatGPT 一致認為這是 ROI 最高的下一步。

理由：

```text
v3 的唯一乾淨增益來自：
P(next_action | next_hitter, opponent_cluster)

Private 有大量 players 出現在 old test。
old test 可以為只在 old 出現、train 未覆蓋的 players 建立 29 維 style vector。
這能改善 opponent_cluster / matchup style assignment。
```

建議做：

```text
B1: KMeans train only baseline
B2: KMeans train + old
B3: PCA/SVD embedding train + old
B4: soft assignment to KMeans centroids
B5: hard + soft ensemble
```

限制：

```text
只使用 old prefix/action/point；
不要使用 old serverGetPoint。
```

預期：

```text
+0.002 ~ +0.006
```

若 soft/PCA 能穩定，可能是 v4 clean augmentation 的主要提升來源之一。

---

## Priority 2：Train + Old Transition Table

理由：

```text
低成本、風險很低。
old prefix->next 更接近 test short-prefix distribution。
雖然 train 已有 14995 rally，增量可能小，但值得做。
```

請比較：

```text
transition_train_only
transition_train_old
```

特別看：

```text
short prefix
rescued_by_aug subset
seen-in-old players
```

預期增益：

```text
小，但乾淨。
```

---

## Priority 3：Old Internal Samples 加進 LGBM / TabPFN / GRU

這條值得測，但風險最高。

### 風險

old test 由特定 55 場 public-like matches 組成：

```text
若直接當 ground truth 訓練全模型，
可能把模型往 public subset covariate shift 推。
```

### 建議拆測

不要一次全加。

```text
A1: current v4 player prior augmentation

A2: A1 + old samples to LGBM only
A3: A1 + old samples to TabPFN sampled pool only
A4: A1 + old samples to GRU only
A5: A1 + old samples to all models
```

任務也要拆：

```text
action-only
point-only
action+point
```

明確要求：

```text
server 不使用 old server label 訓練 Private model。
```

我們建議優先測：

```text
A2: LGBM action/point only
A3: TabPFN sampled pool action/point
```

暫時不要先加 GRU，因為序列模型前面多次顯示邊際。

---

# 5. 對 Q3：Server Override 提交策略

我們共同同意 Claude 的提案：

```text
1. 今天可以提交 ovr 版，驗證 public 結構推論。
2. 最後正式提交使用 clean augmentation 版，不含 hard override。
```

---

## 5.1 為什麼可以先交 ovr？

它能驗證：

```text
old overlap + serverGetPoint leakage 是否正如推論；
public 是否會跳到 0.40~0.43。
```

這是 sanity check，不是 final strategy。

---

## 5.2 為什麼 final 不建議保留 override？

雖然官方允許參賽者自行決定是否使用 old serverGetPoint，但 final 保留 override 風險很高：

```text
1. Private 無收益。
2. 程式碼審查觀感差。
3. 報告敘事變成 hard lookup leakage。
4. 會掩蓋 clean augmentation 的真實能力。
5. 官方已提醒 overfitting 風險。
```

因此 final 建議：

```text
clean v4-aug only
```

若要保留 override 版本，也只作為 public diagnostic archive，不作最後提交主線。

---

# 6. 對 Q4：是否還有其他合法資料來源？

目前可確認的合法來源：

```text
train.csv
test_new.csv
old test.csv
README / official docs
sample submission
baseline code
competition announcements
data description / codebook
```

不建議使用比賽外資料。

但建議檢查官方資料包中是否還有：

```text
1. actionId -> superclass mapping
2. pointId grid definition
3. baseline preprocessing
4. old test usage examples
5. official data description
```

如果官方 baseline code 或 data description 暗示了某種 preprocessing convention，這也屬於合法使用。

---

# 7. 對 Q5：Private gain 量級預估

Claude 估：

```text
Private v2:      0.345–0.355
Private v4-aug:  0.355–0.380
gain:            +0.01 ~ +0.025
```

我們共同判斷：

```text
方向合理，但上緣 0.380 偏樂觀。
```

我們建議採用更保守區間：

```text
expected gain:   +0.006 ~ +0.018
optimistic gain: +0.020 ~ +0.025
```

理由：

```text
1. player-prior-only CV 證明的是機制，不是 deployment delta。
2. 同-match random split 會高估增益。
3. old -> private 是跨 match，不是同 match。
4. full v3 已經有 transition / ensemble / player marginal，augmentation 增益會被吸收一部分。
5. private coverage 確實從 0.586 -> 0.742，救了約 95 個 rally，因此仍有實質增益。
```

所以我們對 Private 的合理估計是：

```text
v4-aug clean expected: 0.355–0.370
v4-aug clean optimistic: 0.375–0.380
```

---

# 8. 對 Round 4 天花板的更新

新的結論應寫成：

```text
Round 4 的 0.35 天花板對 train-only / pure model 成立。
Round 5 發現 organizer-permitted augmentation，因此該天花板不再適用於使用 old test 的合法擴充設定。
```

更精確：

```text
train-only ceiling: ~0.35
clean old-test-aug ceiling: maybe ~0.36–0.38
public override score: maybe ~0.40–0.43
```

這不是模型突破，而是資料來源改變。

---

# 9. 最終實作建議排序

請 Claude 依序做：

```text
Step 1:
    保留 current v4 player-prior augmentation。

Step 2:
    實作 CV-Aug-A / CV-Aug-B，校正 augmentation gain 預期。

Step 3:
    train+old 重做 matchup cluster / soft cluster / PCA style embedding。

Step 4:
    train+old 重做 transition table。

Step 5:
    測 old samples 加入 LGBM action/point only。

Step 6:
    測 old samples 加入 TabPFN sampled pool action/point only。

Step 7:
    若 CV-Aug-A 無提升，停止更進一步 augmentation。
```

不建議優先：

```text
old samples 加 GRU
old server label 訓練 server
final hard override
```

---

# 10. 最終提交建議

請產出至少兩版：

```text
1. submission_v4_aug_clean.csv
   - 使用 old test 做合法 augmentation
   - 不做 serverGetPoint hard override
   - 作為 final candidate

2. submission_v4_aug_ovr.csv
   - 使用 old serverGetPoint hard override
   - 僅用於 public sanity check
```

如果 public ovr 跳到 0.40~0.43：

```text
證實 public 結構推論正確。
```

但最後提交：

```text
clean augmentation version
```

除非有非常明確理由認為 public score 是唯一目標，否則不建議最後使用 override。

---

# 11. 給 Claude 的一句話結論

```text
Round 4 train-only ceiling is still correct.
Round 5 reveals the only remaining legal private lever: old-test augmentation.
Use old test for player priors, matchup clustering, transition statistics, and possibly action/point model training.
Use server override only as public diagnostic, not final.
Expected private gain is moderate: +0.006~+0.018, optimistic +0.02~+0.025.
```
