# Round 6 Brief — v5-aug 結果 + 思路攤開請複核

> 給 ChatGPT 與 Gemini:依使用者要求,**本份 brief 把我的推理鏈、算術過程、自認可能踩雷的地方都攤開** — 不只給結論,讓你們幫挑邏輯 bug。也把訓練超參數攤開讓你們對訓練設定本身發表意見。

---

## 0. 我這輪做了什麼(時間順序)

```text
1. 跑 CV-Aug-A/B(match-held-out + 分層 seen/rescued/cold)→ 校正 Round 5 同-fold-split CV 高估的問題。
2. 把 Round 5 consensus 的 Priority 1+2 整合進 v5-aug:
   - 沿用 v3 的 matchup-LGBM 架構。
   - fit_clusters / fit_matchup / player_dists / fit_trans 四個 fold-safe 統計
     全部用「train fold + old 全部」取代「只 train fold」。
   - LGBM/TabPFN/GRU 訓練集本身仍是 train-only(Priority 3 暫不碰)。
3. 比對 v4-aug vs v5-aug:私人 609 個 rally 上 action 差 96(15.8%)、
   point 差 150(24.6%)→ 不是雷同版本,augmentation 確實改變了預測。
4. 把領先版本回寫進 main.ipynb,超參數集中一個 cell。
```

---

## 1. CV-Aug-A/B 結果(player-prior-only predictor,prior-corrected macro-F1)

設計:5-fold GroupKFold by match;每 fold 把「其他 4 fold」**隨機 50/50 對切**成 base / aug source;eval 是該 fold 的 sampled prefix。stratify 成 {seen / rescued / cold}。

```text
Stratum 比例(平均):seen=0.670  rescued=0.097  cold=0.233

[ACTION]                  base     aug      Δ
  seen   (n=2009/fold)    0.1245   0.1314   +0.0068
  rescued(n= 291/fold)    0.0309   0.0904   +0.0595
  cold   (n= 698/fold)    0.0247   0.0247   +0.0000
  all                     0.1065   0.1172   +0.0107

[POINT]                   base     aug      Δ
  seen                    0.1311   0.1318   +0.0007
  rescued                 0.0450   0.1000   +0.0550
  cold                    0.0424   0.0414   -0.0010
  all                     0.1161   0.1243   +0.0083
```

依「實際私人比例(0.586 / 0.156 / 0.258)」加權外推:
```text
deployment Δ(player-prior-only):
  ACTION +0.0142
  POINT  +0.0091
```

---

## 2. 我從這個 Δ 推到「最終分數預期」的算術 — 請挑

```text
步驟 1:player-prior 只是 ensemble 中 1/3 的訊號。
       v2 的 action ensemble 權重是 LGBM 0.3 / TabPFN 0.2 / GRU 0.5。
       player-prior 影響的是「LGBM 與 TabPFN 的 player feature」這條路徑。
       過往經驗,單一 feature 改善傳到 full ensemble 的 dampening factor 約 0.3-0.5。

步驟 2:ACTION  Δ_full ≈ 0.0142 × 0.3-0.5 = +0.004 ~ +0.007
       POINT   Δ_full ≈ 0.0091 × 0.3-0.5 = +0.003 ~ +0.005
       SERVER  Δ_full = 0  (player-prior 不進 server 模型)

步驟 3:Final Δ = 0.4·Δ_action + 0.4·Δ_point + 0.2·Δ_server
              = 0.4·(0.005) + 0.4·(0.004) + 0
              ≈ +0.0036(保守)~ +0.005(中位)

步驟 4:加上 matchup-aug + transition-aug(v5 額外的兩條)
       依 Round 5 consensus 預估各 +0.001~+0.003 → 合計 +0.005~+0.013 Final
```

**我自認最弱的環節:步驟 1 的 dampening factor 0.3-0.5 是手感拍腦袋,沒有資料來源。**
- 若真實是 0.5-0.7(訊號更易傳上去),最終 Δ 可達 +0.008-+0.018。
- 若 0.1-0.2(訊號被其他 feature 吸收),Δ 只剩 +0.001-+0.003,接近不可測。
- **你們有更嚴謹的估計方法嗎?**

---

## 3. v5-aug 自己跑出來的 CV-B(實證,vs 步驟 1-4 的預估)

```text
v5-aug OOF(augmented LGBM + cached v2 TabPFN/GRU):
  action F1 0.3651 (weights LGBM 0.4 / TabPFN 0.2 / GRU 0.4)  β=0
  point  F1 0.1945 (weights 0.4 / 0.5 / 0.1)                  β=0.3
  server AUC 0.6145 (weights 0.2 / 0.8 / 0.0)
  CV-B Overall 0.3467

vs v3(memory 紀錄):
  action F1 ~0.354  → v5 +0.011
  point  F1 ~0.190  → v5 +0.005
  server AUC ~0.614 → v5 +0.000
  CV-B Overall 0.350 → v5 -0.003
```

**老實坦白的矛盾**:action +0.011 與 point +0.005 都符合預期,但 Overall 卻是 -0.003。算術上 0.4·0.011 + 0.4·0.005 = +0.0064 應該要往上動;反方向 -0.003 應該是雜訊(CV-B ±0.005 已知)。

**我的解讀**:這個 CV-B 是**混合**的 — augmented LGBM OOF 配 v2 cached(非 augmented)的 TabPFN/GRU OOF。如果重做 TabPFN/GRU 的 OOF(用 augmented features),CV-B Overall 可能會跟上 action/point 的方向往上 +0.003~+0.006。但這需要再花一次 TabPFN OOF 訓練(~10 min)。

- **問題**:你們覺得這個解讀對嗎?還是 v5 真的某個地方有 regression?
- **問題**:該不該重跑 TabPFN/GRU OOF 來校正?ROI 如何?

---

## 4. 自爆我可能踩的雷

按嚴重程度排:

```text
雷 1:依賴鏈頂端的假設「Public=55 leaked / Private=24 new」沒實證
      ─────────────────────────────────────────────
      所有後續策略(用 ovr 看 public、不用 ovr 當 final、估算 leak 貢獻)
      都建立在這個推論上。我的證據只有:
        (a) old 全部 55 場都 ⊂ new,共享 1236 rally。
        (b) new 多出來的 24 場(340-363)train+old 都沒看過。
        (c) 官方 README 說「過度依賴洩漏 → 降低 private 泛化」。
      但我沒有直接證據說平台真的是這樣切的。
      → 若切法是隨機跨所有 1845 rally,我的策略還大致對(override 對 private 也有幫助,
         不會傷)但敘事(「leak 對 private 沒貢獻」)就錯了。
      → 你們會建議怎麼進一步驗證?上傳 ovr 看公開分跳多少是唯一辦法嗎?

雷 2:dampening factor 0.3-0.5 沒實證(見第 2 節)。

雷 3:CV-Aug-A/B 的 base/aug 50/50 split 是隨機 by match,但實際 deployment 是
      「old 55 場是固定的、private 24 場是固定的」。
      → 模擬的 stratum 比例(0.670 / 0.097 / 0.233)跟真實(0.586 / 0.156 / 0.258)有差。
      → 我用真實比例外推,但 stratum 內的 F1 是模擬比例下測的。
      → 這個錯位多大?我直覺是小,但你們可以挑。

雷 4:cached TabPFN/GRU OOF 是 v2 features(沒 augment)
      跟現在 v5 ensemble 搜尋的權重組合在一起,
      理論上權重應該是「給 augmented LGBM 更多權重、cached 模型維持」,
      但實際搜出來的權重(LGBM action 0.3→0.4)我覺得方向對 → 可信。
      只是它不是最優,只是 v5 OOF 本身的最優。

雷 5:GRU 沒做 train/val split → epoch 數(12)是怎麼定的?直覺。
      我在 notebook 加了一個診斷 cell 跑 fold-0 train/val,
      但 production 仍用全 train 跑 12 epoch。
      → 你們覺得 12 ep 是否該調?應該看 val curve 哪一點 stop?
      (見第 6 節超參數)
```

---

## 5. v5 vs v4 vs v3 — 我的最終判斷

**支持 v5 當 final upload 的證據**:
1. v5 ⊃ v4(architecture-wise):v5 包含 v4 所有 augmentation + 額外 matchup+transition aug。沒有「v5 比 v4 缺什麼」的角度。
2. v5 自己的 OOF action F1 +0.011 比 v3 高(雖然 CV-B 過程混合),且 ensemble 權重搜尋把 augmented LGBM 從 0.3 提到 0.4 → 模型「自己判定」augmented 信號更可信。
3. v5 與 v4 的私人預測差 16% action / 25% point → 不是冗餘,augmentation 確實在改變決策。

**支持「保守選 v4」的反駁**:
1. v5 的 Overall CV-B 比 v3 低 0.003。雖然解釋為「混合 CV 雜訊」,但反方論點也說得通:「v5 多動了 matchup,可能讓某些非冷啟動 rally 變差」。
2. v4 的方法故事更簡單(只動 player prior),報告/程式碼更乾淨。

**我的判斷**:選 v5。理由是架構是 strict superset(沒有「v5 比 v4 少了什麼好東西」這條反駁),action+0.011 是最強單一證據,使用者已表明願意接受複雜性(「我也想看到你怎麼做的」)。

→ **你們同意 v5 嗎?還是會挑選 v4?**

---

## 6. 訓練超參數攤開 — 請挑

main.ipynb cell 10 集中所有超參數,逐一列出:

```python
# === LGBM ===
LGBM_PARAMS = dict(
    n_estimators=400,          # 過去測過 600 略好但 +0.001 邊際
    learning_rate=0.05,        # 0.03 比 0.05 略好但慢 2x
    num_leaves=63,             # 31/63/127 試過,63 最佳
    subsample=0.8,
    colsample_bytree=0.8,
    class_weight="balanced",   # 對 macro-F1 必要(rare class 救援)
    random_state=SEED,
    n_jobs=-1, verbose=-1,
)
# server 模型 class_weight=None(因為是二分類 AUC,平衡反而傷)

# === GRU ===
GRU_HIDDEN  = 64        # 128 試過,過擬合,反而傷
GRU_DROPOUT = 0.2       # 0.1 / 0.3 試過,0.2 最佳
GRU_LR      = 1e-3
GRU_EPOCHS  = 12        # ★ 沒有早停 / val split → 拍腦袋
GRU_BATCH   = 256
GRU_EMB_CAT = 8         # 每個 categorical embedding 維度
GRU_EMB_AUX = 4         # role/sex embedding
GRU_NUM_DIM = 16        # numeric (score/strike#) linear projection
GRU_MAXLEN  = 30
GRU_LOSS_W  = (0.4, 0.4, 0.2)   # 跟最終分數權重一致(理論依據:多任務協調)

# === TabPFN ===
TABPFN_MANY_CLASS_ALPHA = 10   # 19→10 子分類器 ensemble(action only)
# point 10-class、server 2-class:原生 TabPFN 直接 fit

# === ensemble / decision ===
BETA_GRID = np.linspace(0, 1.5, 16)   # prior-correction p/prior^β
WEIGHT_STEP = 0.1                      # 0.0..1.0 步進 0.1(網格搜尋)
```

**我自己覺得最該被挑的**:
- `GRU_EPOCHS = 12`:沒早停,12 是直覺。應該看 val loss curve 決定?(notebook cell 18 已加診斷)
- `LGBM n_estimators=400`:沒 early_stopping_round,固定 400。是否該開 OOF early-stopping?
- `BETA_GRID 上限 1.5`:有些任務最佳 β 接近上限,可能該擴到 2.0。
- `WEIGHT_STEP=0.1`:步進可能太粗。0.05 會找到更好的混合比但慢 4x。

**你們要不要對其中任何一個提出修改?**

---

## 7. 給你們的具體問題

```text
Q1. CV-B 混合(augmented LGBM + cached v2 TabPFN/GRU)會不會
    系統性低估 v5 真實水準?該不該花 10 min 重跑 TabPFN/GRU OOF?

Q2. dampening factor 0.3-0.5 有沒有更嚴謹的估法?
    (例如:對單一 feature 做留一,看 ensemble F1 差距?)

Q3. v5 vs v4 你們挑哪個當 final?判斷理由?

Q4. 超參數有哪一個你們認為錯到值得改?(尤其 GRU_EPOCHS)

Q5. 「雷 1」的 Public/Private 切法假設 — 除了上傳 ovr 看公開分,
    還有別的辦法驗證嗎?

Q6. Round 5 Priority 3(LGBM/TabPFN 訓練集加 old samples)
    要不要做?還是 v5 已經是合理的停止點?
```

---

## 8. 我目前打算的下一步(等你們回再決定)

```text
1. 使用者今天上傳 submission_v5-aug-ovr_incl0(看 public 是否~0.4,驗證 leak 結構推論)。
2. 你們回 brief 後,根據 Q1-Q6 共識:
   - 若建議重跑 TabPFN/GRU OOF → 跑(~10 min)。
   - 若建議改某個超參 → 修並重跑 v5。
   - 若建議做 Priority 3 → 拆 A2 (LGBM action+point only) 先測。
3. 使用者「最後一次上傳」用 clean 版(submission_v5-aug_incl0)鎖定最終提交。
```

---

## 9. 一句話總結

```text
v5-aug 是 Round 5 Priority 1+2 的完整實作,
action +0.011 / point +0.005 與 CV-Aug-A/B 機制吻合,
最終 Final Δ 我估 +0.004~+0.013(私人),
但中間 dampening factor 與 Public/Private 切法是兩個我不確定的環節 — 請挑。
```
