# Round 7 Brief — v6 fully-augmented OOF 結果(內含程式碼修正紀錄)

> 給 ChatGPT 與 Gemini:這份 brief 報告依 Round 6 consensus 改完的 v6 結果。**請特別關注 §3 的一個反直覺發現** — 你們先前最擔心的「mixed-OOF 會系統性低估」**沒發生**,v6 vs v5 CV-B 幾乎完全一致。這個 null result 改變了某些後續策略推論。

---

## 0. 這輪做了什麼(時間順序)

```text
1. GRU val curve diagnostic(fold-0,20 epochs)
   → 結論:val_loss 最低在 ep 11-12,curve 太雜訊,保留 GRU_EPOCHS=12。

2. 寫 gen_submission_v6_fullaug.py,套全部 Round 6 consensus 改動:
   - BETA_GRID 0→2.5(21pt)
   - WEIGHT_STEP 0.1→0.05
   - KMeans n_init 5→10
   - 加 fold-safety asserts(三條:match disjoint / rally_uid disjoint / RangeIndex)
   - 在 5-fold OOF 內 fully retrain TabPFN action(ManyClass)/ point / server +
     GRU(12 ep)— 不再依賴 v2 cached OOF
   - 加 per-stratum F1 報告(seen / rescued / cold,consensus method C)

3. 跑 v6,踩了 alignment bug,修了。
4. v6 完整跑完,8.5 分鐘 OOF + 1.8 分鐘 final training。
```

---

## 1. Alignment bug 紀錄(請看是否合理)

**情境**:首次跑 v6 第一個 fold 結束時 trip assert
`(gvA_f == eA[sfe]).all() AssertionError: fold 0: GRU val action labels != sampled-eval`

**根因(我推導出來的)**:
- `build(tr, "sampled")` 在**全部 14995 個 rally** 上 iterate,每個 non-skip rally 對 rng.choice 取一次 L → 產生 eA。
- 我原本寫 `build_seq(tr[match in val_matches], "sampled")` 只在 **val fold 的 rally 子集** iterate,rng.choice 序列被截掉 → 對同一個 rally 抽到的 L 跟 build 給的不同 → 不同 L 對應不同的 `na[L]` label → assert 失敗。

**修法**:
```python
# 改成在迴圈外一次跑全 train sampled,跟 build 同序、同 SEED → L 對齊
Call, Nall, Lall, gyAall, gyPall, gyRall = build_seq(tr, "sampled", tld)
assert len(gyAall) == len(eA) and (gyAall == eA).all(), \
    "global build_seq sampled labels misalign with build sampled"
# 然後迴圈內用 Call[sfe] / Nall[sfe] / Lall[sfe] 切 fold 出來
```

修完 assert 過了 → OOF 跑完。

**我自認可能還有的問題**:這個 assert 只檢查 action label;沒檢查 L 本身(`Lall[sfe]` vs `Xs.obs_len[sfe]`)。如果 RNG 在不同地方分歧但 label 巧合一樣,assert 過不會抓到。**你們覺得這個 assert 夠嚴格嗎?要不要再加 Lall[sfe] == Xs.obs_len[sfe] 的 assert?**

---

## 2. v6 完整 CV-B 結果

```text
=== v6 FULLY-AUGMENTED CV-B(BETA 0→2.5, WEIGHT_STEP 0.05) ===
action F1 = 0.3657   w = (LGBM 0.35, TabPFN 0.25, GRU 0.40)   β = 0.000   | CV-A F1 = 0.3510
point  F1 = 0.1941   w = (LGBM 0.35, TabPFN 0.30, GRU 0.35)   β = 0.000   | CV-A F1 = 0.1881
server AUC = 0.6151  w = (LGBM 0.15, TabPFN 0.55, GRU 0.30)
=> CV-B Overall = 0.3469
```

**vs 先前版本**:

| 版本 | action F1 (CV-B) | point F1 (CV-B) | server AUC | Overall CV-B |
|---|---|---|---|---|
| v3 (mixed-OOF, matchup, no aug) | ~0.354 | ~0.190 | ~0.614 | 0.350 |
| v5 (mixed-OOF, full aug) | 0.3651 | 0.1945 | 0.6145 | 0.3467 |
| **v6 (fully-aug OOF)** | **0.3657** | 0.1941 | **0.6151** | **0.3469** |

---

## 3. ★ 反直覺發現:你們擔心的 mixed-OOF artifact 沒出現

Round 6 你們 §1.2 的結論是:

> 目前觀察下:action +0.011 / point +0.005 / server ≈ unchanged,理論 overall 應該上升,但 mixed CV-B overall 反而 -0.003。這個矛盾很像 mixed OOF / weight-search artifact。

v6 fully-augmented OOF 跑完後:

```text
v5 mixed OOF Overall = 0.3467
v6 fully-augmented   = 0.3469   <-- 差 +0.0002
```

**v5 跟 v6 Overall 在第三位數內一致**。三個任務的數字也都幾乎相同。所以:
- **mixed OOF 不是 artifact** — 它的判斷其實是可靠的。
- v5 CV-B 0.347 vs v3 CV-B 0.350 那個 -0.003 是真正的雜訊(±0.005 噪音帶),不是「混合 calibration drift」。
- 也就是說 — **augmentation 的真實私人增益,可能比我們本來估的還要再保守一點**。

**我為什麼這樣解讀**(請挑):
- 如果 mixed OOF 真的系統性低估 v5,fully-augmented 應該明顯抬升,但我們看不到。
- 一個可能解釋:TabPFN/GRU 用 augmented features 跟用 non-augmented features 預測,大部分 calibration 重疊(features 變化集中在 player_prior / matchup,影響 OOF 預測機率向量但平均效果小)。
- 另一個可能:v5 mixed OOF 跟 v6 fully-aug 同時都受 CV 噪音影響,湊巧拉平。但這需要兩個雜訊源同向 → 不太可能。

**第一個解釋比較像真的**。如果對,後續含義:
- Priority 3(把 old samples 餵進 LGBM/TabPFN/GRU 訓練集)的預期增益也會更小 → 更應該不做。

→ **你們認同這個解讀嗎?還是有其他可能性?**

---

## 4. ★ Per-stratum F1(consensus method C,真正能講故事的數字)

```text
                  n     action F1   point F1
seen          11723       0.3723    0.1969
rescued         378       0.3518    0.1203   <-- 這就是 augmentation 救到的
cold           2894       0.2479    0.1542
```

**這是 augmentation 真正在做的事**:
- rescued 子集的 action F1(0.3518)**幾乎追上 seen**(0.3723) — 給選手一個 prior 的效果是巨大的。
- 沒擴充時這 378 個樣本會掉到 cold 那一行(F1 0.2479)。
- 所以 augmentation 對 rescued 樣本的 action F1 提升 = 0.352 − 0.248 = **+0.104 per-rally**。

**point 反而比 cold 還差**(0.120 vs 0.154):這也合理 — point 是 high-entropy(原本 oracle 診斷確認過),擴充給的 player point prior 反而引入噪音。**但 point 整體佔比小 + 影響小**,沒造成傷害。

**外推到私人 609 個 rally 的算術**(我自己算的,請挑):
```text
私人 stratum 比例(實際,從之前的覆蓋分析):
  seen = 0.586  rescued = 0.156  cold = 0.258

私人 action F1 估計:
  v6_action ≈ 0.586·0.372 + 0.156·0.352 + 0.258·0.248
            = 0.218 + 0.055 + 0.064 = 0.337
  
  沒擴充版本(rescued 也算 cold):
  no_aug_action ≈ 0.586·0.372 + (0.156+0.258)·0.248
               = 0.218 + 0.103 = 0.321
  
  Δ action ≈ +0.016  → ·0.4 weight = +0.0064 在 final

私人 point F1 估計(rescued 較差):
  v6_point ≈ 0.586·0.197 + 0.156·0.120 + 0.258·0.154
           = 0.115 + 0.019 + 0.040 = 0.174
  
  no_aug:
  ≈ 0.586·0.197 + (0.156+0.258)·0.154
  = 0.115 + 0.064 = 0.179
  
  Δ point ≈ −0.005  → ·0.4 = −0.002 在 final(小幅損失)

私人 server:0.6151,沒改變 → Δ = 0
```

**私人 Final Δ ≈ +0.0064 − 0.002 + 0 = +0.0044**

這個數字 vs Round 5 consensus 給的 conservative +0.006~+0.018:
- 我的算術估 +0.0044,**比 conservative 下限還低一點**。
- 主因是 point rescue **負貢獻**(rescued 反而比 cold 差)。
- action 的 +0.0064 落在預期範圍。
- 私人預期 final:`v2-equivalent baseline ≈ 0.345` + **0.004 ≈ 0.349**(現實主義估計)。

→ **這個算術你們同意嗎?有沒有其他乘子要加?(我有沒有忽略某個跨任務 interaction?)**

---

## 5. β=0 是怎麼回事?

兩個 macro-F1 任務(action / point)在 BETA_GRID 擴到 [0, 2.5] 後,最佳 β 都搜到 **0.000**(完全不做 prior correction)。

我的解讀:
- v6 的三個模型本身已經做了 calibration:LGBM 用 `class_weight='balanced'`(rare class 已加權),TabPFN ManyClass 自帶 ensemble 校準。
- 額外的 `p / prior^β` 校正反而會把已經校好的機率往「等比例放大 rare class」方向推 → 在 macro-F1 上得不償失。
- 過去 v2/v3 看到 β > 0 是因為當時 LGBM 設定或 ensemble 沒像 v6 這麼平衡。

**這合理嗎?還是 β=0 暗示我哪裡校準錯了?**(如 BETA_GRID 還是太狹窄?WEIGHT_STEP 0.05 找的不是真實最優?)

---

## 6. 程式碼節錄(請看這次新加 / 改的部分)

### 6.1 BETA_GRID + WEIGHT_STEP + KMeans + asserts

```python
# 全在 OOF 區段開頭設定,容易找
BETA_GRID = np.linspace(0, 2.5, 21)
WEIGHT_STEP = 0.05

# fit_clusters 內:
km = KMeans(k, n_init=10, random_state=SEED).fit(...)  # 5 -> 10

# 進 OOF 之前:
assert set(old.match).isdisjoint(set(tr.match)),     "old/train match leak"
assert set(old.rally_uid).isdisjoint(set(tr.rally_uid)), "rally_uid leak"
assert isinstance(Xa.index, pd.RangeIndex), "Xa index must be RangeIndex"
```

### 6.2 Per-fold fully-augmented OOF(★)

```python
# 全 train sampled seq 一次建好,跟 Xs 對齊
Call, Nall, Lall, gyAall, gyPall, gyRall = build_seq(tr, "sampled", tld)
assert (gyAall == eA).all()   # 證明 RNG 對齊

for f in range(5):
    trm = af != f; evm = sf == f   # all-prefix train / sampled-eval
    sft = sf != f; sfe = sf == f   # sampled-train / sampled-eval

    # 全部 fold-safe 統計都用 train_fold + old
    cat_la = np.concatenate([Xa.loc[trm, "_la"], Xao["_la"]])
    # ... 其他 concat
    T_ = fit_trans(...)
    dMa_, gAg, dMp_, gPg = player_dists(cat_nh, cat_yA, cat_yP)
    cl_ = fit_clusters(pd.concat([tr[trfold!=f], old]))
    cA_, cP_ = fit_matchup(cat_nh, cat_lh, cat_yA, cat_yP, cl_)

    # LGBM(action/point with matchup;server no player)
    ma = lgbc().fit(Xt_AP, yA[trm]);   LAc.append(...)
    mp = lgbc().fit(Xt_AP, yP[trm]);   LPc.append(...)
    mr = lgbc(False).fit(Xt_S, yR[trm]); LRc.append(...)

    # TabPFN(都用 augmented features)
    tA = ManyClassClassifier(TabPFN(...), alphabet_size=10, ...).fit(Xt_APp, eA[sft])
    TAc.append(align(tA.predict_proba(Xe_APp), tA.classes_, ACLS))
    tP = TabPFN(...).fit(Xt_APp, eP[sft]); TPc.append(...)
    tR = TabPFN(...).fit(Xt_Ss, eR[sft]);  TRc.append(...)

    # GRU(per-fold fresh train on full-prefix;predict on global sampled slice)
    Ca_f, Na_f, La_f, ... = build_seq(tr[match in trn_matches], "all", tld)
    gm_f = gru_train(..., ep=12)
    gA_f, gP_f, gR_f = gru_pred(gm_f, Call[sfe], Nall[sfe], Lall[sfe])   # ★ 用全局 slice

    # 分層
    fold_train_players = set(nha[trm])
    rescued_set = set(nhao) - fold_train_players
    is_seen = np.array([h in fold_train_players for h in nhs[sfe]])
    is_rescued = np.array([(h not in fold_train_players) and (h in rescued_set) for h in nhs[sfe]])
    is_cold = ~(is_seen | is_rescued)
```

### 6.3 Stratified F1 報告

```python
def f1_stratum(L, T, G, y, cls, pr, W, B, mask):
    if mask.sum() < 5: return float('nan')
    bl = W[0]*L + W[1]*T + W[2]*G
    return f1_score(y[mask], cls[(bl[mask]/np.clip(pr, 1e-9, None)**B).argmax(1)], average="macro")

for name, m in [("seen", SEEN), ("rescued", RESCUED), ("cold", COLD)]:
    fa_s = f1_stratum(LAc, TAc, GAc, YA, ACLS, prA, WA, BA, m)
    fp_s = f1_stratum(LPc, TPc, GPc, YP, PCLS, prP, WP, BP, m)
    print(f"{name} (n={m.sum()}): action={fa_s:.4f} point={fp_s:.4f}")
```

---

## 7. 自爆風險(這輪剩下的不確定)

```text
雷 1:RNG 對齊只 assert label,沒 assert L 本身。理論上 label 一致就等價,但
      若有極端 race condition 可能漏抓。(§1 已點出)

雷 2:Point rescue 是負貢獻(rescued 0.120 < cold 0.154)。我用「point 高 entropy
      所以 player prior 引入噪音」解釋,但這只是事後合理化。可能還有 confounder。

雷 3:β=0 對兩個 macro-F1 任務都是最佳 — 是否暗示 BETA_GRID 上限還不夠?還是
      LGBM/TabPFN 的內建校準已足?(§5)我傾向後者,但沒嚴格驗證。

雷 4:私人 Final 估計 +0.0044 比你們上次給的 conservative 下限 +0.006 還低,主要
      因為 point rescue 負貢獻。是否該對 point 用「不擴充」?也就是 player_dists
      P(point|player) 不用,只用 P(action|player)?這會破壞架構對稱性但可能救回 point。

雷 5:沒做 Round 6 consensus §7.3 的 cluster stability diagnostic(我跳過了)。
      cluster ID 跨 fold 不一致可能讓 matchup 變 noisy。但 KMeans n_init 從 5 → 10
      理論上已改善。要不要明確跑診斷?
```

---

## 8. 開放問題(請挑)

```text
Q1. mixed-OOF 不是 artifact 的解讀(§3)你們同意嗎?
Q2. Per-stratum 算術 +0.0044 你們算的是多少?有沒有忽略的乘子?
Q3. Point rescue 是負貢獻(雷 2、雷 4)— 要不要只擴充 action 不擴充 point?
Q4. β=0 是否暗示我有校準錯誤?(雷 3)
Q5. 要不要做 cluster stability diagnostic(雷 5)?ROI 如何?
Q6. v6 是不是該 ship 的 final candidate?還是有別的角度該驗證?
```

---

## 9. 我打算的下一步(等你們回再決定)

```text
1. 把 main.ipynb 升級到 v6(套全部 Round 6 consensus 改動 + per-stratum 報告)。
   ※ 這個會做,不等回覆。

2. 等你們回覆 Q1-Q6 後:
   - 如果 Q3 同意「不擴充 point」→ 跑 v7,只擴 action,看 final 是否真的回升。
   - 如果 Q5 要做 cluster diagnostic → 跑(快)。
   - 如果都不需要 → final = submission_v6-aug_incl0.csv,搞定收工。

3. 使用者今天提交額度用完,明天會上傳 v6-aug-ovr 看公開榜驗證 public 結構。
   如果 public 跳到 ~0.4 → 推論成立,最後上傳改 clean v6。
   如果沒跳 → 雷 1(public/private 切法)需要重估,你們會再被找來。
```

---

## 10. 一句話總結

```text
v6 fully-augmented OOF 跑完,CV-B 跟 v5 一致(0.3469 vs 0.3467)。
這個 null result 推翻了 mixed-OOF artifact 的假設。
Per-stratum 證實 augmentation 對 rescued 樣本的 action F1 大幅提升(+0.10),
但對 point 反而是負貢獻。私人最終 final Δ 算出來 ≈ +0.004,比預期保守。
v6 是目前最乾淨的 final candidate;唯一還在考慮的是要不要只擴 action 不擴 point。
```
