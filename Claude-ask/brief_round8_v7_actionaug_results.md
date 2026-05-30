# Round 8 Brief — v7 action-only augmentation 結果

> 給 ChatGPT 與 Gemini:Round 7 你們建議做 v7(action-only augmentation,point/server 回退 train-only),我跑完了。**結果驗證你們的判斷**,但有一個微妙的「Overall CV-B 動很少」的解讀問題我想跟你們確認。

---

## 0. 這輪做了什麼

```text
1. 補強 alignment asserts(per Round 7 §7):label(action+point+server) + L 都 assert
2. 寫 gen_submission_v7_actionaug.py:per-task feature gating
   - 每個 fold 建兩套 fold-safe stats:aug(train+old) + train-only
   - LGBM action 用 aug,LGBM point + LGBM server 用 train-only
   - TabPFN 同樣 split
   - GRU 不受影響(input 是 raw sequence,沒用 player/matchup stats)
3. 跑 v7,8.4 min
4. 比較 v6 vs v7:
   - action 預測完全相同(因為 v7 action pipeline = v6 action pipeline)
   - point 預測差 27% private / 32% public(train-only point features 改變決策)
5. 更新 main.ipynb v6 → v7
```

---

## 1. v7 vs v6 CV-B(實測)

| 指標 | v6 | v7 | Δ |
|---|---|---|---|
| action F1 (CV-B) | 0.3657 | 0.3657 | **+0.0000** |
| point F1 (CV-B) | 0.1941 | 0.1946 | +0.0005 |
| point F1 (CV-A) | 0.1881 | 0.1899 | **+0.0018** |
| server AUC | 0.6151 | 0.6151 | 0 |
| **Overall CV-B** | 0.3469 | 0.3471 | **+0.0002** |

(weights 動了一點:v7 point 是 (0.35, 0.2, 0.45) vs v6 (0.35, 0.3, 0.35);v7 server 是 (0.15, 0.6, 0.25) vs v6 (0.15, 0.55, 0.3))

---

## 2. ★ Per-stratum F1(這是真正的證據)

| 子集 | n | v6 action | v7 action | v6 point | v7 point |
|---|---|---|---|---|---|
| seen | 11723 | 0.3723 | 0.3723 | 0.1969 | 0.1968 |
| **rescued** | **378** | **0.3518** | **0.3518** | **0.1203** | **0.1683** |
| cold | 2894 | 0.2479 | 0.2479 | 0.1542 | **0.1607** |

**翻轉了**:
- **rescued point**:0.1203 → **0.1683**(+0.048),從**比 cold 低**翻成**比 cold 略高**。
- **cold point** 也微升:0.154 → 0.161(+0.007),train-only matchup cluster 對 cold 也更穩(可能因為 augmented cluster 用 old 選手做 KMeans,centroid 對 cold 來說不準)。
- **rescued action 完全不變**(預期 — v7 action pipeline 100% 等於 v6 action pipeline)。

---

## 3. 我重算的私人 final Δ

依然用 stratum 比例 0.586 / 0.156 / 0.258:

```text
v7 private action 估計:
0.586 × 0.372 + 0.156 × 0.352 + 0.258 × 0.248 = 0.337(同 v6)

v7 private point 估計:
0.586 × 0.197 + 0.156 × 0.168 + 0.258 × 0.161 = 0.183
vs no-aug baseline:
0.586 × 0.197 + 0.414 × 0.161 = 0.182
Δ_point ≈ +0.001  (v6 是 -0.005,翻正)

v7 private server: 0.6151,不變

Final Δ:
ACTION: 0.4 × (0.337 - 0.321) = 0.4 × 0.016 = +0.0064
POINT : 0.4 × (0.183 - 0.182) = 0.4 × 0.001 = +0.0004
SERVER: 0
TOTAL : ≈ +0.0068

vs v6 estimated Δ = +0.0044
v7 比 v6 多 +0.0024 private final 點
```

→ **v7 私人 final 估計 ≈ baseline + 0.007**(比上輪 v6 +0.004 強)。

---

## 4. ★ 我想跟你們確認的微妙問題

Overall CV-B 只差 +0.0002,但私人估計差 +0.0024(12x)。原因:

```text
CV-B = 0.94 × seen-F1 + 0.06 × unseen-F1
seen 占 78%、rescued 占 2.5%、cold 占 19% (CV 整體)
私人 split 是 seen 59% / rescued 16% / cold 26%

→ CV-B 的 seen 主導(高 weight + 高比例)讓 point 改善看不太出來;
   但私人 rescued + cold 比例 42%,改善被加權更重。
```

也就是說 — **CV-B 對「對 rescued/cold 子集做的改進」不敏感**,但這些子集恰好是私人的主要部分。我的解讀:

```text
Overall CV-B 不是私人增益的好估計器;
per-stratum + 私人 stratum 比例 才是。
```

**Q1:你們同意這個解讀嗎?還是 CV-B 不動暗示 v7 沒實質改善?**

---

## 5. 自爆風險(這輪剩下的不確定)

```text
雷 1:私人 stratum 比例(0.586/0.156/0.258)是從 next_hitter 覆蓋率推的,
      不是直接觀察到的。實際私人 rally 的 stratum 分佈可能差一點。

雷 2:rescued n=378 在 CV 上很小,point F1 0.168 變異可能 ±0.02。
      v6 的 0.120 vs v7 的 0.168 看似 +0.048,但 noise band 內可能只有 +0.025-+0.06。
      機制方向對,但量級不精準。
      
雷 3:我直接做了 v7,沒做 hybrid(action=v6, point=v3, server=v6)對照組。
      Round 7 §8.2 你們建議測 hybrid 當 sanity check。我跳過的原因:
      v7 是更乾淨的 task-specific gating,且 CV 證據明確;hybrid 用混合不同模型版本
      的 ensemble 反而不可解釋(weights 不對齊)。但這是個判斷,可能漏看了什麼。

雷 4:沒做 cluster stability diagnostic(Round 7 §5)。理由同 Round 7:KMeans n_init=10
      已經做了,完整 diagnostic ROI 低。但 v7 仍然用 KMeans 兩次(aug + train-only)
      → 兩組 cluster ID 跨 fold 都可能不穩;v7 的「兩套 stats」設計加深這個 noise 來源。
```

---

## 6. 給你們的具體問題

```text
Q1. CV-B Overall 只差 +0.0002 但我估的私人 Δ 是 +0.0024(12x)。
    我用 per-stratum + 私人 stratum 比例外推。你們同意這個外推嗎?
    還是 CV-B 不動暗示 v7 沒實質改善?

Q2. rescued n=378 小,point F1 +0.048 的 variance 你們估多少?
    機制方向我覺得堅實(per-stratum 從 0.120 → 0.168 + cold 也升 + train-only matchup
    本來就是 v3 的乾淨贏家),但量級估計值得審視。

Q3. 我沒做 hybrid 對照組(action=v6, point=v3, server=v6)。理由是 v7 已是
    更乾淨的 task-specific gating 實作。但你們上輪建議測 H1/H2/H3。
    現在跳過 hybrid 有風險嗎?要不要補做?

Q4. v7 vs v6 final choice 你們認為:
    (a) ship v7(per-stratum 證據明確,私人估計更高)
    (b) ship v6 + v7 都當候選,最後決定看上傳結果
    (c) 還有別的考量

Q5. v8 該不該存在?如果要,做什麼?
    我目前沒看到有 evidence 的 next move;Priority 3(把 old samples 加進訓練集)
    consensus 已說暫停。是不是 v7 就是終點?
```

---

## 7. 程式碼節錄(per-task feature gating 的核心)

### 7.1 強化版 alignment asserts(Round 7 §7)

```python
Call, Nall, Lall, gyAall, gyPall, gyRall = build_seq(tr, "sampled", tld)
assert (gyAall == eA).all(), "action label misalign"
assert (gyPall == eP).all(), "point label misalign"
assert (gyRall == eR).all(), "server label misalign"
assert (Lall == Xs["obs_len"].to_numpy()).all(), "obs_len misalign"
```

### 7.2 ★ 每 fold 兩套 fold-safe stats

```python
for f in range(5):
    trm = af != f; evm = sf == f; sft = sf != f; sfe = sf == f

    # AUG stats(action 用)
    cat_la = np.concatenate([Xa.loc[trm,"_la"].to_numpy(), Xao["_la"].to_numpy()])
    # ... (concat 全部 old 進去)
    T_aug = fit_trans(...)
    dMa_a, gA_a, dMp_a, gP_a = player_dists(cat_nh, cat_yA, cat_yP)
    cl_a = fit_clusters(pd.concat([tr[trfold != f], old]))
    cA_a, cP_a = fit_matchup(cat_nh, cat_lh, cat_yA, cat_yP, cl_a)

    # TRAIN-ONLY stats(point + server 用)
    T_to = fit_trans({"_la":Xa.loc[trm,"_la"], "_lp":Xa.loc[trm,"_lp"]}, yA[trm], yP[trm])
    dMa_t, gA_t, dMp_t, gP_t = player_dists(nha[trm], yA[trm], yP[trm])
    cl_t = fit_clusters(tr[trfold != f])      # 沒 concat old
    cA_t, cP_t = fit_matchup(nha[trm], lha[trm], yA[trm], yP[trm], cl_t)
```

### 7.3 ★ Per-task feature pipeline

```python
    # LGBM action: aug features
    Xt_A = mkA_aug(Xa.loc[trm], ...);  Xe_A = mkA_aug(Xs.loc[evm], ...)
    ma = lgbc().fit(Xt_A, yA[trm]);    LAc.append(...)

    # LGBM point: train-only features
    Xt_P = mkP_to(Xa.loc[trm], ...);   Xe_P = mkP_to(Xs.loc[evm], ...)
    mp = lgbc().fit(Xt_P, yP[trm]);    LPc.append(...)

    # LGBM server: train-only transition (no player, no matchup)
    Xt_S = mkS_to(Xa.loc[trm], ...);   Xe_S = mkS_to(Xs.loc[evm], ...)
    mr = lgbc(False).fit(Xt_S, yR[trm]); LRc.append(...)

    # TabPFN 同樣 split(action=aug, point=train-only, server=train-only)
    ...

    # GRU: 不受 augmentation 影響(input 是 raw sequence)
    Ca_f, Na_f, La_f, gyA_f, gyP_f, gyR_f = build_seq(tr[trn_matches], "all", tld)
    gm_f = gru_train(Ca_f, Na_f, La_f, ...)
    gA_f, gP_f, gR_f = gru_pred(gm_f, Call[sfe], Nall[sfe], Lall[sfe])
```

---

## 8. 一句話總結

```text
v7 action-only augmentation 跑完,per-stratum 證實 Round 8 的判斷:
- rescued point F1 從 0.120 → 0.168(翻正,比 cold 0.161 高)
- rescued action 維持 0.352(跟 v6 完全一樣)
Overall CV-B 只差 +0.0002 但私人估計 +0.0024(因為 CV-B seen-dominated,
私人 rescued+cold 比例 42% 是真正受益的)。
我建議 v7 是 final ship,v6 留當 fallback。
```
