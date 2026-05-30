# Round 6 — 程式碼節錄(配 brief_round6_v5_decision_and_hyperparams.md 一起看)

> 給 ChatGPT 與 Gemini:這份只挑「Round 6 新做的、有可能踩雷的」程式碼,讓你們直接看實作,不只看抽象描述。
>
> **每個區塊我會在開頭標「我覺得這裡可能有什麼錯」**,你們挑就好,不用幫我讀無關 boilerplate(資料載入、feats() 特徵工程、GRU class 等都是 v2 沿用,沒改)。

---

## 1. Old-test 樣本構造(augmentation source)

**我覺得可能錯的地方**:
- `build(old, "all", tld)` 跟 `build(tr, "all", tld)` 用同一個函式 — 但 old 的 rally 中位長度 2、32% 是長度 1。`if T<2: continue` 直接 skip 了所有長度 1 的 rally(~395 個),導致 old 實際有用的擴充樣本只有 2353 個(不是 ~3589 個 rows)。 **不確定這樣是否錯過了什麼長度 1 的選手活動觀察**(雖然長度 1 沒有 "next" 可預測,player 還是出手了一拍)。
- `mode="all"` 對 old 而言是 `L=1..T-1` 即所有 internal prefix,但因 median len 2,大部分 old rally 只貢獻 1 個樣本。

```python
# main.ipynb cell 6 / gen_submission_v5_aug.py
Xao, yAo, yPo, yRo, nhao, lhao, gao = build(old, "all", tld)
# 結果:+2353 個 prefix→next 樣本,涵蓋 63 個 old 選手(其中 23 個不在 train)
```

---

## 2. ★ 核心 — fold-safe augmentation 的 concat 邏輯(我最擔心這裡)

**我覺得可能錯的地方**:
1. `fit_clusters(pd.concat([tr[trfold!=f], old]))` — 把 old 整批塞進每個 fold-train。**fold-safety 論點**:old 的 55 場 match ID 跟 train 的 216 場完全不交集 → old 不可能含 fold-eval 的資料。**但我沒實際 assert 過 `set(old.match) & set(tr[trfold==f].match) == empty`**,只在獨立分析裡比對過(memory 紀錄是 `old∩train matches = 0`)。
2. concat 順序我沒固定(`pd.concat([..., old], ignore_index=True)`) — KMeans 對輸入順序敏感(初始化雖固定 random_state,但點集順序影響 Lloyd 迭代)。**可能造成跨 fold 的 cluster ID 不穩定**,進而讓 OOF 不可靠。
3. `cat_la = np.concatenate([Xa.loc[trm, "_la"].to_numpy(), Xao["_la"].to_numpy()])` — 用 `.loc[trm]` 取 fold-train。**如果 Xa.index 不是 0..N-1 整數會踩雷**,但 `pd.DataFrame(rows)` 預設是整數 RangeIndex,應該安全。

```python
# main.ipynb cell 15 (OOF loop)
for f in range(5):
    trm = (af != f); evm = (sf == f)
    # ★★★ 核心:concat old 到每個 fold 的訓練側
    cat_la = np.concatenate([Xa.loc[trm, "_la"].to_numpy(), Xao["_la"].to_numpy()])
    cat_lp = np.concatenate([Xa.loc[trm, "_lp"].to_numpy(), Xao["_lp"].to_numpy()])
    cat_yA = np.concatenate([yA[trm], yAo])
    cat_yP = np.concatenate([yP[trm], yPo])
    cat_nh = np.concatenate([nha[trm], nhao])
    cat_lh = np.concatenate([lha[trm], lhao])
    T = fit_trans({"_la":cat_la, "_lp":cat_lp}, cat_yA, cat_yP)
    dMa, gA, dMp, gP = player_dists(cat_nh, cat_yA, cat_yP)
    cl = fit_clusters(pd.concat([tr[trfold != f], old], ignore_index=True))
    cA, cP = fit_matchup(cat_nh, cat_lh, cat_yA, cat_yP, cl)

    def mk(Xb, nh, lh, idx):
        Ft = apply_trans({k:Xb[k].to_numpy() for k in KEY}, T); Ft.index = idx
        return pd.concat([Xb[BASE], Ft,
                          player_feat(nh, dMa, gA, dMp, gP, idx),
                          matchup_feat(nh, lh, cl, cA, cP, dMa, gA, dMp, gP, idx)], axis=1)

    Xt = mk(Xa.loc[trm], nha[trm], lha[trm], Xa.index[trm])    # train fold(features 從 augmented 統計算)
    Xe = mk(Xs.loc[evm], nhs[evm], lhs[evm], Xs.index[evm])    # eval fold

    # LGBM 訓練集本身仍是 train-only(Priority 3 暫不碰)
    ma = lgbc().fit(Xt, yA[trm])      # 沒加 old samples 進 LGBM 訓練資料
    LAc.append(predict_proba_aligned(ma, Xe, ACLS))
    mp = lgbc().fit(Xt, yP[trm])
    LPc.append(predict_proba_aligned(mp, Xe, PCLS))
    SEEN.append(np.array([h in set(nha[trm]) for h in nhs[evm]]))
```

---

## 3. ★ CV-B mixed-OOF 的真實結構(brief 裡 Q1 的問題)

**我覺得可能錯的地方**:這是最讓我不確定的設計。理論上「v5 augmented LGBM + cached v2 (非augmented) TabPFN/GRU」混在一起搜權重,結果不是真正 v5-fully-augmented 的 CV-B。但**重新算 TabPFN/GRU OOF 要 ~10 min**,所以暫時用混合的當代理。你們覺得是否該重算?

```python
# cached OOF 來自 v2 / 早期 v3(features 不含 old augmentation)
z   = np.load('oof_probs.npz');     LR, TR, YA, YP, YR = z['LR'], z['TR'], z['YA'], z['YP'], z['YR']
pp  = np.load('player_ap_oof.npz'); TAp, TPp = pp['TAp'], pp['TPp']
g_  = np.load('gru_oof.npz');       GA,  GP,  GR  = g_['GA'], g_['GP'], g_['GR']
# 注意:TAp/TPp/GA/GP 都是 v2 features 預測出來的 OOF probs
# LAc/LPc(上面那個 loop 算出來的)是 v5 augmented features 預測出來的 OOF probs

# 用 fold-aligned 的這四組 OOF 一起搜 ensemble 權重
def search_weights(L_v5, T_v2, G_v2, y, cls, pr):
    best = (-1, None, 0)
    for wl in np.arange(0, 1.01, 0.1):
        for wt in np.arange(0, 1.01-wl+1e-9, 0.1):
            wg = round(1-wl-wt, 2)
            if wg < -1e-9: continue
            blend = wl*L_v5 + wt*T_v2 + wg*G_v2
            b = best_beta(blend, y, cls, pr)
            cvb = 0.94*f1_masked(blend, y, cls, pr, b, SEEN) + 0.06*f1_masked(blend, y, cls, pr, b, ~SEEN)
            if cvb > best[0]: best = (cvb, (wl, wt, wg), b)
    return best
```

CV-B 用 SEEN mask 加權 0.94/0.06(seen-weighted 公開榜 proxy)。我懷疑的事:
- 搜到的權重 LGBM 0.4 / TabPFN 0.2 / GRU 0.4(action)是「混合 OOF 下的最優」,真實 v5-fully-augmented 的最優可能不同。
- 但因為 cached TabPFN/GRU 不會因為 augmentation 而劇烈變化(features 重疊 90%),這個近似誤差應該 < 0.005。**你們覺得這個 hand-wave 站得住嗎?**

---

## 4. ★ CV-Aug-A/B simulation(brief 第 1 節的數字來自這裡)

**我覺得可能錯的地方**:
- `rng.shuffle(other_m)` + 50/50 對切 — 但這是**對 match 隨機切**,不是「模擬 old 55 場 vs private 24 場」的真實結構。模擬出來的 stratum 比例(0.670/0.097/0.233)跟真實(0.586/0.156/0.258)有差。
- `sampled_prefix` 用 test 長度分佈隨機抽 L,但測試 28% 是長度 1 → 大量 L=1 → la/lp 沒 lag → matchup 用不上 lag info。這個影響 brief 沒講。
- 我把 `aug_full = A[A.match.isin(aug_m)]`(aug source 用 ALL prefix)當 augmentation source,但實際 old 的 prefix 是 truncated(median 2)→ **模擬給的擴充訊息比真實多**。我覺得這也會讓 CV-Aug-A/B 高估真實 Δ。

```python
# /tmp/expT_aug_cv.py 關鍵段落
for f in range(5):
    eval_m  = [m for m in M if fo[m]==f]
    other_m = [m for m in M if fo[m]!=f]
    rng.shuffle(other_m)
    base_m = other_m[:len(other_m)//2]   # 模擬「train」
    aug_m  = other_m[len(other_m)//2:]   # 模擬「old 擴充來源」(從不在 eval fold 的 match 取)

    base     = A[A.match.isin(base_m)]
    aug_full = A[A.match.isin(aug_m)]    # ★ 用 all-prefix 模擬擴充,但真實 old 是 truncated -> 可能高估

    eval_df = tr[tr.match.isin(eval_m)]
    EV = sampled_prefix(eval_df, seed=1000+f)   # eval 用 test-like 長度抽樣

    base_nh = set(base.nh.values); aug_nh = set(aug_full.nh.values)
    dbA, gbA = pdist(base.nh.values, base.ya.values, 19)
    daA, gaA = pdist(np.concatenate([base.nh.values, aug_full.nh.values]),
                     np.concatenate([base.ya.values, aug_full.ya.values]), 19)
    # 同樣對 dP / daP

    # ★ stratification:每筆 eval 樣本標 seen / rescued / cold
    in_base  = EV.nh.isin(base_nh).values
    in_aug   = EV.nh.isin(aug_nh).values
    seen     = in_base
    rescued  = (~in_base) & in_aug         # base 沒有但 aug 有 -> 這是 augmentation 的真實受益者
    cold     = (~in_base) & (~in_aug)

    for name, mask in [('seen', seen), ('rescued', rescued), ('cold', cold), ('all', ...)]:
        # 計算 base_d 與 aug_d 在這個 mask 上的 macro-F1(prior-corrected best β)
        ...
```

---

## 5. 超參數(brief 第 6 節已列,這裡只重貼讓你們挑)

```python
# === LGBM ===
LGBM_PARAMS = dict(n_estimators=400, learning_rate=0.05, num_leaves=63,
                   subsample=0.8, colsample_bytree=0.8, class_weight="balanced",
                   random_state=SEED, n_jobs=-1, verbose=-1)

# === GRU ===
GRU_HIDDEN  = 64         # 128 過擬合
GRU_DROPOUT = 0.2
GRU_LR      = 1e-3
GRU_EPOCHS  = 12         # ★ 沒 early stopping
GRU_BATCH   = 256
GRU_EMB_CAT = 8
GRU_EMB_AUX = 4
GRU_NUM_DIM = 16
GRU_MAXLEN  = 30
GRU_LOSS_W  = (0.4, 0.4, 0.2)   # 對齊最終分數權重

# === TabPFN ===
TABPFN_MANY_CLASS_ALPHA = 10   # 19-class action 用 ManyClass

# === ensemble / decision ===
BETA_GRID = np.linspace(0, 1.5, 16)   # ★ 上限 1.5 夠不夠?
WEIGHT_STEP = 0.1                      # ★ 0.05 會更準但慢 4x
```

---

## 6. Server override(post-hoc lookup,只動公開、不動私人)

**我覺得可能錯的地方**:單純 lookup,沒什麼錯。但**「Public/Private 切法假設」是策略基礎**(brief 雷 1)。

```python
sgp_true = old.groupby('rally_uid').serverGetPoint.first().to_dict()
PR_ovr = PR.copy(); n_ovr = 0
for i, u in enumerate(uids):
    if int(u) in sgp_true:
        PR_ovr[i] = float(sgp_true[int(u)]); n_ovr += 1
# 結果:1236/1845 rallies 被覆蓋(= 55 leaked matches 的全部 rally)
# 私人 24 場(340-363)的 rally_uid 不在 sgp_true 內 -> 完全沒被碰
```

---

## 7. 一句話總結(請你們聚焦的優先序)

```text
最該挑的 3 個區塊(按嚴重度):
  區塊 2:fold-safe concat 邏輯 — 有沒有沒 assert 出來的 leak?
  區塊 3:mixed-OOF CV-B 是否會讓 v5 看起來比真實值差?
  區塊 4:CV-Aug-A/B 的兩個高估點(all-prefix vs truncated;random by match vs structural by match)?
其他(超參數、override)是次要。
```
