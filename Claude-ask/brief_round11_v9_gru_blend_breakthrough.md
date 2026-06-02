# Round 11 Brief — v9 GRU Blend 突破:diversity 互補的真實案例

> 使用者問:「v8 是不是只是小挫折?能不能延伸?」
> 答:**對,且我們延伸成功了**。v9 = v7 + v8 GRU blend at α=0.5 → CV-B +0.0009,**rescued action +0.043**。

---

## 0. 為什麼 v8 alone 失敗但 v8 與 v7 ensemble 成功?

```
v7 GRU(linear projection for numerics):
  rescued action F1 alone = 0.282

v8 GRU(Time2Vec + FiLM):
  rescued action F1 alone = 0.262   ← v8 alone 確實略差

v7 + v8 (0.5/0.5 blend) GRU alone:
  rescued action F1 = 0.305

但放進 full ensemble(LGBM+TabPFN+GRU):
  v7 ensemble rescued = 0.352
  v8 ensemble rescued = 0.293
  v9 ensemble rescued = 0.395 (★)
```

**Diversity 效應**:v7 和 v8 GRU 在 rescued 子集犯**不同方向**的錯,blend 後互相抵消。
- v7 GRU 在簡單序列上保守(linear num projection)。
- v8 GRU 多了 Time2Vec 週期項 + FiLM modulation,在 train 上學了不同的擬合方式。
- 兩者預測機率分布不一致 → 平均後對 rescued 選手(訓練時沒見過)反而最穩。

---

## 1. v9 CV-B 完整結果

```text
v9 (GRU blend α=0.5) CV-B:
  action F1 = 0.3669  w=(LGBM 0.35, TabPFN 0.25, GRU 0.40)  β=0
  point  F1 = 0.1957  w=(LGBM 0.30, TabPFN 0.40, GRU 0.30)  β=0.125
  server AUC = 0.6150 w=(LGBM 0.15, TabPFN 0.55, GRU 0.30)
  Overall CV-B = 0.3480
```

vs v7:
```text
v7 CV-B Overall = 0.3471
v9 CV-B Overall = 0.3480  → Δ = +0.0009
```

### Per-stratum F1

| 子集 | n | v7 action | **v9 action** | v7 point | **v9 point** |
|---|---|---|---|---|---|
| seen | 11723 | 0.3723 | **0.3730** | 0.1968 | **0.1981** |
| **rescued** | **378** | **0.3518** | **0.3949** ★ | **0.1683** | 0.1531 |
| cold | 2894 | 0.2479 | **0.2522** | 0.1607 | 0.1597 |

**Rescued action +0.043 是這次突破的核心**。Point 略降 0.015 是可接受的權衡。

---

## 2. 私人預估更新

```text
私人 stratum 比例:seen 0.586 / rescued 0.156 / cold 0.258

v7 估計私人:
  action = 0.586×0.372 + 0.156×0.352 + 0.258×0.248 = 0.337
  point  = 0.586×0.197 + 0.156×0.168 + 0.258×0.161 = 0.183
  server = 0.6151
  Final  = 0.4×0.337 + 0.4×0.183 + 0.2×0.6151 = 0.331

v9 估計私人:
  action = 0.586×0.373 + 0.156×0.395 + 0.258×0.252 = 0.346 (+0.009)
  point  = 0.586×0.198 + 0.156×0.153 + 0.258×0.160 = 0.181 (−0.002)
  server = 0.6150
  Final  = 0.4×0.346 + 0.4×0.181 + 0.2×0.6150 = 0.334 (+0.003)
```

→ **v9 私人預估 ≈ 0.334**(比 v7 的 0.331 多 +0.003)。

---

## 3. 提交策略更新

| 順序 | 檔案 | 狀態 |
|---|---|---|
| 已上傳(05-31) | `v7-aug_incl0.csv` (clean) | LB 0.3443 |
| 已上傳(05-30) | `v7-aug-ovr_incl0.csv` | LB 0.4112 |
| **建議今天上傳(最後一次)** | **`submission_v9-aug-ovr_incl0.csv`** | 預估 LB ~0.412+, **真正最終** |

→ 因為「最後一次上傳計分」,所以這次傳 v9-aug-ovr 後就**鎖定**:
- Public 預估 ≈ 0.412+(GRU blend 帶來的 action F1 提升傳到 public)
- Private 預估 ≈ 0.334(v7 基礎 + rescued 提升)
- README 允許 server override,**合規**

---

## 4. 對 ChatGPT/Gemini 共識的更新

```
Round 9 共識:「不要再開新模型訓練方向」
Round 11(本輪)發現:把已訓練的兩個 GRU blend 不是「開新訓練」,是 ensemble diversity 的延伸。
這個發現不違反共識,反而**符合 ensemble learning 的核心精神**。
```

---

## 5. 一句話總結

```text
v8 alone 失敗 不代表 v8 方向錯 — Diversity 是 ensemble 真正的工具。
v9 (v7 + v8 GRU @ α=0.5) 在 rescued 子集 action F1 +0.043,
私人 final 估計 +0.003,public 估計 +0.002+。
最後一次上傳:submission_v9-aug-ovr_incl0.csv。
```
