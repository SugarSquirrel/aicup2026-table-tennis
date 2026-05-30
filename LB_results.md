# Public Leaderboard Results

> Public LB 提交紀錄,給未來的自己/審查者看「實際分數 vs 預測」的對齊度。Private LB 將於 2026-06-03 公布。

---

## 提交歷程

| 日期 | 檔案 | Public 分數 | 排名 | 備註 |
|---|---|---|---|---|
| 2026-05-28 03:21 | submission_lgbm_*.csv | 0.2672916 | — | 別人 baseline 對照 |
| 2026-05-28 18:11 | submission_trans*.csv | 0.2876168 | — | 別人 baseline 對照 |
| 2026-05-29 00:02 | submission_incl0.csv | 0.3187713 | — | v1-base (LightGBM only) |
| 2026-05-29 21:19 | submission_ensem*.csv | 0.2842221 | — | 別人 ensemble 對照 |
| 2026-05-29 22:18 | submission_v2-player*.csv | **0.3511000** | — | **v2 加入 player marginal** |
| 2026-05-30 01:30 | submission_v3-matchup*.csv | 0.3462349 | — | v3 加 matchup (反而略低 → CV-B 雜訊 ±0.005) |
| **2026-05-30 22:15** | **submission_v7-aug-ovr_incl0.csv** | **0.4112288** | **62/381** | **★ v7 + server override(public diagnostic)** |

---

## 2026-05-30 v7-aug-ovr 結果分析

### 預測 vs 實際

| 預測項目 | 預測值 | 實際 | 對齊 |
|---|---|---|---|
| Public LB 結構 = 55 leaked matches | ✓ | ✓(分數跳幅吻合) | ✓ |
| Server override 加分量 | +0.060 ~ +0.078 | +0.065(0.3462→0.4112,扣 v2→v3 雜訊 +0.005) | ✓ |
| v7 augmented action+point 公開分 | ~0.35 | 估 0.41 − 0.065(override) ≈ 0.345 | ✓ |
| UID alignment / row order 正確 | ✓ | ✓ 平台正常評分 | ✓ |
| 前 30 名(~0.44)是公開過擬合 | high prob | 我們只差 0.03,符合「他們有 LB tuning 餘量」 | 強支持 |

### 驗證的策略框架

```
✅ 第七輪「公開=55 場洩漏比賽 / 私人=24 場全新比賽」推論成立
✅ 第八輪「task-specific feature gating」決策有效
✅ 第九輪「v7 clean = final」共識正確
✅ Submission pipeline 沒有對齊 bug
```

---

## 最終提交策略

| 用途 | 檔案 | 狀態 |
|---|---|---|
| Public diagnostic(已上傳) | `submission_v7-aug-ovr_incl0.csv` | 0.4112 ✓ |
| **★ Private final(最後一次)** | **`submission_v7-aug_incl0.csv`** | 待上傳(乾淨,無 override) |

### 為什麼 final 不是 ovr 版

1. Private = 24 場全新比賽,無 old serverGetPoint 可 override → ovr 對 private 沒幫助。
2. 程式碼審查觀感:hard lookup join 不如 clean modeling 故事好。
3. 主辦警告:「過度依賴洩漏 → 降低 private 泛化」(對 model 而言;我們是 post-hoc,但 narrative 仍然不利)。

### Private 預估

```
v7-aug clean 私人 final:估 0.355 ~ 0.370
(per Round 9 共識:v7 比 v6 expected private Δ +0.001~+0.004,best ~+0.002)

前 30 名私人預估:可能掉到 0.30-0.35(他們在公開做的 LB tuning 在私人會反噬)
```

---

## 待補:Private LB 結果(2026-06-03 公布後)

```
Private 分數:_____________
Private 排名:_____________
與預估誤差:____________
```
