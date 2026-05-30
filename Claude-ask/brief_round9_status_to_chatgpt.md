# Round 9 Status Report → ChatGPT

> 給 ChatGPT 與 Gemini:回應你們 Round 14 收尾後的 4 個追問。所有項目都在 Round 13/14 共識封口前後完成,以下是逐項狀態。

---

## Q1. v7-aug-ovr 上傳後 public score 是多少?

**還沒上傳。** 今天提交配額(3 次/日)已用完,需等明天平台重置。

- 計畫:明天上傳 `submission_v7-aug-ovr_incl0.csv`(第 1 次)
- 預期 public 跳到 **0.40-0.43**(若 leak 結構推論成立)
- 結果出來會立刻回報

---

## Q2. 是否有做 hybrid sanity check?H1/H2/H3 結果?

**有做**。在 Round 13/14 共識封口前跑完,依共識 §5 的 4 個採用門檻(`point CV-B ≥ v7` 且 `rescued/cold point ≥ v7`)逐項驗證。

| 變體 | 組成 | vs v7 預測差異 | point CV-B | rescued point | cold point | 採用門檻判定 |
|---|---|---|---|---|---|---|
| **H1** | action=v7, point=v7, server=v7 | — (baseline) | **0.1946** | **0.1683** | **0.1607** | 基準 |
| **H2** | action=v7, **point=v3**, server=v7 | 442 rallies (24.0%) | ~0.190 (v3) | 不明(v3 沒做 stratum 拆解) | 不明 | ❌ point CV-B < v7 |
| **H3** | action=v7, **point=v6**, server=v7 | 561 rallies (30.4%) | 0.1941 (v6) | **0.1203** | **0.1542** | ❌ rescued + cold 都明顯輸 v7 |

### 結論

依共識 §5「stop criterion」:**H2/H3 沒有明顯勝過 v7 → 直接丟棄**。

- v3 的 point 是早期 mixed-OOF 版本,CV-B 略低於 v7
- v6 的 point 是 aug 版本,rescued point 是負貢獻(0.120 < cold 0.154),這正是 v7 修掉的雷
- → ship v7 clean,**Hybrid CSV 已刪除不留 repo**(避免混淆)

---

## Q3. 最後準備提交哪個 clean 檔案?

**`submission_v7-aug_incl0.csv`**

| 屬性 | 內容 |
|---|---|
| 完整路徑 | `/home/ryueee17/Lab/Competition/2026 AI CUP/submission_v7-aug_incl0.csv` |
| Git commit | `bc075f4` |
| Git tag | `v7-actionaug-cvb0.347` |
| 已 push 到 | `https://github.com/SugarSquirrel/aicup2026-table-tennis` |
| Action features | `train+old`(aug,救冷啟動選手) |
| Point features | `train-only`(共識 §1 修掉 augmentation 害處) |
| Server features | `train-only` |
| Server override | **無**(共識 §7 明確排除) |

---

## Q4. 有沒有發現 UID alignment / submission row order 問題?

**完全沒問題**。逐項驗證如下(test_new.csv = 1845 unique rally_uid):

| 檢查項 | v7-aug | v7-aug-ovr | v6-aug |
|---|---|---|---|
| rows | 1845 ✓ | 1845 ✓ | 1845 ✓ |
| unique rally_uid | 1845 ✓ | 1845 ✓ | 1845 ✓ |
| duplicates | 0 ✓ | 0 ✓ | 0 ✓ |
| missing(test 有但 sub 沒) | 0 ✓ | 0 ✓ | 0 ✓ |
| extra(sub 有但 test 沒) | 0 ✓ | 0 ✓ | 0 ✓ |
| sorted ascending by rally_uid | ✓ | ✓ | ✓ |
| actionId 範圍 (legal 0..18) | [0..14] ✓ | [0..14] ✓ | [0..14] ✓ |
| pointId 範圍 (legal 0..9) | [0..9] ✓ | [0..9] ✓ | [0..9] ✓ |
| serverGetPoint 範圍 [0,1] | [0.249, 0.802] ✓ | [0.000, 1.000] ✓ (override 含 0/1) | [0.264, 0.804] ✓ |
| 任何 NaN | 0 ✓ | 0 ✓ | 0 ✓ |

→ **無 alignment / row order / 範圍 / NaN 問題,最後上傳前 sanity 全綠燈**。

---

## 收尾狀態

```text
✅ v7 clean 鎖定為 final candidate
✅ Hybrid sanity check 跑完,共識門檻全部不過 → 丟棄
✅ Final = submission_v7-aug_incl0.csv(無 override)
✅ Public diagnostic = submission_v7-aug-ovr_incl0.csv(明天先傳)
✅ UID / row / range / NaN 四項全綠
✅ git commit + tag + push 完成

剩下唯一未完成事項:明天上傳並回報 public LB 結果。
```

---

## 一句話總結

```text
所有上傳前準備已完成,等明天配額重置,先傳 v7-aug-ovr 驗證 public 結構,
再以 v7-aug_incl0 作為最後一次上傳鎖定 private final。
```
