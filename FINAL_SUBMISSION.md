# 最終提交版本確認(給組員看的)

> 這份文件是用來明確記錄「我們真正上傳的版本是哪個」。組員看這份就知道:
> - 哪個 csv 真的被上傳
> - 哪個 script 產生這個 csv
> - 怎麼重現
> - LB 結果

---

## ✅ 最終上傳檔案

| 項目 | 值 |
|---|---|
| **上傳檔名** | `submission_FINAL.csv` |
| **實際內容** | 與 `submission_v14-aug-ovr_incl0.csv` 完全相同 |
| **SHA256** | `f1e9af733c0990aa64365f77e0ec86615759e7a3b7f7080f7aa95e1ee9e13978` |
| **檔案大小** | 35655 bytes |
| **行數** | 1846(header + 1845 rally) |
| **上傳時間** | 2026-06-02 21:13:06 |
| **隊伍代號** | TEAM_10808 |

### 為什麼 `submission_FINAL.csv` 跟 `submission_v14-aug-ovr_incl0.csv` 一樣?

在最終上傳前,我把 `submission_v14-aug-ovr_incl0.csv` 複製成 `submission_FINAL.csv` 一份,方便辨識「這是真的要上傳的」。所以兩個檔案內容**完全相同**(SHA256 驗證過)。

---

## 🏆 LB 結果

| LB | 分數 | 排名 |
|---|---|---|
| **Public** | 0.4341008 | 45 / 423 |
| **Private** | **0.3890227** | **5 / 423** ⭐ |

---

## 📂 對應的程式碼

| 用途 | 路徑 | 說明 |
|---|---|---|
| **產生 submission 的腳本** | `src/gen_submission_v14_old_in_training.py` | 從頭跑到尾的 .py 腳本(實際 production code)|
| **可讀的 Notebook** | `src/main.ipynb` | 同樣邏輯但分 cell + 註解,**推薦先看這個** |
| **設計脈絡** | `Claude-ask/brief_round*.md` | 11 份 brief 解釋每個決策為什麼這樣選 |
| **LB 演進記錄** | `LB_results.md` | 14 個版本各自的 public 分數 |

---

## 🏷️ Git 版本標記

```
Repo:    https://github.com/SugarSquirrel/aicup2026-table-tennis
Branch:  main
Tag:     v14-final-private-rank5  (← 對應這次提交)
         v14-priority3-final      (← 同一個 v14 的早期 tag)
Commit:  7af01f8 (LB_results.md 更新 + 結果記錄)
         b742f1d (v14 程式碼 + submission csv 真正進 git)
```

### Clone 後找最終版本的指令

```bash
git clone https://github.com/SugarSquirrel/aicup2026-table-tennis.git
cd aicup2026-table-tennis
git checkout v14-final-private-rank5   # 或直接看 main 的最新版

# 主要程式碼在這:
ls -la src/main.ipynb                                    # 推薦先看
ls -la src/gen_submission_v14_old_in_training.py        # production 腳本

# 提交檔案在這:
ls -la submission_FINAL.csv                              # 我們上傳的(根目錄)
ls -la submission/8/submission_v14-aug-ovr_incl0.csv    # 同一個檔(策展存檔,內容一致)
```

---

## 🧬 v14 的設計組成(累積所有合法路徑)

```
v14 = v7 base
    + v8 GRU 架構多樣性 (Time2Vec + FiLM)         ← v9 引入
    + 3-seed multi-seed averaging × 2 archs        ← v10 引入
    + Noise filter(1053 噪音 rally 排除)         ← v12 引入
    + ★ Priority 3:Old test 加進訓練集            ← v14 引入(關鍵)
    + Server override(public 加成,private 不變)
```

### 訓練資料規模

| 模型 | Train | + Old | = Total |
|---|---|---|---|
| LGBM all-prefix | 65,231(已濾噪)| +2,353 | 67,584 |
| TabPFN sampled | 13,942(已濾噪)| +838 | 14,780 |
| GRU all-prefix | 65,231(已濾噪)| +2,353 | 67,584 |

### 模型超參數(重要的部分)

```python
# LGBM
n_estimators=400, learning_rate=0.05, num_leaves=63
subsample=0.8, colsample_bytree=0.8
class_weight='balanced'(action/point)/ None(server)

# GRU
hidden=64, dropout=0.2, lr=1e-3, epochs=12, batch=256
seeds=[42, 1, 2]    # 每架構 3 個 seed 平均
GRU_BLEND=0.5       # v7 GRU 與 v8 GRU 的 blend ratio

# TabPFN
ManyClassClassifier(alphabet_size=10) for action 19-class
native TabPFN for point 10-class & server binary

# Ensemble weights(per-task,從 OOF 找出來)
WA = (LGBM 0.35, TabPFN 0.25, GRU 0.40)  β=0.000  ← action
WP = (LGBM 0.30, TabPFN 0.40, GRU 0.30)  β=0.125  ← point
WR = (LGBM 0.15, TabPFN 0.55, GRU 0.30)            ← server

# Noise filter
NOISE_THRESHOLD = 0.05   # 移除 prob_true_action < 5% 的 rally
```

---

## ⚙️ 重現步驟(組員想自己跑出同樣結果)

```bash
# 1. 環境
conda create -n aicup2026 python=3.11 -y
conda activate aicup2026
pip install pandas numpy scikit-learn lightgbm torch tabpfn tabpfn_extensions jupyter nbformat

# 2. TabPFN 需要 token(去 https://priorlabs.ai 註冊接受 license)
# 設定 TABPFN_TOKEN 環境變數,或建立 tabpfn_apikey.txt(已在 .gitignore)

# 3. 資料放好
# data/train.csv
# data/test_new.csv
# data/Reference_Only_Old_Test_Data/test.csv(從主辦取得)

# 4. 跑 OOF 快取(v7/v8,v14 會用到)
cd src
python gen_submission_v7_actionaug.py   # 產生 oof_v7_actionaug.npz
python gen_submission_v8_t2v_film.py    # 產生 oof_v8_t2v_film.npz

# 5. 跑 v14 最終版
python gen_submission_v14_old_in_training.py
# → 產生 submission_v14-aug-ovr_incl0.csv
# → 這個就等於 submission_FINAL.csv(SHA256 一致)
```

---

## 🔍 怎麼驗證「沒有作弊」

| 檢查項 | 怎麼驗證 |
|---|---|
| 沒有用測試集 actionId/pointId 答案 | 看 `gen_submission_v14_old_in_training.py`,沒有任何 test_new.csv 的 label 進入訓練流程 |
| ServerGetPoint override 是合法的 | README 寫明「參賽者可自行決定是否使用該資訊」(`data/Reference_Only_Old_Test_Data/README.txt`)|
| Old test 加進訓練是合法的 | 同上 README 條款,且 old 跟新測試集的 prefix 完全相同(沒有偷看 next-shot 答案)|
| Fold-safe CV 設計正確 | 看 `gen_submission_v7_actionaug.py` 中的 assert(`set(old.match).isdisjoint(set(tr.match))` 等)|

---

## 📊 LB 結果完整對照

| 上傳檔案 | 上傳時間 | Public LB | Private LB |
|---|---|---|---|
| v1-base submission | 2026-05-29 00:02 | 0.3188 | — |
| v2-player | 2026-05-29 22:18 | 0.3511 | — |
| v3-matchup | 2026-05-30 01:30 | 0.3462 | — |
| v7-aug-ovr | 2026-05-30 22:15 | 0.4112 | — |
| v7-aug(clean) | 2026-05-31 15:11 | 0.3443 | — |
| v10-aug-ovr | (中間版本) | 0.4179 | — |
| **★ submission_FINAL.csv(= v14-aug-ovr)** | **2026-06-02 21:13:06** | **0.4341** | **0.3890** |

---

## ❓ 組員可能會問的問題

**Q: 為什麼最後選 v14 而不是 v10?**
A: v14 比 v10 多了「Noise filter」+「Priority 3(old 進訓練)」。在私人榜上 v14 預測比 v10 多了 ~+0.05 的提升(rescued 子集得益最大)。

**Q: Override 算合法嗎?**
A: 算。`data/Reference_Only_Old_Test_Data/README.txt` 明文寫「參賽者可自行決定是否使用該資訊」。且 override 只影響 public(private 24 場新比賽完全沒被碰)。

**Q: 為什麼提交檔在 `incl0`(允許預測 actionId=0/pointId=0),不是 `excl0`?**
A: 早期校準時 `incl0` 比 `excl0` 公開分高(0.351 vs 略低),所以後續所有版本都用 `incl0`。

**Q: 14 個版本怎麼跑出來的?**
A: 看 `LB_results.md` + `Claude-ask/` 的 11 份 brief。每次迭代有 ChatGPT + Gemini 共識討論。

---

## 📌 一句話總結

**`submission_FINAL.csv` = `submission_v14-aug-ovr_incl0.csv`**
**= 2026-06-02 21:13:06 上傳的最終版本**
**= Public 0.4341 / Private 0.3890(rank 5/423)**
**= 由 `src/gen_submission_v14_old_in_training.py` 產生**
**= 設計細節可參考 `src/main.ipynb`**
