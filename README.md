# AI CUP 2026 — 基於時序資料之桌球戰術與結果預測

> 🏆 **Private Leaderboard 第 5 名 / 423 隊**(0.3890227)
> 📊 Public Leaderboard 第 45 名(0.4341008)
> 🎯 14 次提交達成此結果(前 5 名中提交次數最少之一)

每個 rally 用前 n−1 拍預測第 n 拍的 `actionId`(球種,19 類)、`pointId`(落點,10 類)與 `serverGetPoint`(該回合發球方是否得分)。
評分公式: **`Final = 0.4·F1_action(macro) + 0.4·F1_point(macro) + 0.2·AUC`**

---

## 📌 快速導航

| 想看什麼 | 看哪個檔 |
|---|---|
| **我們最終提交的是哪個版本?** | [`FINAL_SUBMISSION.md`](./FINAL_SUBMISSION.md) |
| **最終可執行的 Notebook(推薦先看)** | [`src/main.ipynb`](./src/main.ipynb) |
| **每個版本的 public LB 對照** | [`LB_results.md`](./LB_results.md) |
| **每個迭代的策略討論(11 份 brief)** | [`Claude-ask/`](./Claude-ask/) |
| **ChatGPT / Gemini 共識回覆(6 份)** | [`ChatGPT-reply/`](./ChatGPT-reply/) |
| **歷次提交的 csv 存檔** | [`submission/`](./submission/) |

---

## 🏆 最終結果

| LB | 分數 | 排名 | 上傳時間 |
|---|---|---|---|
| **Public** | 0.4341008 | 45 / 423 | 2026-06-02 21:13:06 |
| **Private** | **0.3890227** | **5 / 423** ★ | 同上(私人 6/3 揭曉)|

**排名跳躍 +40**(public 45 → private 5):驗證「robust generalization 勝 LB tuning」的策略選擇。

---

## 🧬 最終方法(v14)架構

```
                   train.csv + test_new.csv + Old test (Reference Only)
                              ↓
                  [v12 Noise Filter] 移除 1053 個 prob_true_action<5% 的噪音 rally
                              ↓
              [Fold-safe 統計擴充] train+old 重做 transition / player_dists /
                                  matchup cluster(只用於 action;point/server 用 train-only)
                              ↓
              [★ Priority 3] 把 Old test 1236 rally 加進 LGBM/TabPFN/GRU 訓練集
                              ↓
              ┌──────────────┬─────────────────┬──────────────────────────┐
              │   LGBM       │   TabPFN        │       GRU                │
              │              │                 │  ┌──────────┬──────────┐ │
              │              │   ManyClass     │  │ v7       │ v8       │ │
              │              │   (action only) │  │ Linear   │ Time2Vec │ │
              │              │                 │  │ baseline │ + FiLM   │ │
              │              │                 │  └──── 3 seeds × 2 ────┘ │
              │              │                 │      blend at α=0.5      │
              └──────────────┴─────────────────┴──────────────────────────┘
                              ↓
              [Per-task weighted ensemble + prior-correction β]
                              ↓
              [Server Override] 用 Old test 的真值覆蓋 1236 公開 rally
                                (README 明文允許;private 24 場無影響)
                              ↓
                       submission_FINAL.csv
```

---

## 🎯 5 個核心設計決策

### 1. Task-Specific Feature Gating(v7 引入)
不同 task 用不同的統計來源:
- **Action** 用 `train + old` augmented features(選手球風跨場次可轉移)
- **Point/Server** 用 `train-only` features(point 太依賴對手/局勢,aug 反而傷)

### 2. Multi-Architecture Diversity(v9 引入)
- **v7 GRU**:linear num projection,簡單基準
- **v8 GRU**:Time2Vec(時序周期嵌入)+ FiLM(條件化最終 hidden)
- 兩架構**故意設計成不同的歸納偏置**,blend at α=0.5
- 結果:rescued action F1 +0.043(從 0.352 跳到 0.395)

### 3. Multi-Seed Averaging(v10 引入)
- 每架構 3 個 seed × 2 架構 = 6 個 GRU 模型平均
- Variance reduction → public LB +0.0066 實測

### 4. Noise Filter(v12 引入)
- 用 v10 ensemble OOF 識別 `prob_true_action < 5%` 的 rally
- 移除 1053/14995 個訓練樣本(7%)
- Per-stratum 分析:rescued noise 率最低(4%)→ filter 對 private 安全

### 5. ★ Priority 3:Old test 加進訓練集(v14 引入,最大槓桿)
- LGBM all-prefix +2353 樣本、TabPFN sampled +838 樣本、GRU all-prefix +2353 樣本
- 16% 私人選手只在 old 出現過 → 給他們真實 labeled 訓練樣本
- **這是 private +0.05 的主要貢獻**

---

## 📊 版本演進(完整 14 個版本)

| Tag | 版本 | Public LB | 設計新增 | 提交檔 |
|---|---|---|---|---|
| `v1-base-lb0.319` | v1 | 0.3188 | LGBM+TabPFN+GRU 基礎 | `submission/1/` |
| `v2-player-cv0.341` | v2 | **0.3511** | + 選手球種傾向(player-marginal)| `submission/2/` |
| `v3-matchup-cvb0.350` | v3 | 0.3462 | + matchup KMeans cluster | `submission/3/` |
| — | v4 | 未上傳 | + Old test player-prior 擴充 | — |
| — | v5 | 未上傳 | + matchup/transition 用 Old 擴充 | — |
| — | v6 | 未上傳 | + Fully-augmented OOF | — |
| `v7-actionaug-cvb0.347` | v7 | 0.4112 (ovr) | ★ Task-specific feature gating + server override | `submission/4/` |
| — | v7 clean | 0.3443 | 同上但無 override(乾淨版)| `submission/5/` |
| — | v8 | 未上傳 | Time2Vec + FiLM GRU(alone 失敗) | — |
| — | v9 | 未上傳 | v7+v8 GRU diversity blend | — |
| `v10-final-private-est-0.334` | v10 | 0.4179 | + Multi-seed averaging | — |
| — | v11 | 未上傳 | + BiGRU + LSTM(差異邊際,放棄)| — |
| — | v12 | 未上傳 | + Noise filter | — |
| — | v13 | n/a | (跳過) | — |
| **`v14-final-private-rank5`** | **v14 ★** | **0.4341 / private 0.3890** | + Priority 3(Old → 訓練集)| **`submission_FINAL.csv`** |

詳細迭代分析見 [`LB_results.md`](./LB_results.md)。

---

## 🤖 全程 3-AI 協作流程

| 角色 | 任務 |
|---|---|
| **Claude (主)** | 主 coder + CV validator + 撰寫 brief |
| **ChatGPT** | 策略 reviewer + 共識整合 |
| **Gemini** | 異議 + 補強 + 機制深挖 |

### 共識輪次

11 份 brief(`Claude-ask/`)+ 6 份共識回覆(`ChatGPT-reply/`),共 9 輪結構化討論。
關鍵轉折:
- Round 5:從「ceiling 0.35 停止」翻轉到「發現 leak 結構,可繼續」
- Round 8:Per-stratum 證據驅動 task-specific gating
- Round 11:v8 alone 失敗 → diversity blend 化敵為友

詳細見 [`Claude-ask/`](./Claude-ask/) 和 [`ChatGPT-reply/`](./ChatGPT-reply/)。

---

## ⚙️ 重現環境

```bash
# 1. 建立環境(Python 3.11)並安裝依賴
conda create -n aicup2026 python=3.11 -y
conda activate aicup2026
pip install -r requirements.txt
#   ↑ requirements.txt 已固定版本,並含 torch cu121 的 --extra-index-url。
#     CPU-only 或不同 CUDA 版本者,請改裝對應的 torch(本專案神經網路需 GPU)。

# 2. TabPFN token
#   去 https://priorlabs.ai 註冊接受 license,設定環境變數 TABPFN_TOKEN
#   (或建立 tabpfn_apikey.txt,已在 .gitignore,絕對不要 commit)
#   權重於首次執行時下載並快取,之後可離線推論。

# 3. 資料放好(從 AIdea 下載,勿公開重新散布)
#   data/train.csv
#   data/test_new.csv
#   data/Reference_Only_Old_Test_Data/test.csv
#   data/Reference_Only_Old_Test_Data/README.txt
```

## 🚀 重現最終結果

```bash
cd src

# Step 1:跑出 v7 / v8 OOF 快取(v14 會用到)
python gen_submission_v7_actionaug.py    # 產生 oof_v7_actionaug.npz
python gen_submission_v8_t2v_film.py     # 產生 oof_v8_t2v_film.npz

# Step 2:跑最終 v14 版本
python gen_submission_v14_old_in_training.py
# → 產生 submission_v14-aug-ovr_incl0.csv(等於 submission_FINAL.csv,SHA256 一致)

# 或用 Notebook 互動式跑
jupyter notebook main.ipynb
```

---

## 📁 檔案結構

```
.
├── README.md                              ← 你正在看
├── requirements.txt                       ← Python 依賴(pip install -r)
├── FINAL_SUBMISSION.md                    ← 組員想看「最終版本是哪個」必看
├── LB_results.md                          ← 每個版本的 public/private LB 對照
├── data/                                  ← 從 AIdea 下載放這(已 gitignore)
│   ├── train.csv
│   ├── test_new.csv
│   └── Reference_Only_Old_Test_Data/
├── src/
│   ├── main.ipynb                         ← ★ v14 canonical notebook(推薦先看)
│   ├── gen_submission_v14_old_in_training.py  ← 最終版 production script
│   ├── gen_submission_v{1..13}_*.py       ← 14 個歷史版本
│   └── *.npz                              ← OOF 快取(已 gitignore)
├── Claude-ask/                            ← 11 份策略 brief
├── ChatGPT-reply/                         ← 6 份共識回覆
└── submission/                            ← 關鍵版本的 CSV 存檔(中間版見 src 重跑)
    ├── 1/  v1-base       ├── 5/  v7-aug clean
    ├── 2/  v2-player     ├── 6/  v10-aug-ovr
    ├── 3/  v3-matchup    ├── 7/  v12-aug-ovr
    ├── 4/  v7-aug-ovr    └── 8/  v14-aug-ovr（最終）
```

> 註:根目錄僅保留最終結果 `submission_FINAL.csv`(= `submission/8/` 的 v14);各版本中間 CSV 可由 `src/gen_submission_v*.py` 重新產生,不另外公開。

---

## 🔍 合法性聲明

| 檢查項 | 說明 |
|---|---|
| 沒有用 test_new.csv 的 actionId/pointId 答案 | `src/gen_submission_v14_old_in_training.py` 完全不引用 test_new 的 next-shot label |
| ServerGetPoint override 是合法的 | `data/Reference_Only_Old_Test_Data/README.txt` 寫明「參賽者可自行決定是否使用該資訊」|
| Old test 加進訓練是合法的 | 同上 README 條款;且 Old 跟新測試集 prefix 完全相同(沒有偷看 next-shot 答案)|
| Fold-safe CV 設計正確 | 程式碼有 `assert(set(old.match).isdisjoint(set(tr.match)))` 等防呆檢查 |

---

## 🙏 致謝

- **指導教授**:國立中正大學 資訊工程學系 江振國 教授
- **主辦單位**:教育部主辦、國立中央大學執行(競賽資料經 AIdea 平台釋出)
- **協作 AI**:Claude (Anthropic)、ChatGPT (OpenAI)、Gemini (Google)
- **隊伍代號**:TEAM_10808

---

## 一句話總結

```
14 個版本、9 輪 AI 共識、12 份策略 brief、Oracle 上限診斷、
Public/Private 結構推論、Per-stratum 分析、Task-specific gating、
Multi-arch diversity blend、Multi-seed averaging、Noise filter、Priority 3
                                  ↓
                  Private Rank 5/423 (0.389) 🏆
```
