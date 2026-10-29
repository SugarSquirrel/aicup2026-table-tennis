# paper_analysis — 論文用分析腳本

IEEE BigData 2026 AI-Sports workshop 論文裡,賽後分析的數字都由這裡的腳本產生。
論文 LaTeX 本身在 `paper/`(Overleaf git clone,不進本 repo)。

## 前置

- 根目錄放官方賽後釋出的 `test_answer_solution.csv`(已 gitignore,**勿公開重新散布**)
- `data/train.csv`、`data/test_new.csv`、`data/Reference_Only_Old_Test_Data/test.csv`(已 gitignore)
- `submission/1..8/` 的歷次提交 CSV(已在 repo 內)
- Python 套件:`pandas`、`numpy`、`scikit-learn`

## 腳本

| 腳本 | 產出 | 對應論文 |
|---|---|---|
| `score_submissions.py` | 每個版本的 public / private 分數與三項子分數 → `results/scores.csv` | Table IV、Sec. V-C(private AUC、終結拍比例)|
| `bootstrap_private.py` | 版本間 private 分數差的 paired bootstrap 95% CI → `results/bootstrap_private.csv` | Table V |
| `data_stats.py` | 資料集統計、public/private 切分驗證、選手分層比例、截斷分析(前綴長度、終結拍、奇偶規則)| Table I、Table II、Sec. III、Sec. V-C |
| `private_strata.py` | 各版本在 private 上的 seen/rescued/cold 分層 F1 + 分層內 bootstrap → `results/private_strata*.csv` | Table VI、Sec. V-D、VI-B/C |
| `pair_bootstrap_private.py` | 以選手配對為群集的 paired bootstrap(59 組配對;含 v14→v14g 與分層)→ `results/pair_bootstrap_private.csv`、`results/pair_bootstrap_strata.csv` | Table V caption、Sec. V-C/V-D、VI、VII(v2)|
| `private_by_idtype.py` | 各版本 private 分數與三項子分數,拆成 pooled IDs(340–354)與新比賽(355–363)→ `results/private_by_idtype.csv` | Sec. V-D(v2 熟悉配對/新比賽分組)|
| `check_split_ids.py` | private ID 結構檢查(rally 編號=比分和+1、配對是否在 reference、比分進程一致性)、Table I 兩列、private 選手、資料瑕疵 → `results/split_checks.csv` | Table I、Sec. III-C、Sec. VII(v2;即 TODO 中的 check_a2_ids.py)|
| `cvb_weight_and_val_strata.py` | CV-B 權重 0.94 的來源、test 下一拍打者 seen 比例、重建 v7 驗證分層並計算 rescued 樣本的配對是否在 reference → `results/cvb_weight_and_val_strata.csv` | Sec. III-D、V-D、VII(v2)|

```bash
python paper_analysis/score_submissions.py
python paper_analysis/bootstrap_private.py
python paper_analysis/data_stats.py
python paper_analysis/private_strata.py
```

**驗證**:`score_submissions.py` 對 `submission/8`(最終版)應算出 public 0.434101 / private 0.389023,
與官方公布的 0.4341008 / 0.3890227 一致,代表評分公式(macro-F1 只平均真值出現的類別、含 class 0)正確。

## 賽後重跑實驗(需要 GPU)

| 步驟 | 腳本(`src/`) | 產出 | 用途 |
|---|---|---|---|
| v7 | `gen_submission_v7_actionaug.py` | `src/oof_v7_actionaug.npz`、根目錄 `submission_v7-*.csv` | OOF 快取 + 重現度檢查 |
| v8 | `gen_submission_v8_t2v_film.py` | `src/oof_v8_t2v_film.npz`、`submission_v8-*.csv` | OOF 快取 |
| v14 | `gen_submission_v14_old_in_training.py` | `submission_v14-*.csv` | 同機重現最終版(v14g 的比較基準)|
| v14g | `gen_submission_v14g_gated_old_training.py` | `submission_v14g-*.csv` | old 樣本只給 action 模型(Sec. VI-C 的直接檢驗)|
| oof | `oof_ablation_v10_v14.py` | `results/oof_ablation.csv` | Table III (e)–(h) 列 |

`run_experiments.sh` 依序執行,log 寫到 `logs/`(已 gitignore),失敗即停;可只跑部分步驟,例如 `bash paper_analysis/run_experiments.sh v14g oof`。
跑完後用 `score_reruns.py` 評估重現度與 v14g(→ `results/reruns_*.csv`)。

需要 TabPFN token:放在根目錄 `tabpfn_apikey.txt`(已 gitignore)或設環境變數 `TABPFN_TOKEN`,
且帳號需在 <https://ux.priorlabs.ai> 的 Licenses 分頁同意 TabPFN-3 授權。
TabPFN-3 在 1.2 萬筆訓練資料上很吃記憶體,12 GB 顯存的卡會溢到系統 RAM;建議 24 GB 顯存(論文原始結果在 RTX 4090 上產生)。

### 在新機器上從零開始(uv)

```bash
git clone https://github.com/SugarSquirrel/aicup2026-table-tennis.git
cd aicup2026-table-tennis
uv venv .venv --python 3.11
UV_HTTP_TIMEOUT=300 uv pip install --python .venv -r requirements.txt --index-strategy unsafe-best-match
# 放入 data/、test_answer_solution.csv、tabpfn_apikey.txt 後:
bash paper_analysis/run_experiments.sh
.venv/Scripts/python.exe paper_analysis/score_reruns.py   # Linux: .venv/bin/python
```

## 注意

- 正解只用於**賽後評估**,沒有任何模型、特徵或超參數是用它挑選的(論文 Sec. V-A 有聲明)。
- private 只有 609 筆,單一步驟的差異 CI 約 ±0.02–0.03,只有 v1→v7 的累積進步顯著。
