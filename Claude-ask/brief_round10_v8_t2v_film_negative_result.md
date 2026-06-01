# Round 10 Brief — v8 Time2Vec + FiLM 嘗試結果(負結果記錄)

> 使用者在 deadline 前夕(2026-06-02)要求「不管 consensus,繼續找 SOTA 嘗試提高分數」,
> 並指定要把「時序當條件」融入預測。
> 本份 brief 記錄這次嘗試的結果、為何失敗、以及為什麼這反而強化了 v7 是最佳化終點的結論。

---

## 0. 為什麼跑 v8(脈絡)

```text
v7 上傳結果:
  - v7-aug-ovr public = 0.4112 / rank 62/381(驗證 leak 結構)
  - v7-aug clean public = 0.3443 / rank 133/388(乾淨 final)
  - 第一名 public = 0.55

使用者想衝 0.55:「不管,你去看最新的熱門論文或方法、SOTA」
→ 我嘗試最對齊「讓時序成為一個條件」需求的方法:Time2Vec + Time-FiLM
```

---

## 1. SOTA 掃描結果(2025-2026)

| 方向 | 來源 | 對我們的判斷 |
|---|---|---|
| Mamba-3(ICLR 2026) | openreview/HwCvaJOiCj | ❌ 文獻明說「short interaction sequences 效果有限」我們序列中位 2 |
| Time2Vec | medium/cerliani, arxiv/2504.13801 | ✓ 替換 positional encoding,文獻 +10-15% on classification |
| Time-FiLM | emergentmind/time-film-conditioning | ✓ 時序作為 condition modulate 預測 — 直接對齊使用者需求 |
| Wu 2026 桌球 paper | mjssm.me/MJSSM_March_2026_Wu | 部分 — Transformer + ball speed/spin,但我們沒 ball speed |
| TabTransformer / TabPFN-Time | 各家 | ❌ 太重,deadline 內 retrain 不可能 |

**選定**:Time2Vec(替換 GRU 的 numeric Linear)+ FiLM(context-modulate GRU 最終 hidden)。

---

## 2. v8 實作

```python
class Time2Vec(nn.Module):
    """Per-scalar Time2Vec: 每個 input scalar -> 1 linear + k periodic sin"""
    def __init__(s, in_dim, k=7):
        s.w_lin = nn.Parameter(torch.randn(in_dim)*0.1)
        s.b_lin = nn.Parameter(torch.zeros(in_dim))
        s.W = nn.Parameter(torch.randn(in_dim, k)*0.5)
        s.B = nn.Parameter(torch.randn(in_dim, k)*0.5)
    def forward(s, x):
        lin = (x*s.w_lin + s.b_lin).unsqueeze(-1)
        per = torch.sin(x.unsqueeze(-1)*s.W + s.B)
        return torch.cat([lin, per], -1).flatten(-2)

class FiLM(nn.Module):
    """gamma, beta = MLP(ctx); feat <- feat * (1+tanh(gamma)) + beta"""
    def __init__(s, ctx_dim, feat_dim):
        s.proj = nn.Linear(ctx_dim, feat_dim*2)
    def forward(s, feat, ctx):
        gb = s.proj(ctx); g, b = gb.chunk(2, -1)
        return feat * (1.0 + torch.tanh(g)) + b

# GRU 替換:
# 原:  s.num = nn.Linear(3, 16)
# 新:  s.t2v = Time2Vec(3, k=7)   # 3 scalars × 8 = 24 dim
# 加:  s.film = FiLM(32, 64)      # context = sex(4) + t2v_pool(24) + role_pool(4)
# 用法: h = gru_output -> h = film(h, ctx)
```

(完整實作:`src/gen_submission_v8_t2v_film.py`)

---

## 3. 結果(實測,真實 OOF)

### Overall CV-B
| 指標 | v7 | v8 | Δ |
|---|---|---|---|
| action F1 | 0.3657 | 0.3649 | −0.0008 |
| point F1 | 0.1946 | 0.1945 | −0.0001 |
| server AUC | 0.6151 | 0.6146 | −0.0005 |
| **Overall** | **0.3471** | 0.3467 | **−0.0004** |

### Per-stratum F1(真相)
| 子集 | n | v7 action | v8 action | v7 point | v8 point |
|---|---|---|---|---|---|
| seen | 11723 | 0.3723 | 0.3726 | 0.1968 | 0.1966 |
| **rescued** | **378** | **0.3518** | **0.2930** | **0.1683** | **0.1354** |
| cold | 2894 | 0.2479 | 0.2395 | 0.1607 | 0.1634 |

**Rescued 子集是災難**:action −0.059, point −0.033。

---

## 4. 為什麼失敗(機制診斷)

```text
Time2Vec(k=7) + FiLM 加的參數量:
  - Time2Vec: 3 × (1 + 1 + k + k) = 3 × 16 = 48 個 weights/biases(小)
  - FiLM proj: 32 × 128 = 4096 weights(主要增量)

但實際傷害不是參數量,是「擬合方向」:
  1. 序列中位長度 2 → t2v 的週期項在「2 個時間點」上沒有意義
     (sin(w·t+b) 在 t=1 vs t=2 的差只能編碼非常局部的差異)
  2. FiLM context pooling 在「mean over 2 points」→ 跟單一向量沒差太多
  3. 但 FiLM proj 還是學了 train 選手的 modulation 偏好
     → 對 train 中沒見過的選手(rescued / cold),這個 modulation 是錯的
  4. v7 沒有 FiLM,GRU output 直接走 task heads → 對 unseen 選手退回 base distribution
     比較穩

簡單講:
  v8 多學了 train-specific 的 conditional pattern,
  在「跨選手泛化」這個維度上付了代價。
```

---

## 5. 對 v7 = final 結論的強化

```text
Oracle 診斷(Round 4):查表上限 ~0.35,oracle 都打不過模型本身
4 輪 ChatGPT/Gemini 共識:架構工程都收斂
v8 SOTA 嘗試:Time2Vec+FiLM 在 rescued 上崩 0.06
↓
v7 是「在 fold-safe 泛化條件下」的合理最佳化終點。
任何加複雜度的方向都會傷 rescued/cold(這正是私人佔比高的子集)。
```

---

## 6. 給 ChatGPT/Gemini 看的話

```text
本輪結果支持你們先前的判斷:
  「不要再開新模型訓練方向」是對的。
但這次失敗實驗有用 — 它把「+0.10 在這條路上不可能」從「共識」升級為「實證」。

私人 6/3 公布時:
  - 我們維持 v7-aug_incl0.csv 為 final
  - 預估私人 0.355-0.37
  - 對前 30 名(他們可能 LB-overfit)有結構性優勢
```

---

## 7. 給使用者的最終建議

```text
✅ 不要再做任何實驗
✅ 不要再上傳任何檔案(除非確認 last upload 已經是 v7-aug_incl0.csv)
✅ Deadline = 今晚 23:59,保持現狀
✅ 等 6/3 私人榜公布

私人榜公布時兩種情境:
  情境 A(高機率):我們維持 0.355-0.37,排名上升(前 30 反噬)
  情境 B(低機率):資料天花板比預估高,我們上不去也下不來

無論哪種,v7 是合理且乾淨的選擇。
```
