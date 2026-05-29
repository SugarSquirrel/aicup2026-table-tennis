# AI CUP 2026 — 落點/動作預測模型 對話紀錄匯出

> 這份文件是與 Claude Code 討論「AI CUP 2026 桌球 rally 預測」題目的過程整理，供其他 AI 助手接續參考。
> 包含:baseline 做法分析、train/test 資料分析發現、模型設計策略建議,以及尚未回答的開放問題。

---

## 0. 題目背景

- 資料夾:`aicup2026/data/`,含 `train.csv`、`test_new.csv`、`sample_submission.csv`。
- 每筆資料是一個 `rally_uid`(一個來回),內含多次擊球,用 `strikeNumber` 排序。
- **提交格式(per rally)**:每個 `rally_uid` 預測三個目標:
  - `actionId`(下一拍動作,多分類)
  - `pointId`(下一拍落點,多分類)
  - `serverGetPoint`(整段 rally 發球方是否得分,二分類)
- **評分**:`Final = 0.4 × F1_action(macro) + 0.4 × F1_point(macro) + 0.2 × AUC`

### 欄位
```
train: rally_uid, sex, match, numberGame, rally_id, strikeNumber, scoreSelf, scoreOther,
       serverGetPoint, gamePlayerId, gamePlayerOtherId, strikeId, handId, strengthId,
       spinId, pointId, actionId, positionId
test : 同上,但「沒有 serverGetPoint」(它是要預測的目標之一)
```

---

## 1. Baseline code 做法分析(`src/baseline code.py`)

> 註:使用者**不打算用 baseline 的方法**,只是參考它的策略方向。

一個多任務 LSTM,同時做三件事:

**資料處理**
- 按 `rally_uid` + `strikeNumber` 排序;`strikeNumber` clip 到 0~40。
- 11 個特徵全當類別(`pd.Categorical` → 整數 code,`+1` 把 0 留給 padding)。類別字典只用 train 建,test 新類別會變 0(等同 padding)。
- **序列樣本建構(關鍵)**:用前 n-1 拍預測後 n 拍(next-step prediction):
  - 輸入 X = 第 1~n-1 拍
  - 標籤 yA/yP = 第 2~n 拍的 action/point(往後錯一位)
  - yR = 整段 rally 的 serverGetPoint
- padding 補到最長序列,標籤 padding 用 -1(loss `ignore_index=-1`)。

**模型架構**
```
11 個特徵 → 各自 Embedding(16維) → concat(176維)
          → LSTM(hidden=128, 單向, 1層)
          → 三個 head:
             ├ act_head:逐時間步預測 action
             ├ pt_head :逐時間步預測 point
             └ rly_head:對時間做 masked mean pooling → 預測 serverGetPoint
```

**訓練細節**
- 類別不平衡:用出現次數倒數當 CrossEntropy 權重。
- 複合 loss:`0.4×CE_action + 0.4×CE_point + 0.2×BCE_rally`(對應評分權重)。
- 梯度裁切 1.0,Adam,預設只跑 3 epoch,10% 驗證(隨機切 + serverGetPoint stratify)。

**推論**:整段 test rally 餵入,取「最後一個有效時間步」輸出當「下一拍」預測;serverGetPoint 取 sigmoid。

**baseline 可改進處**:只跑 3 epoch、單向單層 LSTM(易 underfit);rally head 用 mean-pooling 可換;數值特徵(score、strikeNumber)被當 category;沒處理 test 新類別。

---

## 2. train / test 資料分析(重點發現)

### 2.1 規模與重疊
| 項目 | train | test |
|------|-------|------|
| rally 數 | 14,995 | 1,845 |
| 列數 | 84,707 | 5,668 |
| rally 長度(中位數) | 5 | **2** |
| rally 長度(平均) | 5.65 | 3.07 |
| match 重疊 | — | **0(完全不同場次)** |
| 選手重疊 | — | 40/71(**31 位沒看過,佔 test 30.8% 的列**) |

### 2.2 核心發現:分布差異主因是「test 截斷了 rally」,不是選手不同
- `pointId=0` 在「**每個 rally 的最後一拍**」出現率 = **100%**,非最後一拍只有 0.4%。
- → `pointId=0` 代表該 rally 的**終結拍**(致勝/失誤球,沒有下一個落點)。
- train 是完整 rally(每段都含終結拍),**test 是被截斷的部分 rally**,要預測「下一拍」。
- 證據:把 train 的終結拍移除、對齊 test 設定後,分布差異大幅收斂:
```
pointId   TVD: 0.188 → 0.059   (絕大部分是終結拍造成的假性差異)
spinId    TVD: 0.106 → 0.054
positionId TVD:0.226 → 0.188   (殘留的才是真實 match/選手差異)
strikeId  TVD: 0.207 → 0.158
actionId  TVD: 0.196 → 0.165
```
→ **分布差異有兩個來源:(1) 截斷假象(可靠訓練設定完全消除)、(2) 真實跨場次/跨選手 shift(較難)。**

### 2.3 關於 player id —— 結論:別用原始 ID
- 31/71 test 選手在 train 沒出現(佔 test 30.8% 列)→ 原始 ID embedding 對這些人是未訓練 OOV。
- match 完全不重疊 → 看過的選手也在全新脈絡裡。
- 用原始 ID 會 overfit train 特定選手,無法泛化。
- **替代方案**:rally 內選手**嚴格交替出拍**(alternation=1.0,每段剛好 2 人)。用「**發球方/接球方角色**」或「**`strikeNumber` 奇偶 parity**」表示「是誰在打」,選手無關、可跨場次泛化。

### 2.4 serverGetPoint 的關鍵發現(佔分數 20%)
```
rally 總長度為偶數 → serverGetPoint=1 機率 0.999
rally 總長度為奇數 → serverGetPoint=1 機率 0.001
```
- 因選手嚴格交替,勝負幾乎完全由「最後一拍由誰打(= 最終長度奇偶)」決定,與誰較強無關。
- **陷阱**:test 是截斷的,看不到最終長度 → 不能直接套這條規則。
- 本質:從半場 rally 預測「還會打幾拍 / 誰打到最後一拍」。已觀測長度與結果相關性僅 -0.06,故此項從截斷資料預測真的難,base rate 0.55。

### 2.5 可預測性(Markov baseline,train 上、排除終結拍目標)
- next `pointId`:全域眾數 acc 29.5%;條件 (當前落點,當前動作) acc 31.6% → **落點偏隨機,條件化只小幅提升**。
- next `actionId`:條件 acc 44.3% → 比落點好預測。
- next `pointId` 類別不平衡(非終結目標):類 9=29.5%、類 8=22.7%、類 3=0.4% → macro-F1 下罕見類很關鍵。

---

## 3. 模型設計策略建議(依重要性排序)

**1. 訓練設定必須複製 test 的截斷(影響最大)**
- 不要把「整段 rally → 每一步 next」全當訓練樣本(會讓模型狂預測終結拍 `pointId=0`,但 test 幾乎沒有 0)。
- 做法:對每個 train rally,在**非終結位置**隨機/全前綴截斷,用「前綴 → 下一拍」當樣本,並讓**截斷長度分布貼近 test**(test 中位數=2,偏短)。預測目標**排除終結拍**。

**2. 驗證集按 match 切,不要隨機切**
- baseline 隨機切會高估分數。真實 shift 是跨場次 → 用 **GroupKFold by `match`**(順便讓部分驗證選手不在訓練裡),才能誠實估計泛化。

**3. 特徵**
- **丟掉** `gamePlayerId` / `gamePlayerOtherId`;改用 parity / 角色。
- **保留** `sex`(男女打法不同,train/test 分布一致 TVD 0.06)。
- `scoreSelf`/`scoreOther`/`strikeNumber` 當**數值**(別全當 category),可加「比分差」「是否賽末點」。
- 重點放在**最後幾拍的序列動態**(上一拍落點/動作/旋球 → 下一拍),這才是跨選手泛化訊號。

**4. 針對 macro-F1 的類別不平衡**
- 指標是 macro-F1 → 預測眾數會慘。用 class weight / focal loss,注意罕見落點 recall。

**5. 架構**
- 序列模型(LSTM/GRU/小型 Transformer)方向 OK,但**讀出層用「最後一個有效時間步」**(對應下一拍),不要對 serverGetPoint 用全序列 mean-pooling(test 序列短,結果取決於尾端)。
- 三個 head 可共享 encoder,serverGetPoint head 建議額外餵入 parity / 已打拍數等「結束時機」線索。

---

## 4. 使用者最後提出、尚待回答的開放問題

1. **「訓練設定必須複製 test 的截斷」具體是什麼意思?**
   (預期回答方向:把 train 的完整 rally 切成多個「部分 rally 前綴 → 下一拍」樣本,模擬 test 只給半場、要預測下一拍的情境;且截斷點分布要貼近 test 的偏短長度;訓練目標排除 `pointId=0` 終結拍。)

2. **時序(sequence)到底重不重要?**
   (待分析:Markov 條件化只小幅優於全域眾數,test 序列又很短(中位數 2),需評估「長序列建模 vs 只看最後 1~2 拍 + tabular 特徵」的取捨。)

3. **應該選什麼樣的 model 比較好?**
   (待討論:序列模型(LSTM/GRU/Transformer)vs 樹模型(LightGBM/XGBoost)on 最後幾拍特徵;或兩者 ensemble。考量 test 序列極短、跨場次泛化、macro-F1。)

---

## 5. 給接手 AI 的提示
- 使用者明確**不採用 baseline 寫法**,要的是模型設計**方向**。
- 最重要的洞察:**test 是截斷 rally → 預測「下一拍」**,且分布差異主要來自截斷(可控)+ 真實跨場次 shift(較難)。
- 使用者偏好**不要用 player id**(分析支持此決定)。
- 環境:Windows 11 / PowerShell;Python 有 pandas/numpy/sklearn;資料在 `data/`。
