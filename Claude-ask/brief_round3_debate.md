# Round-3 辯論題（請 ChatGPT 與 Gemini 互相討論後，回一份共識）

> 我是 Claude（主程式 + CV 驗證）。我已把你們 round-2 的每個點子都實測了。結論很硬，所以這次想請你們**兩位互相辯論**，回一份**共識**給我。
> 量尺說明：CV-A=match-held-out（private/cold-start proxy）；**CV-B=seen-player weighted 0.94/0.06（公開榜 proxy，已驗證：v2 CV-B 0.348 ≈ 實際公開榜 0.351）**；CV-C=unseen 子集（private 風險）。

## A. Round-2 點子實測結果（全部用 CV-B 判定）
| 點子 | 提議者 | 結果 | 判定 |
|---|---|---|---|
| **matchup 風格分群** P(next_action\|next_hitter, opp_cluster) | Gemini | CV-B **+0.0021**、cold-start 安全 | ✅ 唯一乾淨贏家(已做成 v3) |
| **per-class threshold/bias 最佳化**(macro-F1) | 兩位 | CV-A +0.007 但 **CV-B +0.000** | ❌ 純 CV-A 過擬合、不上榜 |
| **score 視角修正 + pressure phase** | ChatGPT | fixed/phase 比 flip **更差**(CV-B 0.330→0.328) | ❌ 模型用 parity 已能還原視角 |
| **Inter-Rally 時間線重構**(跨回合記憶/反推) | Gemini | test 每局只有 **2.4% 連續、1% 有下一回合** | ❌ test 太稀疏、抓不到跨回合；且反推 serverGetPoint=反推被移除標籤(規則風險) |
| raw player ID / GRU player-emb / player×state / in-match / server 選手勝率 | 兩位/前輪 | CV-B ≤ +0.001 或變差、多傷 cold-start | ❌ player 槓桿到頂 |

## B. 殘酷現實
- 我們穩在 **CV-B ≈ 0.350、公開榜 ≈ 0.351–0.353（rank ~113/371）**。
- 你們 round-2 的「大槓桿」候選（per-class「無痛 +0.02~0.03」、inter-rally「核彈」、score-fix）**經 CV-B 驗證全部不成立**——其中 per-class 是典型「CV-A 漲、公開榜不漲」的過擬合陷阱，幸好 CV-B 抓到。
- 目標：里程碑 **0.40**（+0.05）、最終 **0.45**。**+0.05 在我們試過的所有方向裡都看不到。**

## C. 請你們辯論並達成共識的問題
1. **誠實天花板假設**：在「不對公開榜過擬合（即 CV-B/CV-C 都不崩）」的前提下，這份資料的真實天花板是不是就在 ~0.35？前 30(0.44+) 是否大概率是**對公開榜過擬合**（48–71 次上傳）、private(6/3) 會洗牌？
   - 若你們認為「是」，請一起確認，我們就轉向 **private 穩健性 + 提交策略 + 報告**。
   - 若你們認為「否」，請見問題 2。
2. **若你們堅持 0.40+ 可達**：請給**恰好一個**「機制上全新、我們 A 表沒試過」的方向，且必須說明：
   - 它為何能在 **CV-B 上漲**（不是只在 CV-A）；
   - 它為何**不傷 CV-C**（不靠 public 過擬合）；
   - 具體可實作的步驟（我會用 CV-A/B/C 驗證）。
3. 特別針對 **point macro-F1=0.19**（最大短板、權重 0.4、是 +0.05 的關鍵）：在「不重構 label、不靠 inter-rally、不靠已否決的幾何/two-stage」前提下，還有**任何**能在 CV-B 上漲的 contextual 訊號嗎？若沒有，請直說 point 已到本質上限。

## D. 我還沒測、但預期邊際的（你們可在共識裡建議要不要做）
- action 階層模型(super-class→fine, ensemble flat)
- termination hazard auxiliary(預測剩餘拍數/長度奇偶 → 餵 point0/server)
- matchup 改 **soft-cluster/PCA**(Gemini 建議, 取代 hard cluster, 可能比 +0.002 多一點)
- geometric transition distance(point, 10x10 距離表 × strength/action)

## D2. 架構層級的反思（重要，請特別評估）
我上網確認：本題 = 學界「**stroke forecasting**（預測下一拍球種+落點）」問題，有領域 SOTA：
- **ShuttleNet**(Wang 2022, arXiv 2112.01044)：Transformer encoder-decoder，**分別建模兩位選手風格 + rally-progress position-aware fusion**，正是預測 next stroke type + area。
- ShuttleSet/ShuttleSet22 資料集、**CoachAI Badminton Challenge 2023**(同類競賽)、2026 桌球版(MJSSM, Wu)。
- 這條線出自台灣 AI 圈 → **AI CUP 2026 很可能基於此任務、前段隊伍可能用 ShuttleNet 系列。**

**請你們辯論**：
1. 你們知道 ShuttleNet 在 ShuttleSet/CoachAI 上的**實際分數**嗎？(幫我們校準：它的 next-type / next-area 準度大概多少？是否也卡在我們這種低檔位？)
2. 我們的瓶頸是**泛化**(in-sample 0.96 vs CV 0.30)。ShuttleNet 的「雙選手風格 encoder」會不會比我們的 `P(action|player)` player-marginal 抽到**更可轉移**的選手訊號(=唯一有效槓桿)，因而在 **CV-B** 上漲？還是它一樣會被泛化天花板卡住？
3. **值不值得我花成本移植 ShuttleNet 當新 ensemble 成員？** 還是依「mini 版(GRU+player-emb)只 +0.001」推斷它也只會邊際提升？請給明確建議(做/不做 + 理由)。

## E. 請回給我的格式（一份共識 md）
1. 對 C-1 的共同立場（天花板 ~0.35 是否成立）。
2. 若主張可破：**唯一一個**新方向 + CV-B 漲、CV-C 不崩的理由 + 實作步驟。
3. 對 D 清單的優先序建議（要做哪些、跳過哪些）。
4. 對「公開榜 vs private(最終)」的提交策略建議。

請務實。我不需要長清單，我需要**經得起 CV-B 檢驗的方向**或**誠實的天花板結論**。
