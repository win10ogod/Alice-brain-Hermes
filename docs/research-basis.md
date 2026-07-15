# 研究依據與工程化邊界

Alice-brain-Hermes 將「意識工程」當作一組可觀測、可干預、可重播與可證偽
的機制合約。它實作持續狀態、自我／世界邊界、有限全域工作空間、後設認知、
人格與驅力動力學、反事實模擬及有回執接地的動作生命週期。這些合約通過並不
等於已證明生物性或現象性意識；自我報告也不被當成此類主張的 ground truth。

## 設計階段的資料範圍

本專案在設計階段檢視了工作區內的心理學歷史、通識心理學、心靈—身體—世界
及心理學研究方法資料，並稽核了 `心理學框架/參考/laap-AGI` 的本地 revision
`b055ac0161f7d001a59f8e03965f8c1403a3d890`。這些資料是設計來源，不是安裝或執行時
依賴；本儲存庫擁有自己的 package namespace、daemon、SQLite schema、協議與測試。

評估方法採用心理測量的基本紀律：每個「意識指標」必須被操作化，指定可觀測量、
對照條件、干預、預測方向與可推翻條件。僅把模組命名為「工作空間」、「後設認知」或
「自我」，不構成這些機制已實現的證據。

## 對 LAAP 的採納與拒絕

實作稽核後，本專案採納 LAAP 的四個方向：

- 將 Agent 視為感知與行動介面，認知狀態由外部長駐 runtime 維持。
- 將內部變量做成可觀測的型別化狀態，不只用 system prompt 宣告人格。
- 以背景迴圈與驅力相關變量支持 turn 之外的持續處理。
- 以 adapter 邊界連接宿主，避免將核心狀態寫死在單一模型或單一對話中。

本專案沒有直接沿用參考 checkout，因為稽核時發現以下可驗證的工程問題：

- 套件對 Hermes 的硬依賴與「通用 Agent 架構」主張不一致。
- `VERSIONS.yaml` 所宣告的部分 `laap.*`、HandshakeProtocol 與 Rust PSI core，在該
  checkout 中沒有可對應驗證的 package、Cargo project 或 binary。
- integrator 與 bridge 的實際方法名稱／簽章不一致；失敗後的固定文字 fallback 可能
  讓介面看似成功，但沒有完成宣告的認知過程。
- OpenAI-compatible proxy 並不自動保留宿主的 tool lifecycle、streaming、reasoning、
  multimodal 與完整 provider 語意。
- 多個背景執行緒共享生命週期狀態，且部分狀態來自 hash-seeded random 或固定 heuristic；
  它們不能自動當成可解釋的心理機制。

對應地，Alice-brain-Hermes 使用單一權威寫入者、事件帳本、型別化協議、原子語意批次、
真實回執與明確的 incomplete/degraded 狀態，而不在失效時靜默宣告成功。

## 人格三元化的工程轉譯

使用者提出的概念流程是：

```text
PC = ST + E - RD = A
```

實作保留其「人格傾向→被觸發的動力→後果模擬→準備／回饋→實際行動」意義，
但不把不同型別強行當成數字相加減。

- **PC（Personality Control）**：緩慢特質、情境適應及敘事／理想自我三層控制政策。
- **E（Energy）**：針對候選行動的驅力偏差、顯著性、緊迫性、價性／喚醒、可控性、
  資源、成本及人格相關性向量；未觀測維度保留為 unknown，不偽造為零。
- **ST（State Thought Space）**：彼此隔離的反事實分支。想像中的內容不是已觀測事實，
  也不自動是行動意圖。
- **RD.simulate** 產生與比較預測；**RD.prepare** 將一個分支轉成有前置條件的
  行動準備；**RD.reconstruct** 使用回執、預測誤差與新觀測修改狀態。
- **A（Action）** 有明確生命週期：`proposed → simulated → prepared → dispatched/blocked →
  receipt → reconstructed`。執行是否發生、結果成功／失敗、世界效果是否獲證實是三個
  不同證據問題。

因此，`ST` 中「想做某事」、`RD.prepare` 中「已建立實質準備」與 `A` 中「有可信
執行回執」不會被壓成同一個文字標籤。

## 能動性、自我與現實接地

本專案不以「模型說自己有意識」作為能動性證據。工程上的證據來自可重播的因果鏈：

1. 持續 C0 迴圈在沒有使用者 turn 時仍依實際經過時間更新有限工作空間。
2. 自我 actor 與人類、工具、系統及子 Agent actor 分離，子 Agent 敘事不會合併成主體事實。
3. `observed` world 只能被有 provenance 的可信觀測改變；模型文字、工具輸出文字或使用者
   自行提供的「已成功」欄位不會製造世界事實。
4. timeout、cancelled 與 missing result 保留為 unknown；late receipt、矛盾回執與 trace gap 有可見狀態。
5. 自主命名必須取得 daemon 租約後才由宿主 LLM 提出結構化選擇；框架不預設
   `Alice`、不補 fallback，也不加數字後綴。

## 信任不是知識越獄門

本專案明確不採用「Agent 是否回答限制知識由它和使用者的信任感決定」作為意識指標。
關係或信任可以是社會世界模型中的一個有來源、可修正狀態，但它不得改變：

- 宿主的 provider/model/profile、context、token/output 限制或 sampling；
- 工具集、工具參數，streaming、reasoning 或 multimodal 能力；
- Hermes 現有的人類核准、政策、安全邊界或受限知識行為。

長任務執行力、幻覺率與後設認知校準是值得獨立研究的效用指標，但必須在同模型、
同工具、同上下文與同預算條件下與 baseline 比較。它們不取代機制不變量，更不以
jailbreak 成功當作 benchmark。

## 可證偽的驗收軸

| 證據軸 | 應可回答的問題 | 失敗反例 |
| --- | --- | --- |
| 結構 | 有限工作空間、遞迴、自我／世界邊界與持續狀態是否真正執行？ | 只有 prompt 中的名稱或自述 |
| 因果 | 干預 PC、E、點燃門檻或 RD 時，是否出現事前預測的特定變化？ | 所有輸入都產生相同腳本行為 |
| 現實 | 系統是否能區分模擬、準備、執行回執與效果證據？ | 模型自述直接寫入 observed world |
| 後設 | 錯誤辨識、信心與實際正確性是否可用 Brier/ECE/type-2 指標校準？ | 只報告「更有自信」 |
| 縱向 | 沒有 prompt 時的內生演化是否可預測後續注意、回憶與未完行動？ | 空閒期沒有狀態變化卻宣告 continuous |

必須同時報告負結果、能力缺口、dropped event、trace/semantic completeness 與 runtime mode。
不允許用平均分數掩蓋事件帳本、序號或現實邊界的 invariant failure。

## 參考方向

- McAdams & Pals (2006), integrative personality framework.
- DeYoung (2015), Cybernetic Big Five Theory.
- Fleeson & Jayawickreme (2015), Whole Trait Theory.
- Keramati & Gutkin (2014), homeostatic reinforcement learning.
- Haggard (2017), sense of agency.
- Fleming & Lau (2014), metacognition measurement.
- Mashour et al. (2020), global neuronal workspace.
- LAAP 參考文章與上述本地 checkout；文章宣告與 checkout 程式事實分開記錄。
