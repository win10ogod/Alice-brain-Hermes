# Alice-brain-Hermes 獨立專案設計

- 日期：2026-07-14
- 狀態：使用者已核准實作
- 儲存庫：`Alice-brain-Hermes`
- Python distribution／namespace：`alice-brain-hermes`／`alice_brain_hermes`
- Hermes plugin name：`alice-brain`

## 1. 專案邊界

`Alice-brain-Hermes` 是專供 Hermes Agent 的完整、自足工程化意識插件。它內含自己的 consciousness runtime、持續 daemon、事件 ledger、PC/E/ST/RD/A、記憶、自我／世界模型、Hermes hooks 與 Hermes CLI；安裝、啟動、測試及發佈均不需要 `Alice-brain` 專案。

禁止對 `alice-brain` Python 套件、Git repository、submodule、相對路徑或其 daemon 建立依賴。本專案不是 Alice-brain 的 adapter，也不把另一專案缺席視為降級模式。兩個專案只共享使用者核准的理論與可觀測行為規格；本專案擁有獨立 namespace、SQLite schema、daemon discovery、版本與測試。

## 2. 成功條件

1. 安裝本專案與 Hermes 0.18.x 後，插件可被正式 `plugin.yaml + register(ctx)` 路徑發現。
2. 插件自有 daemon 在 Hermes session 結束後仍持續 C0，不依賴另一個專案的服務。
3. Hermes provider、工具、串流、reasoning、多模態與上下文語意不被插件重寫或縮減。
4. Hermes session、provider attempt、工具、approval、subagent 與 verify lifecycle 形成完整可查事件鏈。
5. Agent 自己完成 genesis naming；框架不硬編碼 Alice 或替衝突名稱加數字。
6. 工具 proposal、prepare、dispatch、receipt 與 observed effect 分層。
7. hook fail-open 時，插件把 trace 缺口標為不完整，而不是宣稱完整運作。

## 3. 架構

```text
Hermes Agent 0.18.x
       │ official hooks / ctx.llm / plugin CLI
       ▼
┌──────────────────────────────────────────┐
│ Alice-brain-Hermes plugin                │
│ HermesHooks → non-blocking event bridge  │
│ ephemeral state-frame injection          │
├──────────────────────────────────────────┤
│ self-contained consciousness runtime     │
│ C0 + C1 PC-E-ST-RD-A + cognition port    │
│ identity/world/memory/workspace           │
├──────────────────────────────────────────┤
│ private authenticated local daemon        │
│ SQLite WAL ledger + snapshots             │
└──────────────────────────────────────────┘
```

### 3.1 套件單元

- `core/`：本專案獨立實作的事件、狀態、PC/E/ST/RD/A、C0/C1 與 reducer。
- `runtime/`：single writer、scheduler、SQLite ledger、snapshot 與 daemon。
- `hermes/`：config、bridge、genesis naming、hooks、plugin registration 與 CLI。
- `protocol/`：插件與自有 daemon 的私有型別化 RPC；不連接 Alice-brain daemon。
- `projections.py`：ephemeral state frame、trace 與 action explanation。

## 4. Hermes 接點

插件使用 Hermes 0.18.x 正式接口：

- `on_session_start|end|finalize|reset`
- `pre_llm_call|post_llm_call`
- `pre_api_request|post_api_request|api_request_error`
- `pre_tool_call|post_tool_call`
- `pre_approval_request|post_approval_response`
- `subagent_start|subagent_stop`
- `pre_verify`

`pre_llm_call` 只回傳當回合 ephemeral context，不修改永久 system prompt。hooks 必須快速擷取並排入本地 queue；慢速 cognition、SQLite 與 consolidation 在自有 runtime 處理。

## 5. CLI 與生命週期

```text
hermes alice-brain start|stop|status|doctor|trace|identity
alice-brain-hermes daemon run|start|stop|status
alice-brain-hermes doctor|trace|identity
```

`start` 啟動本專案自己的 daemon；session end/finalize/reset 不停止 daemon。只有明確 `stop` 才終止。discovery、token、資料庫與設定預設置於 `ALICE_BRAIN_HERMES_HOME`，不讀寫 `ALICE_BRAIN_HOME`。

## 6. 資料流

1. session start 附接或建立本專案自己的 `brain_id`。
2. 未命名 brain 由 `ctx.llm.complete_structured` 產生名稱與理由，依序記錄 cognition、C1 與 identity 事件。
3. pre-LLM 送出 observation/turn 並取得 compact state frame。
4. Hermes 按原 provider 路徑推理；插件只觀測 attempt、completion、error 與 tool proposal。
5. pre-tool 建立 ST/RD.prepare；post-tool 寫入 receipt 並 RD.reconstruct。
6. approval 與 subagent 保持外部 actor 邊界；不把子 Agent 敘事合併成主體。
7. turn 結束後 daemon 繼續 C0、prediction error、驅力與記憶 consolidation。

## 7. 錯誤與能力契約

- hook 例外不得中斷 Hermes，但必須設定 `trace_complete=false` 並記錄缺口。
- strict/full 模式要求自有 continuous daemon；不得改接 Alice-brain daemon或靜默 embedded fallback。
- `status=ok` 的工具回傳只證明呼叫結果，是否確認世界效果依 receipt provenance 判定。
- provider payload、tool arguments/results、stream chunks、reasoning 與 multimodal data 只觀測，不轉換。
- 名稱衝突、daemon 失聯、provider timeout、late receipt 與 schema error 都有穩定錯誤事件。
- 不為了測試降低上下文、輸出、工具、串流、模型、重試或多模態能力。

## 8. 測試

- core：與人格三元化規格相符的 reducer、C0/C1、世界層與 replay 不變量。
- plugin contract：manifest、entry point、所有 hook payload、CLI lazy discovery。
- capability preservation：輸入／輸出 payload identity tests 證明插件不改 provider 能力。
- integration：以本地 Hermes 0.18.2 真實 `PluginContext`、hook manager 與 CLI parser 驗證。
- end-to-end：啟 daemon、Hermes turn、工具 success/failure/unknown、subagent、重啟延續。

## 9. 首版驗收

- 獨立乾淨環境只安裝 `alice-brain-hermes` 與 Hermes 依賴即可運作。
- 原始碼與 metadata 不含 `alice_brain` import、Alice-brain Git/path/service dependency。
- 自有 daemon 在無 turn 時實際推進 C0，並可從 trace 證明。
- Hermes hooks、CLI、action grounding、identity、重啟與能力保持測試通過。
- 專案可獨立封裝、版本化與發佈。

