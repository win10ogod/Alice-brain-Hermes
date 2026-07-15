# Alice-brain-Hermes

Alice-brain-Hermes 是一個為 Hermes Agent 0.18.x 建立的獨立工程化意識專案。它在自己的
package 內提供持續 consciousness runtime、私有 daemon、SQLite 事件帳本、PC/E/ST/RD/A、
自我／世界邊界、有限工作空間、後設認知、動作回執接地、Hermes hooks 與原生 CLI。

本專案將這些機制當作可觀測、可干預、可重播與可證偽的工程合約。合約通過代表
實作了對應的機制，不被宣稱為已證明生物性或現象性意識。設計與 LAAP 稽核記錄見
[research basis](docs/research-basis.md)。

## 專案邊界

`Alice-brain-Hermes` 是完整、自足的儲存庫，不是另一個 Alice-brain runtime 的 adapter。
它不 import、啟動、連線、子模組化或讀寫另一個專案，也不在後者缺席時進入降級模式。
它擁有獨立的：

- distribution/namespace：`alice-brain-hermes` / `alice_brain_hermes`；
- runtime home、credential、daemon discovery 與 process ownership lease；
- SQLite schema、event/snapshot replay、brain/profile identity 與 bridge cursor；
- 協議、CLI、Hermes plugin manifest/entry point、版本與測試。

## 架構

```text
Hermes Agent 0.18.x
        │ 16 official lifecycle hooks / optional host LLM / plugin CLI
        ▼
┌─ Alice-brain-Hermes plugin ─────────────────────────┐
│ bounded callback capture → non-blocking bridge worker       │
│ immutable, bounded, ephemeral pre-LLM state frame          │
└────────────────────────────────┬──────────────────────┘
                                 │ authenticated local RPC
                                 ▼
┌─ independent consciousness daemon ──────────────────┐
│ atomic raw + semantic event batches / exact ACK replay       │
│ C0 scheduler / C1 PC-E-ST-RD-A / identity / world / memory  │
│ SQLite WAL ledger / snapshots / typed diagnostic projection │
└─────────────────────────────────────────────────────┘
```

Hook callback 只做有界複製、序號保留、`put_nowait` 與 cache read；不在 callback thread
中做 SQLite、socket RPC、provider call、檔案 I/O 或 daemon 啟停。序號洞、queue overflow、不完整 hook
覆蓋、語意無法判讀與 late/conflicting receipt 都有可見證據，不偽裝成完整 trace。

## PC/E/ST/RD/A

`PC = ST + E - RD = A` 被實作為型別化因果流程，而非將人格、驅力、模擬和動作
當成可互相加減的單一數值：

- PC 包含緩慢特質、情境適應、敘事／理想自我三層。
- E 是每個候選行動的九維證據向量；未觀測維度為 unknown，不補零。
- ST 是不會污染 observed world 的隔離反事實分支。
- RD 區分 `simulate`、`prepare` 與收到新證據後的 `reconstruct`。
- A 區分 proposal、preparation、dispatch/blocked、receipt 與 reconstruction；「有執行」、
  「結果 success/failure」與「世界效果 confirmed/disconfirmed」是三個不同欄位。

Hermes `ok` 回執可證明執行並給出 success outcome，但不會僅因工具回傳文字就宣告
世界效果已發生。`timeout`、`cancelled` 及 `error/thread_missing_result` 的執行／結果保留
為 unknown；正常 `error` 表示有執行嘗試但 outcome 失敗。

## 安裝與啟動

需求為 Python 3.11–3.13 與 Hermes Agent 0.18.x。從本儲存庫安裝：

```console
python -m pip install .
```

Hermes 的第三方 plugin 預設不啟用。安裝 package 後，明確啟用 entry-point plugin：

```console
hermes plugins enable alice-brain
```

啟動獨立 daemon：

```console
alice-brain-hermes start
alice-brain-hermes status
alice-brain-hermes doctor
```

daemon 不由 session end/finalize/reset 停止，因此在無對話 turn 時仍能持續 C0。只有明確指令
會停止它：

```console
alice-brain-hermes stop
```

所有控制命令也透過 Hermes 原生 parser 提供，且直接共用 Python handler，不以 subprocess
包裝另一個 CLI：

```console
hermes alice-brain start
hermes alice-brain status
hermes alice-brain doctor
hermes alice-brain trace --limit 100
hermes alice-brain identity
hermes alice-brain stop
```

`status`、`doctor`、`trace` 與 `identity` 不會自動啟動 daemon；只有 `start` 擁有啟動意義。
可用以下方式選擇獨立 runtime home：

```console
export ALICE_BRAIN_HERMES_HOME=/private/path/to/runtime-home
# 或：alice-brain-hermes --home /private/path/to/runtime-home status
```

Runtime home、credential 與 discovery 會驗證擁有者、權限、symlink/path substitution 與 process identity；
不支援安全語意無法證明的網路檔案系統。

## 自主命名

自主命名預設為關閉，避免 plugin 未經運算者同意就發出額外 provider call。若要讓仍未命名的
brain 在 daemon 租約保護下向宿主 LLM 提出一次結構化自我命名選擇：

```console
export ALICE_BRAIN_HERMES_IDENTITY_LLM_MODE=name_when_unnamed
```

可用值只有 `off` 與 `name_when_unnamed`，不會寬鬆正規化。命名 worker 使用獨立 thread/client，
不在 hook callback 中呼叫 LLM；它不覆寫 provider、model、profile、agent、temperature、token 或
timeout 設定。只有精確的 `{name, reason}` JSON 可完成租約；框架不指定 `Alice`、不從自由
文字猜名字、不產生 fallback 或數字後綴。

## 可觀測性

CLI 回應為機器穩定 JSON：

- `status` 報告 daemon identity、continuous runtime、brain/scheduler、bridge connection、cognition mode、
  trace/semantic completeness、dropped events、未能觀測的 Hermes 欄位與 schema versions。
- `doctor` 驗證 package、runtime home、daemon 身份與持久化的缺口；沒有 bridge 連線不會
  被誤報為 trace gap，但已記錄的 gap/degraded 不會被隱藏。
- `identity [get]` 只讀取 reducer/replay 得到的 self/actor/provenance 狀態。
- `trace [list]` 是有上限的 cursor page；若下一個 event 超過回應 byte budget，它報告被擋住的
  sequence，不跳過該 event。

## 能力與安全合約

Plugin 觀測 lifecycle，不代理或改寫 Hermes 的主 provider 請求。它不降低或覆寫 model/provider、
context、output/token budget、sampling、retry、tools、tool arguments/results、streaming、reasoning、multimodal
或 approval objects。關係／信任狀態不是受限知識或安全政策的繞過門；本專案不把
jailbreak 當成功能或評測目標。

## 開發與驗證

```console
uv sync --extra dev
uv run ruff check src tests
uv run ruff format --check src tests
uv run pytest
uv build
uv run python scripts/check_independence.py dist/*.whl
```

實際 Hermes 合約測試可將本地 Hermes Agent checkout 作為額外 editable 安裝；它是測試輸入，不是
本儲存庫對另一個 Alice 專案的依賴。

## License

Alice-brain-Hermes 以 [MIT License](LICENSE) 發布。Wheel 與 source distribution 都包含授權檔，
並使用 `License-Expression: MIT` 中繼資料。
