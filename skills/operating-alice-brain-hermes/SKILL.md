---
name: operating-alice-brain-hermes
description: Use when starting, observing, diagnosing, verifying, or tracing an Alice-brain-Hermes plugin runtime.
---

# Operating Alice-brain-Hermes

Use the installed CLI help as the command authority. Select one surface for the evidence run: `alice-brain-hermes` for the native CLI, or `hermes alice-brain` when verifying the enabled Hermes integration. Run that surface's `--help` first and do not mix surfaces in one conclusion.

## Workflow

| Goal | Commands on the selected surface | Required evidence |
|---|---|---|
| Start | `start`, then `status` | Readiness, PID, instance nonce, and runtime mode |
| Observe | `status`; `identity`; `trace --limit 100` | Current runtime, replay-derived identity, and recent integration events |
| Diagnose | `status`; `doctor` | Exact exit status and every reported check |
| Verify | Re-run `status`, `doctor`, and a fresh trace page | Fresh runtime and persisted integration evidence supporting each claim |
| Trace | `trace [list] [--brain-id <id>] --after-sequence <cursor> --limit <1-1000>` | Ordered events, cursor, gaps, and brain identifiers |

For a complete trace, begin at cursor `0`; while `.has_more` is true, pass `.next_after_sequence` unchanged.

## Operational rules

- Start only when requested. `status`, `doctor`, `identity`, and `trace` are inspections; they must not auto-start the daemon or create a missing runtime home.
- Use the Hermes surface when the claim concerns Hermes plugin wiring; native CLI health alone does not establish that host integration.
- Preserve stdout, stderr, and exit status. An unhealthy `doctor` result exits nonzero even when it emits structured JSON.
- Report the selected surface, commands run, relevant IDs or sequence numbers, and the evidence-backed conclusion.
