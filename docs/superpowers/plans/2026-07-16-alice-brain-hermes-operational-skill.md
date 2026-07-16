# Alice-brain-Hermes Operational Skill Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a minimal repository-owned Agent Skill for starting, observing, diagnosing, verifying, and tracing Alice-brain-Hermes.

**Architecture:** Keep the complete workflow in one concise `SKILL.md`; add only the required OpenAI interface metadata. Support the native CLI and its Hermes-hosted equivalent without adding scripts, references, assets, or auxiliary Skill files.

**Tech Stack:** Agent Skills (`SKILL.md`), YAML metadata, Alice-brain-Hermes CLI, Hermes CLI

## Global Constraints

- The Skill directory contains exactly `SKILL.md` and `agents/openai.yaml`.
- The Skill covers exactly start, observe, diagnose, verify, and trace.
- The Skill contains no project theory, narrative, history, development workflow, or release workflow.
- Read-only operations never start the daemon or create a missing runtime home.
- Native and Hermes command surfaces are not mixed within one evidence run.
- Every verification conclusion cites fresh command evidence.

---

### Task 1: Create and validate `operating-alice-brain-hermes`

**Files:**
- Create: `.agents/skills/operating-alice-brain-hermes/SKILL.md`
- Create: `.agents/skills/operating-alice-brain-hermes/agents/openai.yaml`

**Interfaces:**
- Consumes: `alice-brain-hermes` or the enabled `hermes alice-brain` command
- Produces: repository-local Skill `$operating-alice-brain-hermes`

- [ ] **Step 1: Record the failing no-Skill baseline**

Ask a fresh agent, without loading a Skill, for one concise sequence that starts Alice-brain-Hermes, observes integration state without mutation, diagnoses health, verifies readiness, and paginates a decision trace. Record any invented commands, mixed command surfaces, missing evidence checks, accidental auto-start, or extra material in the ignored SDD report.

- [ ] **Step 2: Initialize the Skill**

Run:

```bash
python3 /home/win10/.codex/skills/.system/skill-creator/scripts/init_skill.py \
  operating-alice-brain-hermes \
  --path .agents/skills \
  --interface 'display_name=Operate Alice-brain-Hermes' \
  --interface 'short_description=Inspect Alice-brain-Hermes runtime evidence' \
  --interface 'default_prompt=Use $operating-alice-brain-hermes to start, observe, diagnose, verify, or trace this plugin runtime.'
```

Expected: the Skill directory contains `SKILL.md` and `agents/openai.yaml` only.

- [ ] **Step 3: Replace the generated template with the minimal Skill**

Write `.agents/skills/operating-alice-brain-hermes/SKILL.md` exactly as follows:

```markdown
---
name: operating-alice-brain-hermes
description: Use when starting, observing, diagnosing, verifying, or tracing an Alice-brain-Hermes plugin runtime.
---

# Operating Alice-brain-Hermes

Use the installed CLI help as the command authority. Select one surface for the evidence run: `alice-brain-hermes` for the native CLI, or `hermes alice-brain` when verifying the enabled Hermes integration. Run that surface's `--help` first and do not mix surfaces in one conclusion.

## Workflow

| Goal | Commands on the selected surface | Required evidence |
|---|---|---|
| Start | `start`, then `status` | Readiness, owner PID, generation, and runtime identity |
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
```

- [ ] **Step 4: Validate structure and scope**

Run:

```bash
python3 /home/win10/.codex/skills/.system/skill-creator/scripts/quick_validate.py .agents/skills/operating-alice-brain-hermes
find .agents/skills/operating-alice-brain-hermes -type f -print | sort
wc -w .agents/skills/operating-alice-brain-hermes/SKILL.md
rg -n 'TO[D]O|TB[D]|consciousness theory|personality|trust|policy|release workflow' .agents/skills/operating-alice-brain-hermes || true
```

Expected: validation succeeds; exactly two files exist; the Skill is under 500 words; the scan prints no matches.

- [ ] **Step 5: Forward-test with the Skill**

Give a fresh agent the same operational request and explicit access to `$operating-alice-brain-hermes`. Verify that it uses only the five workflows, chooses one valid command surface, does not auto-start during observation, preserves diagnostic evidence, and paginates with the returned cursor.

- [ ] **Step 6: Commit**

```bash
git add .agents/skills/operating-alice-brain-hermes
git commit -m "feat: add Hermes operational skill"
```
