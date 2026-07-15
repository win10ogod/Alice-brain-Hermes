# Alice-brain-Hermes Standalone Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a self-contained Hermes 0.18.x consciousness plugin with its own persistent runtime, daemon, PC/E/ST/RD/A state, official hooks, and CLI, without any Alice-brain project dependency.

**Architecture:** Synchronous Hermes hooks only enqueue bounded event records and read an atomic projection cache. A private worker/daemon owns the SQLite ledger and continuously advances C0; all state is reduced from immutable events. The repository uses only the `alice_brain_hermes` namespace and never imports, installs, starts, or connects to the separate Alice-brain project.

**Tech Stack:** Python `>=3.11,<3.14`, Pydantic 2, stdlib asyncio/sqlite3/argparse, psutil, PyYAML, pytest, pytest-asyncio, Ruff, local Hermes Agent 0.18.2 contract tests.

## Global Constraints

- The distribution is `alice-brain-hermes`; the import root is exactly `alice_brain_hermes`.
- Do not declare or import `alice-brain` / `alice_brain`, use its Git checkout, relative path, daemon, state home, or environment variables.
- The plugin owns `ALICE_BRAIN_HERMES_HOME`, its database, credentials, process lease, daemon, schema, and migrations.
- Hermes hooks are synchronous `def` callbacks. They may only do bounded `put_nowait`, cache reads, and pure payload shaping.
- Hooks never call SQLite, RPC, a provider, or `ctx.llm`; a worker performs slow work.
- `on_session_end`, `on_session_finalize`, and `on_session_reset` never stop the daemon.
- `pre_llm_call` returns only compact ephemeral context. All other observer hooks return `None` unless a documented pre-tool/pre-verify directive is intentionally produced.
- No provider request, tool argument/result, stream, reasoning field, multimodal input, context, model, retry, or output limit is rewritten or reduced.
- Hermes 0.18.2 has no stream-chunk or separate reasoning hook. Capabilities report these as `unobserved`; trace completeness never claims they were captured.
- Full mode requires the project-owned continuous daemon. No embedded or external-project fallback is allowed.
- The repository license and Python package license expression are exactly `MIT`; both wheel and sdist include the root `LICENSE` file.
- Run `python -m compileall -q src`, Ruff, and focused tests at every task boundary.

---

## Planned File Map

- `pyproject.toml`: package, console script, Hermes entry point, pinned development dependencies.
- `src/alice_brain_hermes/core/`: immutable event, identity, world, personality, energy, action, memory, workspace, and reducer models.
- `src/alice_brain_hermes/runtime/`: SQLite ledger, single-writer actor, continuous scheduler, daemon, discovery, credential, and process lease.
- `src/alice_brain_hermes/protocol/`: private JSON-RPC/NDJSON service and authenticated client.
- `src/alice_brain_hermes/hermes/`: bridge, hook callbacks, genesis naming, registration, and Hermes CLI.
- `src/alice_brain_hermes/projections.py`: compact frame, complete trace, health, and action explanation.
- `integration/alice-brain/plugin.yaml`: standalone directory-plugin manifest.
- `integration/alice-brain/__init__.py`: directory-plugin `register` shim.
- `scripts/check_independence.py`: AST, metadata, path, environment, and wheel dependency audit.
- `tests/`: core, runtime, protocol, plugin, hook, CLI, independence, and installed E2E tests.

### Task 1: Package Bootstrap and Independence Contract

**Files:**
- Create: `pyproject.toml`
- Create: `README.md`
- Create: `src/alice_brain_hermes/__init__.py`
- Create: `src/alice_brain_hermes/ids.py`
- Create: `scripts/check_independence.py`
- Test: `tests/test_package.py`
- Test: `tests/test_independence.py`

**Interfaces:**
- Consumes: none.
- Produces: `__version__`, `new_id()`, `validate_id()`, console command `alice-brain-hermes`, and an independence audit executable.

- [ ] **Step 1: Write failing package and independence tests**

```python
from importlib.metadata import entry_points, requires
from pathlib import Path

from alice_brain_hermes import __version__
from alice_brain_hermes.ids import new_id, validate_id


def test_package_identity_and_uuid() -> None:
    value = new_id()
    assert __version__ == "0.1.0"
    assert validate_id(value) == value


def test_no_alice_brain_distribution_dependency() -> None:
    normalized = {item.split(";", 1)[0].strip().lower().replace("_", "-") for item in (requires("alice-brain-hermes") or [])}
    assert not any(item.startswith("alice-brain") and not item.startswith("alice-brain-hermes") for item in normalized)
    assert not Path(".gitmodules").exists()
```

- [ ] **Step 2: Verify package tests fail before bootstrap**

Run: `python -m pytest tests/test_package.py tests/test_independence.py -v`

Expected: collection fails because `alice_brain_hermes` and package metadata do not exist.

- [ ] **Step 3: Add exact package metadata and UUID implementation**

```toml
[project]
name = "alice-brain-hermes"
version = "0.1.0"
requires-python = ">=3.11,<3.14"
dependencies = ["pydantic>=2.12,<3", "psutil>=7,<8", "PyYAML>=6,<7"]

[project.scripts]
alice-brain-hermes = "alice_brain_hermes.cli:main"

[project.entry-points."hermes_agent.plugins"]
alice-brain = "alice_brain_hermes.hermes_plugin"
```

The entry point targets the module, not `:register`, because Hermes loads the module and then reads its `register` attribute. Implement UUID validation with `uuid.UUID`, rejecting non-canonical and non-version-4 identifiers.

- [ ] **Step 4: Implement an AST/metadata independence audit**

`scripts/check_independence.py` must reject AST imports whose root is exactly `alice_brain`, PEP 503-normalized dependency `alice-brain`, sibling checkout paths, `.gitmodules`, `ALICE_BRAIN_HOME`, and service calls to an Alice-brain daemon. It must allow the legitimate root `alice_brain_hermes`.

- [ ] **Step 5: Build, audit, test, and commit**

Run: `uv sync --extra dev && uv build && uv run python scripts/check_independence.py dist/*.whl && uv run python -m compileall -q src && uv run ruff check src tests scripts && uv run pytest tests/test_package.py tests/test_independence.py -v`

Expected: wheel audit and all focused checks pass.

Commit: `git add pyproject.toml uv.lock README.md src tests scripts && git commit -m "build: bootstrap independent Alice-brain-Hermes"`

### Task 2: Immutable Events, Ledger, and Domain Boundaries

**Files:**
- Create: `src/alice_brain_hermes/errors.py`
- Create: `src/alice_brain_hermes/core/events.py`
- Create: `src/alice_brain_hermes/core/state.py`
- Create: `src/alice_brain_hermes/core/reducer.py`
- Create: `src/alice_brain_hermes/runtime/store.py`
- Test: `tests/core/test_reducer.py`
- Test: `tests/runtime/test_store.py`

**Interfaces:**
- Consumes: canonical UUID helpers.
- Produces: `EventEnvelope`, `new_event()`, `BrainState`, `reduce_state()`, `SQLiteLedger.append/list_events/save_snapshot/replay`.

- [ ] **Step 1: Write failing event and store invariant tests**

```python
def test_same_event_is_idempotent_but_conflicting_body_is_rejected(tmp_path) -> None:
    ledger = SQLiteLedger.open(tmp_path / "hermes.db")
    event = new_event("brain.created", new_id(), new_id(), {"name": None})
    first = ledger.append(event)
    second = ledger.append(event)
    assert first.sequence == second.sequence == 1
    with pytest.raises(EventConflictError):
        ledger.append(event.model_copy(update={"payload": {"name": "changed"}}))


def test_replay_is_deterministic(tmp_path) -> None:
    ledger = SQLiteLedger.open(tmp_path / "hermes.db")
    events = [new_event("brain.created", BRAIN, ACTOR, {}), new_event("clock.tick", BRAIN, ACTOR, {"elapsed_seconds": 1.0})]
    for event in events:
        ledger.append(event)
    assert ledger.replay(BRAIN) == ledger.replay(BRAIN)
```

- [ ] **Step 2: Run and verify missing modules fail**

Run: `uv run pytest tests/core/test_reducer.py tests/runtime/test_store.py -v`

Expected: imports fail on the not-yet-created modules.

- [ ] **Step 3: Implement frozen envelope and append-only SQLite WAL ledger**

The envelope contains `schema_version,event_id,event_type,brain_id,sequence,wall_time,monotonic_ns,actor_id,adapter_id,session_id,turn_id,action_id,causation_id,correlation_id,payload`. Allocate sequence inside `BEGIN IMMEDIATE`; fingerprint the entire body; use WAL, foreign-key checks, paginated iteration, schema version, snapshots, and transaction rollback on every failure.

- [ ] **Step 4: Implement the foundation reducer and replay**

`BrainState` starts unnamed and records capabilities, clock, trace completeness, and raw lifecycle counts. All later domain reducers compose through `reduce_state`; no hook or LLM mutates it directly.

- [ ] **Step 5: Verify and commit**

Run: `uv run python -m compileall -q src && uv run ruff check src tests && uv run pytest tests/core/test_reducer.py tests/runtime/test_store.py -v`

Expected: idempotency, conflict, monotonic sequence, pagination, snapshot, and replay tests pass.

Commit: `git add src/alice_brain_hermes/errors.py src/alice_brain_hermes/core src/alice_brain_hermes/runtime/store.py tests/core tests/runtime && git commit -m "feat: add independent Hermes event ledger"`

### Task 3: PC/E/ST/RD/A, World Grounding, and Continuous C0

**Files:**
- Create: `src/alice_brain_hermes/core/identity.py`
- Create: `src/alice_brain_hermes/core/world.py`
- Create: `src/alice_brain_hermes/core/personality.py`
- Create: `src/alice_brain_hermes/core/action.py`
- Create: `src/alice_brain_hermes/core/workspace.py`
- Create: `src/alice_brain_hermes/core/cognition.py`
- Create: `src/alice_brain_hermes/runtime/engine.py`
- Create: `src/alice_brain_hermes/runtime/scheduler.py`
- Test: `tests/core/test_consciousness.py`
- Test: `tests/runtime/test_scheduler.py`

**Interfaces:**
- Consumes: immutable events and ledger.
- Produces: three-layer PC, action-indexed E, isolated ST branches, `RDPhase`, grounded action records, recurrent limited workspace, `LocalCognitionPort`, and `ConsciousEngine`.

- [ ] **Step 1: Write failing consciousness invariants**

```python
def test_simulation_and_untrusted_receipt_cannot_change_observed() -> None:
    state = BrainState.genesis(BRAIN)
    state = reduce_state(state, event("simulation.created", {"proposition_id": "s1", "content": {"door": "open"}}))
    state = reduce_state(state, event("action.receipt", {"trusted": False, "effect_confirmed": True, "observations": [{"proposition_id": "o1", "content": {"door": "open"}}]}))
    assert state.world.observed == ()
    assert {item.proposition_id for item in state.world.simulated} == {"s1"}


def test_action_requires_prepare_dispatch_and_trusted_receipt() -> None:
    state = reduce_many(BrainState.genesis(BRAIN), prepared_action_events())
    assert state.actions["a1"].effect_confirmed is True
    with pytest.raises(DomainInvariantError):
        reduce_state(BrainState.genesis(BRAIN), event("action.receipt", {"action_id": "a1", "status": "success"}))
```

- [ ] **Step 2: Verify tests fail on missing domain models**

Run: `uv run pytest tests/core/test_consciousness.py tests/runtime/test_scheduler.py -v`

Expected: collection fails on domain modules.

- [ ] **Step 3: Implement typed state and deterministic local cognition**

PC has traits, adaptations, and narrative/ideal-self layers with bounded update rates. E is a per-action vector containing deficits, salience, urgency, valence/arousal, control, resources, cost, and personality relevance. ST contains isolated counterfactual branches. RD is an enum with `simulate`, `prepare`, and `reconstruct`. A records proposed/prepared/dispatched/receipt/reconstructed phases and separate nullable execution/effect confirmation.

`LocalCognitionPort` deterministically turns ignited structured content into alternative branches, expected consequences, uncertainty, and reflection records; it works off-turn without an external project or provider and records `cognition_mode=local`.

- [ ] **Step 4: Implement recurrent bounded workspace and real-time scheduler**

Specialists produce candidates for prediction error, drives, incomplete action, memory, and self/world conflicts. Attention is bounded and recurrent: broadcast contents change the next specialist cycle. The scheduler records real `elapsed_seconds`, never invents missed ticks, catches pulse/coordinator errors into `runtime.failure`, exposes health, and continues future ticks.

- [ ] **Step 5: Verify and commit**

Run: `uv run python -m compileall -q src && uv run ruff check src tests && uv run pytest tests/core/test_consciousness.py tests/runtime/test_scheduler.py -v`

Expected: world isolation, identity, PC/E, action lifecycle, recurrent feedback, off-turn tick, failure recovery, and deterministic replay tests pass.

Commit: `git add src/alice_brain_hermes/core src/alice_brain_hermes/runtime tests/core tests/runtime && git commit -m "feat: implement Hermes consciousness runtime"`

### Task 4: Private Authenticated Daemon and Non-Competing Writer

**Files:**
- Create: `src/alice_brain_hermes/protocol/models.py`
- Create: `src/alice_brain_hermes/protocol/service.py`
- Create: `src/alice_brain_hermes/protocol/client.py`
- Create: `src/alice_brain_hermes/runtime/lease.py`
- Create: `src/alice_brain_hermes/runtime/discovery.py`
- Create: `src/alice_brain_hermes/runtime/daemon.py`
- Test: `tests/protocol/test_daemon.py`

**Interfaces:**
- Consumes: `ConsciousEngine` and ledger.
- Produces: per-connection initialized RPC session, authenticated loopback daemon, secret reference record, `DaemonClient`, and process ownership lease.

- [ ] **Step 1: Write failing daemon contract tests**

```python
async def test_business_method_requires_initialize(service) -> None:
    response = await service.handle(request("brain.create", {"actor_id": ACTOR}))
    assert response.error.code == "not_initialized"


def test_discovery_record_does_not_contain_token(tmp_path) -> None:
    record = start_test_daemon(tmp_path)
    body = json.loads((tmp_path / "daemon.json").read_text())
    assert "token" not in body
    assert Path(body["credential_ref"]).stat().st_mode & 0o077 == 0
```

- [ ] **Step 2: Verify daemon tests fail**

Run: `uv run pytest tests/protocol/test_daemon.py -v`

Expected: protocol/runtime daemon modules are absent.

- [ ] **Step 3: Implement server-authoritative negotiation and auth**

Each connection owns initialized state. `initialize` compares the requested profile with the engine's actual capabilities; all other methods except health reject before initialize. Discovery stores PID, instance nonce, endpoint, protocol/package version, and credential reference. The credential is mode 0600. RPC errors preserve `{code,message,data}`.

- [ ] **Step 4: Implement lease, readiness, and lifecycle**

Acquire a runtime-home process lease before opening SQLite. A second daemon refuses with `runtime_owned`. Start waits for a readiness handshake rather than sleeping. Only explicit shutdown stops the daemon; crashes leave a stale lease recoverable after PID/nonce validation.

- [ ] **Step 5: Verify and commit**

Run: `uv run python -m compileall -q src && uv run ruff check src tests && uv run pytest tests/protocol/test_daemon.py -v`

Expected: auth, handshake, wrong-token, lease, readiness, shutdown, stale recovery, C0-after-client-exit, and restart/replay tests pass.

Commit: `git add src/alice_brain_hermes/protocol src/alice_brain_hermes/runtime tests/protocol && git commit -m "feat: add private Hermes consciousness daemon"`

### Task 5: Hermes Plugin Packaging and Registration

**Files:**
- Create: `src/alice_brain_hermes/hermes_plugin.py`
- Create: `src/alice_brain_hermes/hermes/registration.py`
- Create: `integration/alice-brain/plugin.yaml`
- Create: `integration/alice-brain/__init__.py`
- Test: `tests/hermes/test_registration.py`

**Interfaces:**
- Consumes: Hermes `PluginContext.register_hook()` and `register_cli_command()`.
- Produces: one idempotent `register(ctx)` through both module entry point and directory-plugin shim.

- [ ] **Step 1: Write failing manifest/entry-point tests**

```python
def test_manifest_is_explicit_standalone() -> None:
    manifest = yaml.safe_load(Path("integration/alice-brain/plugin.yaml").read_text())
    assert manifest["name"] == "alice-brain"
    assert manifest["kind"] == "standalone"


def test_entry_point_loads_module_with_register() -> None:
    ep = next(item for item in entry_points(group="hermes_agent.plugins") if item.name == "alice-brain")
    module = ep.load()
    assert callable(module.register)
```

- [ ] **Step 2: Verify discovery tests fail**

Run: `uv run --with-editable ../hermes-agent pytest tests/hermes/test_registration.py -v`

Expected: manifest and registration modules do not exist.

- [ ] **Step 3: Implement manifest and idempotent registration**

`plugin.yaml` declares `kind: standalone` and every supported hook. `register(ctx)` creates no daemon, network, SQLite, or provider work. It registers callbacks and lazy CLI setup only. Repeated discovery on the same context must not duplicate workers or hooks.

- [ ] **Step 4: Verify and commit**

Run: `uv run --with-editable ../hermes-agent python -m compileall -q src && uv run --with-editable ../hermes-agent ruff check src tests && uv run --with-editable ../hermes-agent pytest tests/hermes/test_registration.py -v`

Expected: directory and pip discovery, declared hooks, CLI registration, and repeated registration pass.

Commit: `git add pyproject.toml uv.lock src/alice_brain_hermes/hermes_plugin.py src/alice_brain_hermes/hermes integration tests/hermes/test_registration.py && git commit -m "feat: package standalone Hermes plugin"`

### Task 6: Bounded Non-Blocking Hook Bridge and Complete Gap Reporting

**Files:**
- Create: `src/alice_brain_hermes/hermes/bridge.py`
- Create: `src/alice_brain_hermes/hermes/hooks.py`
- Create: `src/alice_brain_hermes/projections.py`
- Test: `tests/hermes/test_hooks.py`
- Test: `tests/hermes/test_nonblocking.py`

**Interfaces:**
- Consumes: exact Hermes 0.18.2 observer payloads and daemon client.
- Produces: ordinary synchronous hook callbacks, bounded queue, atomic state-frame cache, gap events, and health projection.

- [ ] **Step 1: Write failing no-I/O and gap tests**

```python
def test_hook_never_touches_transport_or_database(blocked_transport, hooks) -> None:
    started = time.perf_counter()
    assert hooks.post_api_request(response={"ok": True}, correlation_id="c1") is None
    assert time.perf_counter() - started < 0.01
    assert blocked_transport.calls == []


def test_full_queue_marks_trace_gap(hooks) -> None:
    hooks.bridge.queue = queue.Queue(maxsize=1)
    hooks.on_session_start(session_id="s1")
    hooks.on_session_start(session_id="s2")
    assert hooks.bridge.health.trace_complete is False
    assert hooks.bridge.health.dropped_events == 1
```

- [ ] **Step 2: Verify hook tests fail**

Run: `uv run --with-editable ../hermes-agent pytest tests/hermes/test_hooks.py tests/hermes/test_nonblocking.py -v`

Expected: bridge/hooks modules are absent.

- [ ] **Step 3: Implement callbacks and worker**

Callbacks are normal `def **kwargs`, sanitize only copies, add `telemetry_schema_version`, call bounded `put_nowait`, and return immediately. The worker sends queued events and refreshes an atomic compact frame. Queue overflow records a dropped sequence range; the next connection emits `trace.gap` before later events and keeps `trace_complete=false`.

`pre_llm_call` returns only the cached frame. `post_llm`, API, approval, subagent, and session observers return `None`. `chunk_capture` and `reasoning_capture` are explicitly `unobserved` for Hermes 0.18.2.

- [ ] **Step 4: Verify and commit**

Run: `uv run --with-editable ../hermes-agent python -m compileall -q src && uv run --with-editable ../hermes-agent ruff check src tests && uv run --with-editable ../hermes-agent pytest tests/hermes/test_hooks.py tests/hermes/test_nonblocking.py -v`

Expected: payload mapping, non-blocking behavior, gap recovery, session boundary semantics, and capability-gap tests pass.

Commit: `git add src/alice_brain_hermes/hermes src/alice_brain_hermes/projections.py tests/hermes && git commit -m "feat: observe Hermes lifecycle without blocking"`

### Task 7: Agent Naming, Tool Grounding, Approvals, and Subagents

**Files:**
- Create: `src/alice_brain_hermes/hermes/identity.py`
- Modify: `src/alice_brain_hermes/hermes/hooks.py`
- Modify: `src/alice_brain_hermes/hermes/bridge.py`
- Test: `tests/hermes/test_identity.py`
- Test: `tests/hermes/test_action_lifecycle.py`

**Interfaces:**
- Consumes: worker-side `ctx.llm.complete_structured`, pre/post tool, approval, subagent, and pre-verify payloads.
- Produces: causally recorded identity claim, action lifecycle, actor boundaries, and bounded verify directive.

- [ ] **Step 1: Write failing causal identity/action tests**

```python
def test_identity_name_is_agent_generated_and_causally_recorded(worker) -> None:
    name = worker.ensure_identity(FakeStructuredLLM(name="Mira", reason="chosen"))
    assert name == "Mira"
    assert worker.event_types()[-4:] == ["cognition.requested", "cognition.completed", "c1.deliberated", "identity.claimed"]


def test_tool_success_does_not_confirm_effect_without_evidence(hooks, worker) -> None:
    hooks.pre_tool_call(tool_name="write", tool_call_id="a1", arguments={"path": "x"})
    hooks.post_tool_call(tool_name="write", tool_call_id="a1", status="ok", result={})
    action = worker.action("a1")
    assert action.execution_confirmed is True
    assert action.effect_confirmed is None
```

- [ ] **Step 2: Verify identity/action tests fail**

Run: `uv run --with-editable ../hermes-agent pytest tests/hermes/test_identity.py tests/hermes/test_action_lifecycle.py -v`

Expected: identity and complete lifecycle behavior are missing.

- [ ] **Step 3: Implement worker-side naming and lifecycle reducers**

Only the worker may call `ctx.llm.complete_structured`, and never from a hook callback. A local conflict produces a visible conflict result and another bounded agent choice; no default name or numeric suffix exists. Pre-tool may intentionally return documented block/approve directives only when configured. Post-tool creates a trusted receipt with execution/effect kept separate. Approval responses are observations. Subagent mapping keys on `child_session_id`; child identity never merges with the parent.

- [ ] **Step 4: Implement bounded pre-verify**

Pre-verify returns one continuation directive only when the cached projection reports unresolved evidence and the turn has not already continued. All other cases return `None`.

- [ ] **Step 5: Verify and commit**

Run: `uv run --with-editable ../hermes-agent python -m compileall -q src && uv run --with-editable ../hermes-agent ruff check src tests && uv run --with-editable ../hermes-agent pytest tests/hermes/test_identity.py tests/hermes/test_action_lifecycle.py -v`

Expected: conflicts, causal naming, tool error/unknown/late receipt, approvals, subagents, and bounded verification pass.

Commit: `git add src/alice_brain_hermes/hermes tests/hermes && git commit -m "feat: ground Hermes identity and actions"`

### Task 8: Standalone and Hermes CLI, Installed E2E, and Documentation

**Files:**
- Create: `src/alice_brain_hermes/cli.py`
- Create: `src/alice_brain_hermes/hermes/cli.py`
- Create: `docs/hermes-integration.md`
- Modify: `README.md`
- Test: `tests/cli/test_cli.py`
- Test: `tests/e2e/test_installed_plugin.py`

**Interfaces:**
- Consumes: private daemon lifecycle, projections, plugin context, and Hermes 0.18.2.
- Produces: `alice-brain-hermes daemon|doctor|trace|identity` and `hermes alice-brain start|stop|status|doctor|trace|identity`.

- [ ] **Step 1: Write failing CLI and installed lifecycle tests**

```python
def test_machine_error_has_stable_schema(cli) -> None:
    result = cli("trace", "--brain-id", "missing")
    assert result.returncode != 0
    assert result.stdout == ""
    assert set(json.loads(result.stderr)) >= {"code", "message", "data"}


def test_session_events_do_not_stop_daemon(installed_plugin) -> None:
    before = installed_plugin.status()["instance_nonce"]
    installed_plugin.emit("on_session_end")
    installed_plugin.emit("on_session_finalize")
    installed_plugin.emit("on_session_reset")
    assert installed_plugin.status()["instance_nonce"] == before
```

- [ ] **Step 2: Verify CLI/E2E tests fail**

Run: `uv run --with-editable ../hermes-agent pytest tests/cli/test_cli.py tests/e2e/test_installed_plugin.py -v`

Expected: CLI modules and installed workflow are missing.

- [ ] **Step 3: Implement both command trees and documentation**

All machine output is JSON. Start uses readiness handshake; stop is the only normal daemon stop. Status reports actual runtime mode, cognition mode, trace completeness, dropped events, scheduler health, and unobserved Hermes fields. Doctor checks version, plugin enabled state, home permissions, credential, PID/nonce, DB schema, scheduler, and bridge connection.

Document installation, `hermes plugins enable alice-brain`, the fact that bare `hermes --help` does not discover lazy plugin commands, data paths, capabilities, and exact verification commands.

- [ ] **Step 4: Run full acceptance and independence evidence**

Run:

```bash
uv build
uv run python scripts/check_independence.py dist/*.whl
uv run python -m compileall -q src
uv run ruff check src tests scripts
uv run --with-editable ../hermes-agent pytest -q
uv run --with-editable ../hermes-agent alice-brain-hermes daemon start
uv run --with-editable ../hermes-agent alice-brain-hermes status
uv run --with-editable ../hermes-agent alice-brain-hermes daemon stop
```

Expected: all tests pass; the installed E2E records at least two off-turn C0 ticks, preserves one `brain_id` across bridge restart, keeps one daemon nonce across end/finalize/reset, grounds tool effects only with evidence, and verifies `find_spec('alice_brain') is None` in the clean-wheel test.

- [ ] **Step 5: Commit final project acceptance**

Commit: `git add src docs tests scripts README.md pyproject.toml uv.lock && git commit -m "feat: ship independent Alice-brain-Hermes"`

### Task 9: MIT Release License

**Files:**
- Create: `LICENSE`
- Modify: `pyproject.toml`
- Modify: `README.md`
- Modify: `tests/test_package.py`

**Interfaces:**
- Consumes: the existing Hatchling wheel and sdist build.
- Produces: repository-level MIT terms, `License-Expression: MIT`, and packaged license files in both release artifacts.

- [ ] **Step 1: Write failing metadata and artifact tests**

```python
from importlib.metadata import metadata
from pathlib import Path


def test_project_declares_mit_license() -> None:
    assert "MIT License" in Path("LICENSE").read_text(encoding="utf-8")
    assert 'license = "MIT"' in Path("pyproject.toml").read_text(encoding="utf-8")
    assert metadata("alice-brain-hermes")["License-Expression"] == "MIT"
```

- [ ] **Step 2: Verify the license tests fail before metadata changes**

Run: `uv run pytest tests/test_package.py -v`

Expected: the root `LICENSE` and MIT package expression are absent.

- [ ] **Step 3: Add the standard MIT text and PEP 639 metadata**

Set `license = "MIT"` and `license-files = ["LICENSE"]` in `[project]`, add the
standard MIT grant and warranty text with copyright year `2026`, and document
the license in `README.md`.

- [ ] **Step 4: Build and verify both release formats**

Run: `uv build && uv run pytest tests/test_package.py -v && uv run python scripts/check_independence.py dist/*.whl && unzip -l dist/*.whl | grep '/licenses/LICENSE$' && tar -tf dist/*.tar.gz | grep '/LICENSE$'`

Expected: tests and independence audit pass; wheel metadata contains
`License-Expression: MIT`, and both wheel and sdist contain `LICENSE`.

- [ ] **Step 5: Commit**

Commit: `git add LICENSE README.md pyproject.toml tests/test_package.py docs/superpowers && git commit -m "docs: license Alice-brain-Hermes under MIT"`
