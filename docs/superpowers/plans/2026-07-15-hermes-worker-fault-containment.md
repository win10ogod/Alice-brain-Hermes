# Hermes Worker Fault-Containment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Hermes bridge/bootstrap worker wake, stop, ownership, and terminal-child disposition failure-atomic under persistent `BaseException` faults.

**Architecture:** Boolean control latches are authoritative while `threading.Event` objects are best-effort accelerators. Worker waits poll those latches in bounded slices while preserving the configured reconnect deadline. Worker/child ownership is released only after strict dead proof; clean-sealed child reservations use a distinct idempotent late-after-close disposition instead of normal handoff.

**Tech Stack:** Python 3.11+, `threading`, immutable dataclasses, pytest, Ruff, uv

## Global Constraints

- Work only in `/mnt/f/多模態記憶/.worktrees/Alice-brain-Hermes-task6`.
- Preserve exact capture sequence, one commit/ACK per accepted record, and one bootstrap disposition per reservation.
- Preserve configured reconnect delay; bounded control polling may only return early for an authoritative wake/stop latch.
- Do not restart/reconnect a clean-sealed child and do not create an `N+1 -> 1` logical-stream mapping.
- Unknown liveness retains ownership and prevents duplicate workers.
- Public hook callbacks remain fail-open and non-blocking.

---

### Task 1: Fault-contained notification and deadline-preserving control polling

**Files:**
- Modify: `src/alice_brain_hermes/hermes/bridge.py:500-830,1070-1130,1610-1640`
- Test: `tests/hermes/test_nonblocking.py`

**Interfaces:**
- Consumes: existing `HookBridge.capture`, `capture_reserved`, `request_clean_close`, and `start_worker` APIs.
- Produces: `HookBridge._notify_worker(*, force_start: bool = False) -> None`, an authoritative `_wake_requested` latch, and bounded `_wait_for_worker(timeout: float) -> None`.

- [ ] **Step 1: Write failing notification and 60-second wait tests**

Add these tests:

- `test_persistent_wake_set_failure_starts_worker_and_commits_once`
- `test_persistent_wake_set_failure_interrupts_sixty_second_idle_wait`
- `test_persistent_wait_failure_preserves_sixty_second_reconnect_deadline`
- `test_clean_close_starts_worker_when_wake_set_persistently_fails`

The tests use a delegating event whose selected method always raises
`MemoryError`, verify the capture/close reaches a real HookBridge worker, assert
one exact commit or one exact close, and prove an idle/reconnect timeout of 60
seconds neither blocks wake/stop nor collapses into rapid reconnect attempts.

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
uv run pytest -q \
  tests/hermes/test_nonblocking.py::test_persistent_wake_set_failure_starts_worker_and_commits_once \
  tests/hermes/test_nonblocking.py::test_persistent_wake_set_failure_interrupts_sixty_second_idle_wait \
  tests/hermes/test_nonblocking.py::test_persistent_wait_failure_preserves_sixty_second_reconnect_deadline \
  tests/hermes/test_nonblocking.py::test_clean_close_starts_worker_when_wake_set_persistently_fails
```

Expected: all fail against `dcda303` because wake failure skips worker start,
clean close raises, or the 60-second deadline is handled as one uninterruptible
wait / prematurely shortened retry.

- [ ] **Step 3: Implement the minimal notification and wait controls**

Add a small internal poll constant, `_wake_lock`, and `_wake_requested`. Publish
the latch before event `set`; contain event and start failures independently:

```python
def _notify_worker(self, *, force_start: bool = False) -> None:
    with self._wake_lock:
        self._wake_requested = True
    try:
        self._wake_event.set()
    except BaseException as error:
        self._mark_emergency_failure(error)
    finally:
        if force_start or self._start_worker_on_capture:
            try:
                self.start_worker()
            except BaseException as error:
                self._mark_emergency_failure(error)
```

Route successful `capture`, non-terminal `capture_reserved`, and
`request_clean_close(force_start=True)` through this helper. Implement
`_wait_for_worker` as a monotonic-deadline loop with at-most-50ms event slices;
only consumed wake or stop latches return early. Persistent event faults sleep
within the same deadline and cannot cause a spin or shortened reconnect delay.

- [ ] **Step 4: Run the focused tests and the existing bridge control-fault tests**

Run the four node IDs above plus:

```bash
uv run pytest -q tests/hermes/test_nonblocking.py -k 'control_probe or persistent_stop_probe or persistent_wait'
```

Expected: PASS with exact commit/close counts and bounded stop.

- [ ] **Step 5: Commit Task 1**

```bash
git add src/alice_brain_hermes/hermes/bridge.py tests/hermes/test_nonblocking.py
git commit -m "fix: contain Hermes worker notification faults"
```

### Task 2: Strict stop and child ownership proof

**Files:**
- Modify: `src/alice_brain_hermes/hermes/bridge.py:500-540,1017-1065,1742-1753`
- Modify: `src/alice_brain_hermes/hermes/registration.py:227-260,581-632,805-860`
- Test: `tests/hermes/test_nonblocking.py`
- Test: `tests/hermes/test_registration.py`

**Interfaces:**
- Consumes: `_stop_requested`, `_worker`, `_worker_lock`, and bootstrap `_transport_bridge` ownership.
- Produces: serialized lifecycle operations, `HookBridge._worker_alive_strict() -> bool`, strict bootstrap child cleanup, and stop methods whose normal return proves no live owned worker.

- [ ] **Step 1: Write failing stop/ownership regressions**

Add these tests:

- `test_bridge_stop_contains_dual_signal_failure_during_sixty_second_wait`
- `test_bridge_stop_unknown_liveness_retains_owner_and_prevents_duplicate`
- `test_bootstrap_stop_contains_dual_signal_failure_and_stops_child`
- `test_bootstrap_stop_unknown_outer_liveness_retains_owner`
- parameterized `test_bootstrap_stop_retains_child_without_callable_stop`,
  covering a missing attribute and a present `None` value
- `test_bootstrap_stop_retains_child_when_strict_post_stop_probe_is_unknown`

Dual-signal tests make both stop-event and wake-event `set` fail persistently.
Unknown probes raise `MemoryError` after bounded join. Tests assert degraded
health, retained identity on failure, no second thread construction, and no
live outer/child after a successful stop return.

- [ ] **Step 2: Run the focused tests and verify RED**

Run the six new node IDs (including both parameter cases). Expected: failures
show the first `set` escaping, owner snapshot/join being skipped, missing stop
clearing child ownership, or lossy `worker_started=False` being accepted as
strict dead proof.

- [ ] **Step 3: Implement failure-atomic HookBridge stop**

Add a lifecycle-operation lock used by `start_worker` and
`stop_worker_for_test`. Under `_worker_lock`, publish `_stop_requested=True`,
attempt both control events independently, and snapshot `_worker`. Join outside
`_worker_lock`; then call the strict probe:

```python
def _worker_alive_strict(self) -> bool:
    with self._worker_lock:
        worker = self._worker
    return False if worker is None else worker.is_alive()
```

If the snapshotted worker is explicitly dead, clear the identical pointer and
publish stopped health. If join times out or liveness raises, retain the pointer,
mark degraded, and raise `RuntimeError` so stop cannot report false success.

- [ ] **Step 4: Implement matching bootstrap stop and strict child cleanup**

Serialize bootstrap start/stop with its own lifecycle-operation lock. Use the
same latch-first, independent-set, snapshot, lock-free bounded join, and strict
dead proof for the outer worker. Once the outer is certainly dead, call child
cleanup even when no outer was present.

`_stop_transport_bridge` requires callable `stop_worker_for_test` and callable
`_worker_alive_strict`. Only normal stop return plus strict false clears the
identical child pointer. Every missing, non-callable, live, or throwing result
marks degraded, retains ownership, and raises a bounded cleanup error.

- [ ] **Step 5: Run focused and prior lifecycle tests**

Run:

```bash
uv run pytest -q tests/hermes/test_nonblocking.py -k 'stop and worker'
uv run pytest -q tests/hermes/test_registration.py -k 'stop or cleanup or ownership or restart'
```

Expected: PASS; successful stop leaves no live owner, unknown state retains the
same owner, and prior stop-to-restart tests still serialize without overlap.

- [ ] **Step 6: Commit Task 2**

```bash
git add src/alice_brain_hermes/hermes/bridge.py src/alice_brain_hermes/hermes/registration.py tests/hermes/test_nonblocking.py tests/hermes/test_registration.py
git commit -m "fix: prove Hermes worker ownership before release"
```

### Task 3: Idempotent terminal-child disposition

**Files:**
- Modify: `src/alice_brain_hermes/hermes/bridge.py:60-90,672-830`
- Modify: `src/alice_brain_hermes/hermes/registration.py:52-68,263-312,665-693,892-970`
- Test: `tests/hermes/test_nonblocking.py`
- Test: `tests/hermes/test_registration.py`

**Interfaces:**
- Consumes: `_last_bootstrap_reservation`, HookBridge close seal, bootstrap retained-capture identity, and dropped-event reconciliation.
- Produces: `capture_reserved(...) -> Literal["accepted", "late_after_close"]`, receipt-bound duplicate disposition, `close_sealed`, and `_BootstrapCaptureBuffer.mark_late_after_close(capture)`.

- [ ] **Step 1: Write failing terminal regressions**

Add these tests:

- `test_terminal_capture_duplicate_returns_same_late_disposition`
- `test_bootstrap_terminal_observation_is_accounted_without_handoff_or_ack`
- `test_bootstrap_terminal_retry_is_idempotent_after_publication_failure`
- `test_bootstrap_terminal_gap_moves_drop_without_double_counting`

Use a real clean-sealed HookBridge. Assert no `start_worker`, reconnect,
replacement child, normal `mark_handed_off`, daemon commit, or ACK. Assert the
original capture range advances once, duplicate calls return
`late_after_close`, pending reaches zero, observation adds one drop, and a gap
keeps the same total drop while moving out of pending.

- [ ] **Step 2: Run the focused tests and verify RED**

Run the four node IDs. Expected: failures show `capture_reserved` returning
`None`, bootstrap repeatedly trying to restart the terminal child, and the
reservation remaining pending.

- [ ] **Step 3: Bind disposition to the exact receipt**

Extend `_BootstrapReservationReceipt` with a literal disposition. Its identity
comparison continues to compare only immutable reservation content. On exact
duplicate return `previous.disposition`; on first terminal acceptance store and
return `late_after_close`; on normal acceptance store and return `accepted`.
Expose a read-only `close_sealed` property.

- [ ] **Step 4: Implement terminal bootstrap accounting**

Add `late_after_close: int = 0` to `_BootstrapHealth` and preserve it in health
reconstruction. `mark_late_after_close` verifies retained object identity,
moves `capture_count` into the finalized dropped base, excludes the retained
range from pending metrics, increments late count, sets trace incomplete and
the terminal error, then clears retained state atomically.

In `_bootstrap_worker_main`, do not restart a child whose `close_sealed` is
true. Process its next reservation and branch on the returned disposition:
`late_after_close` calls only `mark_late_after_close`; `accepted` calls only
`mark_handed_off`. A terminal-before-call child returning any other disposition
is an error and remains pending.

- [ ] **Step 5: Run focused and all terminal/capture tests**

Run:

```bash
uv run pytest -q tests/hermes/test_nonblocking.py -k 'bootstrap or close or capture_reserved'
uv run pytest -q tests/hermes/test_registration.py -k 'terminal or handoff or capture_seq or gap'
```

Expected: PASS with one terminal receipt, zero commit/ACK, exact late/drop
health, and prior normal handoff behavior unchanged.

- [ ] **Step 6: Commit Task 3**

```bash
git add src/alice_brain_hermes/hermes/bridge.py src/alice_brain_hermes/hermes/registration.py tests/hermes/test_nonblocking.py tests/hermes/test_registration.py
git commit -m "fix: account terminal bootstrap captures exactly"
```

### Task 4: Formatting and full verification

**Files:**
- Modify mechanically: `tests/hermes/test_registration.py`

**Interfaces:**
- Consumes: Tasks 1-3.
- Produces: review-ready commits and fresh verification evidence.

- [ ] **Step 1: Apply and verify Ruff formatting**

```bash
uv run ruff format tests/hermes/test_registration.py
uv run ruff format --check src tests
uv run ruff check src tests
```

Expected: all commands exit 0, including the previously reported formatting
near old lines 2363/2386.

- [ ] **Step 2: Run the two Hermes files**

```bash
uv run pytest -q tests/hermes/test_nonblocking.py tests/hermes/test_registration.py
```

Expected: zero failures.

- [ ] **Step 3: Repeat race-prone node IDs in five fresh processes**

Run the dual-set, unknown-owner, 60-second-idle wake, terminal retry, and
terminal-gap node IDs five times using a shell loop that launches a fresh
`uv run pytest` process each iteration. Expected: every iteration exits 0.

- [ ] **Step 4: Run the full suite and static checks**

```bash
uv run pytest -q
uv run python -m compileall -q src tests
git diff --check 227b315..HEAD
git status --short
```

Expected: full suite and compile exit 0, no whitespace errors, and only intended
tracked changes before the final commit.

- [ ] **Step 5: Commit any formatting-only remainder**

If formatting changed content not already included in Task 2/3, commit it with:

```bash
git add tests/hermes/test_registration.py
git commit -m "style: format Hermes registration lifecycle tests"
```

- [ ] **Step 6: Request a fresh review of `227b315..HEAD`**

Report commit hashes, exact test counts, race-loop results, Ruff/format/compile
results, `git diff --check`, and clean worktree state. Do not merge or modify the
integration worktree.
