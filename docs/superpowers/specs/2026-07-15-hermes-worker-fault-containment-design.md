# Hermes Worker Fault-Containment Design

- Date: 2026-07-15
- Scope: `HookBridge` and bootstrap worker lifecycle control
- Status: approved direction; written-spec review pending

## Goal

Queued captures and close requests must retain a unique worker owner and make
bounded progress when control `Event` methods raise `BaseException`. A
clean-sealed bootstrap transport child must dispose later reservations exactly
without restarting the terminal stream or pretending the reservation reached
the daemon.

## Control latches and waits

`_stop_requested` remains the non-throwing authority for stop. HookBridge also
publishes a boolean wake request before touching the best-effort wake event.
Wake-event failure degrades health, but worker start is attempted independently
in a `finally` path. `request_clean_close()` uses the same notification path and
forces a worker-start attempt even when capture-driven starts are disabled.

The worker preserves the configured reconnect delay as an absolute deadline.
Within that deadline it waits in bounded control slices and checks the boolean
stop/wake latches between slices. A persistent `wait`, `clear`, or dual
stop/wake `set` fault therefore cannot create a CPU spin, shorten reconnect
backoff, or hide a stop request for the full configured 60 seconds.

## Stop and ownership proof

Start and test-stop operations are serialized by a lifecycle-operation lock;
the worker never needs that lock to publish exit. Test stop performs these
steps:

1. Publish the boolean stop latch first.
2. Attempt stop-event `set` and wake-event `set` independently, degrading
   health for each failure.
3. Snapshot the current worker owner under the worker lock regardless of those
   failures.
4. Release the worker lock, perform a bounded join, and probe liveness.
5. Clear ownership only when liveness is explicitly false. Join timeout or an
   unknown/throwing liveness probe retains the owner, degrades health, and makes
   stop fail so a later start cannot create a duplicate.

Bootstrap stop applies the same proof to the outer worker and then proves its
owned child stopped. Child ownership may be cleared only when:

- `stop_worker_for_test` exists and is callable;
- that call returns normally; and
- a strict, non-lossy child-liveness probe returns false.

A missing/non-callable stop, a live result, or an exception from the strict
post-stop probe retains child ownership and degrades health. The lossy
`worker_started` health property is not ownership evidence. A successful
bootstrap stop therefore leaves neither a live outer worker nor an owned/live
transport child.

## Terminal child disposition

A clean-sealed HookBridge is immutable and cannot restart. The bootstrap worker
must not reconnect it and must not construct a replacement stream: bootstrap
sequence `N+1` cannot safely become capture `1` for a new `bridge_instance_id`
without a new protocol-level mapping and corresponding `brain.attach` contract.

Instead, bootstrap passes the original reservation and original capture range
to the terminal child's `capture_reserved()` method. Its sealed branch returns
an explicit `late_after_close` disposition after recording an exact receipt,
advancing the local input cursor, and adding late/drop health without a daemon
commit or ACK. Bootstrap then calls a separate identity-guarded
`mark_late_after_close()` operation. It does not call `mark_handed_off()`.

If child accounting succeeds but bootstrap publication fails, retrying the
same reservation is accepted through `_last_bootstrap_reservation` without
incrementing child health twice. Bootstrap finalization is atomic under its
capture lock and counts exactly once. A pending gap moves from the pending-drop
component to the terminal-drop component without increasing total dropped
events; a pending observation adds one terminal drop. Bootstrap health exposes
`trace_complete=false`, the exact `late_after_close` count, no pending record,
and the terminal error.

## Verification

Adversarial tests cover:

- persistent wake `set` failure with an absent worker and with an existing
  worker inside a 60-second configured idle/reconnect wait;
- clean-close notification when wake `set` fails;
- simultaneous persistent stop-event and wake-event `set` failures;
- join and liveness `BaseException`, including retained ownership and duplicate
  prevention;
- bootstrap outer/child cleanup and missing/non-callable child stop;
- a normal child stop followed by an unknown strict liveness probe;
- terminal observation and gap disposition, retry idempotency, exact capture
  sequence, no normal handoff, and no commit/ACK.

The two Hermes test files, race-prone tests in repeated fresh processes, the
full suite, Ruff lint/format, compile checks, and `git diff --check` form the
completion gate.
