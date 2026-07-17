"""Single-owner Hermes consciousness runtime and daemon orchestration."""

from __future__ import annotations

import argparse
import asyncio
import math
import os
import socket
import sys
import threading
import time
import weakref
from collections.abc import Callable, Coroutine
from concurrent.futures import Future
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import TracebackType
from typing import ClassVar, Protocol, Self

from alice_brain_hermes import __version__
from alice_brain_hermes.core.events import EventEnvelope
from alice_brain_hermes.errors import (
    BridgeBindingError,
    BridgeClosedError,
    EnergyWorkerHeartbeatOwnedError,
    LedgerIntegrityError,
    RuntimeOwnedError,
    SchedulerShutdownError,
)
from alice_brain_hermes.ids import new_id, validate_id
from alice_brain_hermes.protocol.energy import (
    EnergyAssessmentChoiceV1,
    EnergyAssessmentLeaseV1,
    EnergyAssessmentProvenanceV1,
)
from alice_brain_hermes.protocol.identity import IdentityChoiceV1, IdentityNamingLeaseV1
from alice_brain_hermes.protocol.models import (
    BrainProfileV1,
    DaemonDiscoveryV2,
    LoopbackEndpointV1,
)
from alice_brain_hermes.protocol.status import (
    ENERGY_WORKER_STALE_AFTER_MS,
    BridgeConnectionSummaryV1,
    DaemonRuntimeStatusV1,
    EnergyWorkerHealthV1,
    EnergyWorkerReportV1,
    RuntimeSchemaVersionsV1,
    SchedulerHealthSummaryV1,
    SemanticEvidenceSummaryV1,
    SnapshotStatusV1,
)
from alice_brain_hermes.runtime.discovery import (
    CredentialFile,
    cleanup_credential,
    cleanup_discovery,
    cleanup_stale_discovery,
    create_credential,
    publish_discovery,
)
from alice_brain_hermes.runtime.engine import ConsciousEngine
from alice_brain_hermes.runtime.lease import RuntimeLease
from alice_brain_hermes.runtime.scheduler import ContinuousScheduler
from alice_brain_hermes.runtime.snapshot import (
    DEFAULT_SNAPSHOT_INTERVAL_EVENTS,
    SnapshotWorker,
    validate_snapshot_interval,
)
from alice_brain_hermes.runtime.store import SQLiteLedger

LedgerFactory = Callable[[Path], SQLiteLedger]
SchedulerFactory = Callable[..., ContinuousScheduler]
_SHUTDOWN_DRAIN_TIMEOUT_SECONDS = 5.0


@dataclass(frozen=True, slots=True)
class _DynamicFoundationReceipt:
    """Immutable compare-and-swap evidence for one unpublished foundation."""

    brain_id: str
    foundation_event_id: str
    foundation_body_fingerprint: str
    foundation_envelope_fingerprint: str
    foundation_json: str
    profile_key: str | None
    profile_fingerprint: str | None
    profile_json: str | None

    @classmethod
    def capture(
        cls,
        brain_id: str,
        foundation: EventEnvelope,
        profile: BrainProfileV1 | None,
    ) -> _DynamicFoundationReceipt:
        foundation = foundation.revalidated()
        if foundation.brain_id != brain_id:
            raise ValueError("dynamic foundation receipt brain does not match")
        return cls(
            brain_id=brain_id,
            foundation_event_id=foundation.event_id,
            foundation_body_fingerprint=foundation.body_fingerprint(),
            foundation_envelope_fingerprint=foundation.envelope_fingerprint(),
            foundation_json=foundation.canonical_json(),
            profile_key=None if profile is None else profile.profile_key,
            profile_fingerprint=None if profile is None else profile.fingerprint(),
            profile_json=None if profile is None else profile.canonical_json(),
        )

    def evidence(self) -> tuple[EventEnvelope, BrainProfileV1 | None]:
        foundation = EventEnvelope.model_validate_json(self.foundation_json)
        if (
            foundation.brain_id != self.brain_id
            or foundation.event_id != self.foundation_event_id
            or foundation.body_fingerprint() != self.foundation_body_fingerprint
            or foundation.envelope_fingerprint() != self.foundation_envelope_fingerprint
            or foundation.canonical_json() != self.foundation_json
        ):
            raise RuntimeError("dynamic foundation receipt integrity failed")
        profile_fields = (
            self.profile_key,
            self.profile_fingerprint,
            self.profile_json,
        )
        if profile_fields == (None, None, None):
            return foundation, None
        if any(value is None for value in profile_fields):
            raise RuntimeError("dynamic profile receipt is incomplete")
        assert self.profile_key is not None
        assert self.profile_fingerprint is not None
        assert self.profile_json is not None
        profile = BrainProfileV1.model_validate_json(self.profile_json)
        if (
            profile.profile_key != self.profile_key
            or profile.fingerprint() != self.profile_fingerprint
            or profile.canonical_json() != self.profile_json
        ):
            raise RuntimeError("dynamic profile receipt integrity failed")
        return foundation, profile


@dataclass(frozen=True, slots=True)
class _EnergyWorkerHeartbeat:
    report: EnergyWorkerReportV1
    received_monotonic: float


def _positive_seconds(value: object, *, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or value <= 0
    ):
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


class _CleanupOwner(Protocol):
    runtime_home: Path

    @property
    def closed(self) -> bool: ...

    def close(self) -> None: ...


class _PreRuntimeOwner:
    """Retain lease authority if startup fails before runtime construction."""

    def __init__(self, runtime_home: Path, lease: RuntimeLease) -> None:
        self.runtime_home = runtime_home
        self.lease = lease
        self.credential: CredentialFile | None = None
        self.ledger: SQLiteLedger | None = None
        self._sqlite_closed = False
        self._credential_cleaned = False
        self._lease_released = False
        self._close_lock = threading.Lock()
        self._closed = False

    @property
    def closed(self) -> bool:
        self.lease.assert_creator_process()
        with self._close_lock:
            return self._closed

    def adopt_ledger(self, ledger: SQLiteLedger) -> None:
        self.lease.assert_creator_process()
        with self._close_lock:
            if self._closed or self._lease_released:
                raise RuntimeError("SQLite startup owner is already closed")
            if self.ledger is not None:
                raise RuntimeError("SQLite ledger startup owner was already adopted")
            self.ledger = ledger

    def quarantine_unadopted_ledger(self, ledger: SQLiteLedger) -> None:
        """Backstop an exact factory ledger if normal adoption is interrupted."""
        self.lease.assert_creator_process()
        with self._close_lock:
            if self._closed or self._lease_released:
                raise RuntimeError("SQLite startup owner is already closed")
            if self.ledger is None:
                self.ledger = ledger
            elif self.ledger is not ledger:
                raise RuntimeError("a different SQLite ledger is already retained")

    def transfer_to_runtime(self, ledger: SQLiteLedger) -> None:
        self.lease.assert_creator_process()
        with self._close_lock:
            if self.ledger is not ledger:
                raise RuntimeError("SQLite runtime transfer does not match its owner")
            self.ledger = None
            self._sqlite_closed = True

    def close(self) -> None:
        self.lease.assert_creator_process()
        with self._close_lock:
            self._close_serialized()

    def _close_serialized(self) -> None:
        if self._closed:
            return
        if not self._sqlite_closed:
            if self.ledger is not None:
                try:
                    self.ledger.close()
                except BaseException:
                    if self.ledger.closed:
                        self._sqlite_closed = True
                    raise
            self._sqlite_closed = True
        self.lease.close_registered_resources()
        if self.credential is not None and not self._credential_cleaned:
            cleanup_credential(self.lease, self.credential)
            self._credential_cleaned = True
        if not self._lease_released:
            try:
                self.lease.release()
            except BaseException:
                if self.lease.released:
                    self._lease_released = True
                    self._closed = True
                raise
            else:
                self._lease_released = True
        self._closed = True


class HermesDaemonRuntime:
    """Own exactly one lease, ledger, engine and scheduler per persisted brain."""

    _failed_owner_lock: ClassVar[threading.Lock] = threading.Lock()
    _failed_owners: ClassVar[dict[Path, _CleanupOwner]] = {}

    def __init__(
        self,
        runtime_home: Path,
        lease: RuntimeLease,
        credential: CredentialFile,
        ledger: SQLiteLedger,
        *,
        scheduler_interval_seconds: float,
        scheduler_factory: SchedulerFactory,
        snapshot_interval_events: int,
    ) -> None:
        self.runtime_home = runtime_home
        self.lease = lease
        self.credential = credential
        self.ledger = ledger
        self._creator_pid = os.getpid()
        self.scheduler_interval_seconds = scheduler_interval_seconds
        self._scheduler_factory = scheduler_factory
        self._engine_creation_gate = threading.Lock()
        self._registry_lock = threading.Lock()
        self._lifecycle = threading.Condition(self._registry_lock)
        self._shutdown_lock = threading.Lock()
        self._cells: dict[str, Future[ConsciousEngine]] = {}
        self._pending_foundations: dict[str, _DynamicFoundationReceipt] = {}
        self._transport_cleanup_owner: weakref.ReferenceType[_CleanupOwner] | None = (
            None
        )
        self._engines: dict[str, ConsciousEngine] = {}
        self._schedulers: dict[str, ContinuousScheduler] = {}
        self._active_operations = 0
        self._daemon_serving_required = False
        self._daemon_serving_active = True
        self._closing = False
        self._fatal_error: BaseException | None = None
        self._energy_worker_heartbeat: _EnergyWorkerHeartbeat | None = None
        self._energy_worker_monotonic: Callable[[], float] = time.monotonic
        self._closed = False
        self._stopped_scheduler_ids: set[str] = set()
        self._schedulers_stopped = False
        self._ledger_closed = False
        self._discovery_cleaned = False
        self._credential_cleaned = False
        self._lease_released = False
        self.snapshot_worker = SnapshotWorker(
            ledger,
            interval_events=snapshot_interval_events,
            fatal_error_sink=self._mark_fail_stopped,
        )
        self._snapshot_worker_stopped = False

    @classmethod
    def open(
        cls,
        runtime_home: str | Path,
        *,
        ledger_factory: LedgerFactory | None = None,
        scheduler_factory: SchedulerFactory = ContinuousScheduler,
        scheduler_interval_seconds: float = 1.0,
        snapshot_interval_events: int = DEFAULT_SNAPSHOT_INTERVAL_EVENTS,
        launch_nonce: str | None = None,
    ) -> Self:
        """Acquire the process lease before constructing any SQLite handle."""
        scheduler_interval_seconds = _positive_seconds(
            scheduler_interval_seconds,
            name="scheduler_interval_seconds",
        )
        snapshot_interval_events = validate_snapshot_interval(
            snapshot_interval_events
        )
        home = Path(runtime_home).expanduser().absolute()
        lease = RuntimeLease.acquire(home, launch_nonce=launch_nonce)
        startup_owner = _PreRuntimeOwner(home, lease)
        credential: CredentialFile | None = None
        ledger: SQLiteLedger | None = None
        runtime: Self | None = None
        try:
            cleanup_stale_discovery(lease)
            lease.assert_authority()
            credential = create_credential(lease)
            startup_owner.credential = credential
            lease.assert_authority()
            database = lease.home_path("runtime.db")
            database = SQLiteLedger._validate_runtime_paths(database, lease)
            if ledger_factory is None:
                ledger = SQLiteLedger.open(
                    database,
                    authority=lease,
                    owner_sink=startup_owner.adopt_ledger,
                )
            else:
                ledger = ledger_factory(database)
                if not isinstance(ledger, SQLiteLedger):
                    raise TypeError("ledger_factory must return SQLiteLedger")
                try:
                    startup_owner.adopt_ledger(ledger)
                except BaseException:
                    startup_owner.quarantine_unadopted_ledger(ledger)
                    raise
                ledger._adopt_runtime_authority(lease)
            runtime = cls(
                home,
                lease,
                credential,
                ledger,
                scheduler_interval_seconds=scheduler_interval_seconds,
                scheduler_factory=scheduler_factory,
                snapshot_interval_events=snapshot_interval_events,
            )
            startup_owner.transfer_to_runtime(ledger)
            lease.assert_authority()
            for brain_id in ledger.list_brain_ids():
                runtime.engine(brain_id)
            # Startup replay and scheduler construction can be comparatively
            # expensive.  Reset prior-process streams only after those steps so
            # the first maintenance pass grants a full, fresh restart grace.
            ledger.recover_stale_bridge_connections()
            runtime.snapshot_worker.start()
            return runtime
        except BaseException as primary_error:
            traceback = primary_error.__traceback__
            owner: _CleanupOwner = runtime or startup_owner
            cleanup_error: BaseException | None = None
            try:
                owner.close()
            except BaseException as error:
                cleanup_error = error
                owner_closed = (
                    runtime.closed if runtime is not None else startup_owner.closed
                )
                if not owner_closed:
                    cls._retain_failed_owner(owner)
            if cleanup_error is not None:
                if runtime is None:
                    raise primary_error.with_traceback(traceback) from cleanup_error
                raise cleanup_error from primary_error
            raise

    @classmethod
    def _retain_failed_owner(cls, runtime: _CleanupOwner) -> None:
        with cls._failed_owner_lock:
            existing = cls._failed_owners.get(runtime.runtime_home)
            if existing is not None and existing is not runtime:
                # Promote one exact runtime owner to its transport wrapper so
                # cleanup can never release SQLite while that wrapper still
                # has unquiesced handlers or sockets.
                if getattr(runtime, "runtime", None) is existing:
                    cls._failed_owners[runtime.runtime_home] = runtime
                    return
                # A transport wrapper is already the strong owner of its exact
                # runtime. Keep that outer recovery seam and preserve the
                # runtime's original cleanup exception.
                if getattr(existing, "runtime", None) is runtime:
                    return
                raise RuntimeError("a different failed runtime owner is retained")
            cls._failed_owners[runtime.runtime_home] = runtime

    @classmethod
    def _claim_failed_owner(cls, owner: _CleanupOwner) -> bool:
        """Remove only this owner before it can release native authority."""
        with cls._failed_owner_lock:
            if cls._failed_owners.get(owner.runtime_home) is not owner:
                return False
            del cls._failed_owners[owner.runtime_home]
            return True

    @staticmethod
    def _cleanup_owner_closed(owner: _CleanupOwner) -> bool:
        try:
            if isinstance(owner, PrivateDaemonServer):
                return owner._transport_quiesced and owner.runtime.closed
            return owner.closed
        except BaseException:
            return False

    @classmethod
    def _restore_claimed_failed_owner(cls, owner: _CleanupOwner) -> None:
        """Restore an unfinished claim without overwriting a newer owner."""
        if cls._cleanup_owner_closed(owner):
            return
        with cls._failed_owner_lock:
            existing = cls._failed_owners.get(owner.runtime_home)
            if existing is None:
                cls._failed_owners[owner.runtime_home] = owner

    @classmethod
    def _discard_exact_closed_owner(cls, owner: _CleanupOwner) -> bool:
        """Discard only a reinserted exact owner after terminal cleanup proof."""
        if not cls._cleanup_owner_closed(owner):
            return False
        with cls._failed_owner_lock:
            if cls._failed_owners.get(owner.runtime_home) is not owner:
                return False
            del cls._failed_owners[owner.runtime_home]
            return True

    @classmethod
    def retry_failed_cleanup(cls, runtime_home: str | Path) -> bool:
        """Retry one fail-stop owner cleanup while retaining strong ownership."""
        home = Path(runtime_home).expanduser().absolute()
        with cls._failed_owner_lock:
            runtime = cls._failed_owners.pop(home, None)
        if runtime is None:
            return False
        try:
            runtime.close()
        except BaseException:
            cls._restore_claimed_failed_owner(runtime)
            raise
        cls._discard_exact_closed_owner(runtime)
        return True

    @classmethod
    async def retry_failed_cleanup_async(cls, runtime_home: str | Path) -> bool:
        """Drive async transport recovery before releasing a retained owner."""

        home = Path(runtime_home).expanduser().absolute()
        with cls._failed_owner_lock:
            owner = cls._failed_owners.pop(home, None)
        if owner is None:
            return False

        async def cleanup_claimed_owner() -> None:
            if isinstance(owner, PrivateDaemonServer):
                failure = await owner._cleanup_after_run()
                if failure is not None:
                    raise failure
            else:
                await asyncio.to_thread(owner.close)

        cleanup = asyncio.create_task(
            cleanup_claimed_owner(),
            name="alice-brain-hermes-failed-owner-retry",
        )
        cancellation: asyncio.CancelledError | None = None
        cleanup_error: BaseException | None = None
        while True:
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError as error:
                if cleanup.done():
                    if cleanup.cancelled():
                        cleanup_error = error
                    else:
                        try:
                            cleanup.result()
                        except BaseException as task_error:
                            cleanup_error = task_error
                        else:
                            cancellation = error
                    break
                if cancellation is None:
                    cancellation = error
                continue
            except BaseException as error:
                cleanup_error = error
            break
        if cleanup_error is not None:
            cls._restore_claimed_failed_owner(owner)
        if cancellation is not None:
            if cleanup_error is not None and cancellation.__cause__ is None:
                cancellation.__cause__ = cleanup_error
            raise cancellation
        if cleanup_error is not None:
            raise cleanup_error
        cls._discard_exact_closed_owner(owner)
        return True

    @classmethod
    def failed_owner_count(cls) -> int:
        with cls._failed_owner_lock:
            return len(cls._failed_owners)

    def _assert_creator_process(self) -> None:
        if os.getpid() != self._creator_pid:
            raise PermissionError("daemon runtime belongs to another process")

    def _begin_operation(self) -> None:
        self._assert_creator_process()
        self.lease.assert_authority()
        with self._lifecycle:
            if self._closed or self._closing:
                raise RuntimeError("daemon runtime is closing or closed")
            self._active_operations += 1

    def _end_operation(self) -> None:
        with self._lifecycle:
            self._active_operations -= 1
            self._lifecycle.notify_all()

    def engine(self, brain_id: str) -> ConsciousEngine:
        """Return the once-created engine without holding registry locks on wait."""
        brain_id = validate_id(brain_id)
        self._begin_operation()
        try:
            return self._engine_once(brain_id)
        finally:
            self._end_operation()

    def _mark_fail_stopped(self, error: BaseException) -> None:
        """Reject new work while the daemon still owns cleanup authority."""
        with self._lifecycle:
            if self._fatal_error is None:
                self._fatal_error = error
            self._closing = True
            self._lifecycle.notify_all()
        type(self)._retain_failed_owner(self._fail_stop_cleanup_owner())

    def _install_transport_cleanup_owner(self, owner: _CleanupOwner) -> None:
        """Register the exact outer owner before it can expose transport state."""
        if getattr(owner, "runtime", None) is not self:
            raise ValueError("transport cleanup owner does not wrap this runtime")
        with self._registry_lock:
            existing = (
                None
                if self._transport_cleanup_owner is None
                else self._transport_cleanup_owner()
            )
            if existing is not None and existing is not owner:
                raise RuntimeError("a different transport cleanup owner is installed")
            self._transport_cleanup_owner = weakref.ref(owner)
        with type(self)._failed_owner_lock:
            if type(self)._failed_owners.get(self.runtime_home) is self:
                type(self)._failed_owners[self.runtime_home] = owner

    def _fail_stop_cleanup_owner(self) -> _CleanupOwner:
        with self._registry_lock:
            owner = (
                None
                if self._transport_cleanup_owner is None
                else self._transport_cleanup_owner()
            )
        return self if owner is None else owner

    @property
    def fail_stopped(self) -> bool:
        self._assert_creator_process()
        with self._registry_lock:
            return self._fatal_error is not None

    @property
    def fail_stop_error(self) -> BaseException | None:
        self._assert_creator_process()
        with self._registry_lock:
            return self._fatal_error

    def _claim_engine_cell(self, brain_id: str) -> tuple[Future[ConsciousEngine], bool]:
        # The common existing-cell path never waits for the global creation
        # gate. Missing claims serialize gate -> registry so a dynamic ledger
        # transaction can never publish a brain without an owning once-cell.
        with self._registry_lock:
            cell = self._cells.get(brain_id)
        if cell is not None:
            return cell, False
        with self._engine_creation_gate, self._registry_lock:
            cell = self._cells.get(brain_id)
            if cell is not None:
                return cell, False
            cell = Future()
            self._cells[brain_id] = cell
            return cell, True

    def _engine_once(
        self,
        brain_id: str,
        *,
        owned_cell: Future[ConsciousEngine] | None = None,
        pending_receipt: _DynamicFoundationReceipt | None = None,
    ) -> ConsciousEngine:
        if owned_cell is None:
            cell, creator = self._claim_engine_cell(brain_id)
        else:
            with self._registry_lock:
                if self._cells.get(brain_id) is not owned_cell:
                    raise RuntimeError("dynamic engine once-cell ownership was lost")
                if self._pending_foundations.get(brain_id) is not pending_receipt:
                    raise RuntimeError("dynamic foundation receipt ownership was lost")
            cell = owned_cell
            creator = True
        if not creator:
            return cell.result()

        scheduler: ContinuousScheduler | None = None
        try:
            if brain_id not in self.ledger.list_brain_ids():
                raise KeyError(brain_id)
            engine = ConsciousEngine(
                self.ledger,
                brain_id,
                actor_id=brain_id,
                on_state_published=self.snapshot_worker.notify,
            )
            scheduler = self._scheduler_factory(
                engine,
                interval_seconds=self.scheduler_interval_seconds,
            )
            scheduler.start()
            with self._registry_lock:
                if self._fatal_error is not None:
                    raise RuntimeError(
                        "daemon entered fail-stop during engine creation"
                    ) from self._fatal_error
                if owned_cell is not None:
                    if self._pending_foundations.get(brain_id) is not pending_receipt:
                        raise RuntimeError(
                            "dynamic foundation receipt changed before publication"
                        )
                    del self._pending_foundations[brain_id]
                self._engines[brain_id] = engine
                self._schedulers[brain_id] = scheduler
            self.snapshot_worker.register(
                engine,
                known_snapshot_sequence=0 if owned_cell is not None else None,
            )
            cell.set_result(engine)
            return engine
        except BaseException as error:
            cleanup_error: BaseException | None = None
            if scheduler is not None:
                try:
                    scheduler.stop()
                except BaseException as stop_error:
                    cleanup_error = stop_error
                    with self._registry_lock:
                        self._schedulers[brain_id] = scheduler
                    try:
                        self._mark_fail_stopped(stop_error)
                    except BaseException as retention_error:
                        cleanup_error = retention_error
            failure = cleanup_error or error
            if not cell.done():
                cell.set_exception(failure)
            with self._registry_lock:
                removable = owned_cell is None and cleanup_error is None
                if removable and self._cells.get(brain_id) is cell:
                    del self._cells[brain_id]
            if cleanup_error is not None:
                raise cleanup_error from error
            raise

    def _reserve_dynamic_cell(
        self,
    ) -> tuple[str, Future[ConsciousEngine]]:
        while True:
            brain_id = new_id()
            with self._registry_lock:
                if brain_id in self._cells:
                    continue
                cell: Future[ConsciousEngine] = Future()
                self._cells[brain_id] = cell
                return brain_id, cell

    def _finish_failed_dynamic_persistence(
        self,
        brain_id: str,
        cell: Future[ConsciousEngine],
        error: BaseException,
    ) -> BaseException | None:
        """Clear a reservation only after proving its transaction left no brain."""
        audit_error: BaseException | None = None
        try:
            absent = brain_id not in self.ledger.list_brain_ids()
        except BaseException as failure:
            absent = False
            audit_error = failure
        if not cell.done():
            cell.set_exception(error)
        with self._registry_lock:
            owned = self._cells.get(brain_id) is cell
            if absent and owned and brain_id not in self._pending_foundations:
                del self._cells[brain_id]
                return audit_error
        self._mark_fail_stopped(audit_error or error)
        return audit_error

    def _discard_unused_dynamic_cell(
        self, brain_id: str, cell: Future[ConsciousEngine]
    ) -> None:
        with self._registry_lock:
            if (
                self._cells.get(brain_id) is not cell
                or brain_id in self._pending_foundations
            ):
                raise RuntimeError("unused dynamic once-cell ownership was lost")
            del self._cells[brain_id]
        cell.cancel()

    def _start_dynamic_foundation(
        self,
        receipt: _DynamicFoundationReceipt,
        cell: Future[ConsciousEngine],
    ) -> ConsciousEngine:
        try:
            return self._engine_once(
                receipt.brain_id,
                owned_cell=cell,
                pending_receipt=receipt,
            )
        except BaseException as start_error:
            traceback = start_error.__traceback__
            if self.fail_stopped:
                raise
            compensation_error: BaseException | None = None
            try:
                foundation, profile = receipt.evidence()
                compensated = self.ledger.compensate_unpublished_brain_foundation(
                    receipt.brain_id,
                    foundation=foundation,
                    profile=profile,
                )
            except BaseException as error:
                compensated = False
                compensation_error = error

            cleared = False
            if compensated:
                with self._registry_lock:
                    exact_pending = (
                        self._pending_foundations.get(receipt.brain_id) is receipt
                    )
                    exact_cell = self._cells.get(receipt.brain_id) is cell
                    unpublished = (
                        receipt.brain_id not in self._engines
                        and receipt.brain_id not in self._schedulers
                    )
                    if exact_pending and exact_cell and unpublished:
                        del self._pending_foundations[receipt.brain_id]
                        del self._cells[receipt.brain_id]
                        cleared = True
            if not compensated or not cleared:
                fatal = compensation_error or RuntimeError(
                    "dynamic foundation compensation could not be proven"
                )
                self._mark_fail_stopped(fatal)
            if compensation_error is not None:
                raise start_error.with_traceback(traceback) from compensation_error
            raise

    def scheduler(self, brain_id: str) -> ContinuousScheduler:
        brain_id = validate_id(brain_id)
        self._begin_operation()
        try:
            self._engine_once(brain_id)
            with self._registry_lock:
                return self._schedulers[brain_id]
        finally:
            self._end_operation()

    def create_brain(self, *, name: str | None) -> ConsciousEngine:
        """Create one server-owned brain and persist its foundation event."""
        self._begin_operation()
        try:
            with self._engine_creation_gate:
                brain_id, cell = self._reserve_dynamic_cell()
                try:
                    foundation = self.ledger.create_brain_foundation(
                        brain_id, name=name
                    )
                    receipt = _DynamicFoundationReceipt.capture(
                        brain_id, foundation, None
                    )
                    with self._registry_lock:
                        if self._cells.get(brain_id) is not cell:
                            raise RuntimeError(
                                "dynamic brain once-cell ownership was lost"
                            )
                        self._pending_foundations[brain_id] = receipt
                except BaseException as error:
                    traceback = error.__traceback__
                    audit_error = self._finish_failed_dynamic_persistence(
                        brain_id, cell, error
                    )
                    if audit_error is not None:
                        raise error.with_traceback(traceback) from audit_error
                    raise
            return self._start_dynamic_foundation(receipt, cell)
        finally:
            self._end_operation()

    def resolve_brain(self, profile: BrainProfileV1) -> tuple[ConsciousEngine, bool]:
        """Resolve one persisted stable profile without client brain identity."""
        self._begin_operation()
        try:
            with self._engine_creation_gate:
                candidate_id, candidate_cell = self._reserve_dynamic_cell()
                try:
                    resolved = self.ledger.resolve_brain_profile(
                        profile, new_brain_id=candidate_id
                    )
                    if not resolved.created:
                        self._discard_unused_dynamic_cell(candidate_id, candidate_cell)
                        receipt = None
                    else:
                        if (
                            resolved.brain_id != candidate_id
                            or resolved.foundation is None
                        ):
                            raise RuntimeError(
                                "new stable profile did not return its exact foundation"
                            )
                        receipt = _DynamicFoundationReceipt.capture(
                            candidate_id, resolved.foundation, profile
                        )
                        with self._registry_lock:
                            if self._cells.get(candidate_id) is not candidate_cell:
                                raise RuntimeError(
                                    "dynamic profile once-cell ownership was lost"
                                )
                            self._pending_foundations[candidate_id] = receipt
                except BaseException as error:
                    traceback = error.__traceback__
                    audit_error = self._finish_failed_dynamic_persistence(
                        candidate_id, candidate_cell, error
                    )
                    if audit_error is not None:
                        raise error.with_traceback(traceback) from audit_error
                    raise
            if receipt is None:
                return self._engine_once(resolved.brain_id), False
            return self._start_dynamic_foundation(receipt, candidate_cell), True
        finally:
            self._end_operation()

    def claim_identity_naming(self, brain_id: str) -> IdentityNamingLeaseV1 | None:
        """Claim through the brain's single authoritative engine writer."""

        brain_id = validate_id(brain_id)
        self._begin_operation()
        try:
            return self._engine_once(brain_id).claim_identity_naming()
        finally:
            self._end_operation()

    def complete_identity_naming(
        self,
        lease_id: str,
        choice: IdentityChoiceV1,
    ) -> str:
        """Route a lease completion without trusting client-owned brain identity."""

        lease_id = validate_id(lease_id)
        self._begin_operation()
        try:
            brain_id = self.ledger.identity_naming_brain_id(lease_id)
            return self._engine_once(brain_id).complete_identity_naming(
                lease_id,
                choice,
            )
        finally:
            self._end_operation()

    def fail_identity_naming(self, lease_id: str, failure_code: str) -> str:
        """Route one sanitized naming failure through its durable lease."""

        lease_id = validate_id(lease_id)
        self._begin_operation()
        try:
            brain_id = self.ledger.identity_naming_brain_id(lease_id)
            return self._engine_once(brain_id).fail_identity_naming(
                lease_id,
                failure_code,
            )
        finally:
            self._end_operation()

    def claim_energy_assessment(
        self,
        brain_id: str,
    ) -> EnergyAssessmentLeaseV1 | None:
        """Claim through the brain's single authoritative engine writer."""

        brain_id = validate_id(brain_id)
        self._begin_operation()
        try:
            return self._engine_once(brain_id).claim_energy_assessment()
        finally:
            self._end_operation()

    def complete_energy_assessment(
        self,
        lease_id: str,
        choice: EnergyAssessmentChoiceV1,
        provenance: EnergyAssessmentProvenanceV1,
    ) -> str:
        """Route host evidence using only the lease's durable brain identity."""

        lease_id = validate_id(lease_id)
        self._begin_operation()
        try:
            brain_id = self.ledger.energy_assessment_brain_id(lease_id)
            return self._engine_once(brain_id).complete_energy_assessment(
                lease_id,
                choice,
                provenance,
            )
        finally:
            self._end_operation()

    def fail_energy_assessment(self, lease_id: str, failure_code: str) -> str:
        """Route one sanitized host failure through its durable energy lease."""

        lease_id = validate_id(lease_id)
        self._begin_operation()
        try:
            brain_id = self.ledger.energy_assessment_brain_id(lease_id)
            return self._engine_once(brain_id).fail_energy_assessment(
                lease_id,
                failure_code,
            )
        finally:
            self._end_operation()

    def report_energy_worker(self, report: EnergyWorkerReportV1) -> bool:
        """Accept one strict heartbeat while preventing fresh-owner replacement."""

        if type(report) is not EnergyWorkerReportV1:
            raise TypeError("report must be an exact EnergyWorkerReportV1")
        self._begin_operation()
        try:
            now = self._energy_worker_monotonic()
            if (
                isinstance(now, bool)
                or not isinstance(now, (int, float))
                or not math.isfinite(float(now))
                or now < 0
            ):
                raise RuntimeError("energy worker monotonic clock is invalid")
            now = float(now)
            with self._registry_lock:
                current = self._energy_worker_heartbeat
                if current is not None:
                    if current.report.reporter_id == report.reporter_id:
                        if report.report_sequence <= current.report.report_sequence:
                            raise ValueError(
                                "energy worker report sequence must increase"
                            )
                    elif (
                        now - current.received_monotonic
                        < ENERGY_WORKER_STALE_AFTER_MS / 1_000
                    ):
                        raise EnergyWorkerHeartbeatOwnedError(
                            "energy worker heartbeat belongs to a fresh reporter"
                        )
                self._energy_worker_heartbeat = _EnergyWorkerHeartbeat(report, now)
            return True
        finally:
            self._end_operation()

    def _energy_worker_health_locked(self) -> EnergyWorkerHealthV1:
        heartbeat = self._energy_worker_heartbeat
        if heartbeat is None:
            return EnergyWorkerHealthV1.unreported(
                stale_after_ms=ENERGY_WORKER_STALE_AFTER_MS
            )
        now = self._energy_worker_monotonic()
        if (
            isinstance(now, bool)
            or not isinstance(now, (int, float))
            or not math.isfinite(float(now))
            or now < heartbeat.received_monotonic
        ):
            raise RuntimeError("energy worker monotonic clock is invalid")
        elapsed = float(now) - heartbeat.received_monotonic
        age_ms = min(int(elapsed * 1_000), 2**63 - 1)
        if elapsed >= ENERGY_WORKER_STALE_AFTER_MS / 1_000:
            age_ms = max(age_ms, ENERGY_WORKER_STALE_AFTER_MS)
        report = heartbeat.report
        if age_ms >= ENERGY_WORKER_STALE_AFTER_MS:
            status = "stale"
        elif (
            not report.worker_started
            or report.terminal_intent_pending
            or report.last_error_type is not None
        ):
            status = "degraded"
        else:
            status = "healthy"
        return EnergyWorkerHealthV1(
            status=status,
            worker_started=report.worker_started,
            terminal_intent_pending=report.terminal_intent_pending,
            last_error_type=report.last_error_type,
            reporter_id=report.reporter_id,
            report_sequence=report.report_sequence,
            last_report_age_ms=age_ms,
            stale_after_ms=ENERGY_WORKER_STALE_AFTER_MS,
        )

    @property
    def brain_ids(self) -> tuple[str, ...]:
        self._begin_operation()
        try:
            return tuple(self.ledger.list_brain_ids())
        finally:
            self._end_operation()

    @property
    def engine_count(self) -> int:
        self._assert_creator_process()
        with self._registry_lock:
            return len(self._engines)

    @property
    def scheduler_count(self) -> int:
        self._assert_creator_process()
        with self._registry_lock:
            return len(self._schedulers)

    def _require_daemon_serving_boundary(self) -> None:
        self._assert_creator_process()
        with self._registry_lock:
            self._daemon_serving_required = True
            self._daemon_serving_active = False

    def _enter_daemon_serving_boundary(self) -> None:
        self._assert_creator_process()
        with self._registry_lock:
            if not self._daemon_serving_required:
                raise RuntimeError("daemon serving boundary was not required")
            self._daemon_serving_active = True

    def _leave_daemon_serving_boundary(self) -> None:
        self._assert_creator_process()
        with self._registry_lock:
            self._daemon_serving_active = False

    @property
    def daemon_serving_active(self) -> bool:
        self._assert_creator_process()
        with self._registry_lock:
            return (
                not self._daemon_serving_required
                or self._daemon_serving_active
            )

    def readiness_snapshot(self) -> dict[str, object]:
        """Return one exact proof of continuous-writer readiness."""
        self._begin_operation()
        try:
            persisted = set(self.ledger.list_brain_ids())
            with self._registry_lock:
                engines = dict(self._engines)
                schedulers = dict(self._schedulers)
                serving = (
                    not self._daemon_serving_required
                    or self._daemon_serving_active
                )
            running = {
                brain_id
                for brain_id, scheduler in schedulers.items()
                if scheduler.health.running
            }
            degraded = {
                brain_id
                for brain_id, scheduler in schedulers.items()
                if scheduler.health.status != "healthy"
            }
            ready = (
                set(engines) == persisted
                and set(schedulers) == persisted
                and running == persisted
                and serving
            )
            return {
                "runtime_ready": ready,
                "brain_count": len(persisted),
                "engine_count": len(engines),
                "scheduler_count": len(schedulers),
                "running_scheduler_count": len(running),
                "degraded_brain_count": len(degraded),
            }
        finally:
            self._end_operation()

    def status_snapshot(self) -> DaemonRuntimeStatusV1:
        """Return one bounded status projection from persisted evidence."""

        self._begin_operation()
        try:
            # Use the same creation lock order as dynamic persistence. A status
            # sample may truthfully report a transient degraded registry, but it
            # can never pair a newer registry with an older persisted ledger.
            with self._engine_creation_gate:
                persisted = tuple(sorted(self.ledger.list_brain_ids()))
                observability = self.ledger.observability_snapshot()
                sqlite_schema_version = self.ledger.schema_version
                with self._registry_lock:
                    engines = dict(self._engines)
                    schedulers = dict(self._schedulers)
                    serving = (
                        not self._daemon_serving_required
                        or self._daemon_serving_active
                    )
                    fail_stopped = self._fatal_error is not None
                    energy_worker_health = self._energy_worker_health_locked()
            if observability.brain_count != len(persisted):
                raise LedgerIntegrityError(
                    "persisted observability coverage does not match runtime brains"
                )

            scheduler_health = {
                brain_id: scheduler.health for brain_id, scheduler in schedulers.items()
            }
            running = sum(health.running for health in scheduler_health.values())
            degraded = sum(
                health.status != "healthy" for health in scheduler_health.values()
            )
            writers_ready = (
                set(engines) == set(persisted)
                and set(schedulers) == set(persisted)
                and running == len(persisted)
            )
            healthy = not fail_stopped and writers_ready and degraded == 0
            scheduler_summary = SchedulerHealthSummaryV1(
                status="healthy" if healthy else "degraded",
                fail_stopped=fail_stopped,
                brain_count=len(persisted),
                engine_count=len(engines),
                scheduler_count=len(schedulers),
                running_scheduler_count=running,
                degraded_brain_count=degraded,
            )

            if observability.disconnected_open_bridges:
                bridge_state = "degraded"
            elif observability.connected_open_bridges:
                bridge_state = "connected"
            elif observability.total_bridges:
                bridge_state = "idle"
            else:
                bridge_state = "never_connected"
            bridge_summary = BridgeConnectionSummaryV1(
                state=bridge_state,
                total_bridges=observability.total_bridges,
                connected_open_bridges=observability.connected_open_bridges,
                disconnected_open_bridges=observability.disconnected_open_bridges,
                clean_closed_bridges=observability.clean_closed_bridges,
                abandoned_bridges=observability.abandoned_bridges,
            )
            return DaemonRuntimeStatusV1(
                brain_ids=persisted,
                engine_count=len(engines),
                scheduler_count=len(schedulers),
                runtime_ready=writers_ready and serving,
                scheduler_health=scheduler_summary,
                bridge_connection=bridge_summary,
                trace_complete=observability.trace_complete,
                semantic_complete=observability.semantic_complete,
                dropped_events=observability.dropped_events,
                semantic_evidence=SemanticEvidenceSummaryV1(
                    semantic_records=observability.semantic_records,
                    legacy_raw_only_records=observability.legacy_raw_only_records,
                    semantic_gap_records=observability.semantic_gap_records,
                ),
                energy_worker_health=energy_worker_health,
                schema_versions=RuntimeSchemaVersionsV1(
                    semantic=observability.semantic_schema_version,
                    sqlite=sqlite_schema_version,
                ),
            )
        finally:
            self._end_operation()

    def snapshot_status(self) -> SnapshotStatusV1:
        """Return persisted snapshot coverage and volatile worker health."""

        self._begin_operation()
        try:
            persisted = self.ledger.snapshot_observability()
            worker = self.snapshot_worker.health
            return SnapshotStatusV1(
                status=worker.status,
                worker_running=worker.running,
                interval_events=self.snapshot_worker.interval_events,
                pending_brain_count=worker.pending_brain_count,
                snapshot_count=persisted.snapshot_count,
                latest_sequence=persisted.latest_sequence,
                last_error_type=worker.last_error_type,
            )
        finally:
            self._end_operation()

    @property
    def closed(self) -> bool:
        self._assert_creator_process()
        return self._closed

    def _stop_all_schedulers(self) -> None:
        with self._registry_lock:
            schedulers = tuple(self._schedulers.items())
        first_error: BaseException | None = None
        for brain_id, scheduler in schedulers:
            if brain_id in self._stopped_scheduler_ids:
                continue
            try:
                scheduler.stop()
            except BaseException as error:
                if first_error is None:
                    first_error = error
            else:
                self._stopped_scheduler_ids.add(brain_id)
        if first_error is not None:
            raise first_error

    def close(self) -> None:
        """Stop and join every writer before closing SQLite and releasing lease."""
        self._assert_creator_process()
        # A previous failure may have registered this exact runtime. Remove the
        # stale handoff entry before native lease release makes room for a new
        # runtime; failures below re-register while authority is still held.
        type(self)._claim_failed_owner(self)
        terminal_release_error: BaseException | None = None
        with self._shutdown_lock:
            with self._lifecycle:
                if self._closed:
                    return
                self._closing = True
                while self._active_operations:
                    self._lifecycle.wait()
            try:
                if not self._schedulers_stopped:
                    self._stop_all_schedulers()
                    self._schedulers_stopped = True
                if not self._snapshot_worker_stopped:
                    self.snapshot_worker.stop()
                    self._snapshot_worker_stopped = True
                if not self._ledger_closed:
                    self.ledger.close()
                    self._ledger_closed = True
                if not self._credential_cleaned:
                    cleanup_credential(self.lease, self.credential)
                    self._credential_cleaned = True
                if not self._discovery_cleaned:
                    cleanup_discovery(self.lease)
                    self._discovery_cleaned = True
                if not self._lease_released:
                    try:
                        self.lease.release()
                    except BaseException as error:
                        if not self.lease.released:
                            raise
                        self._lease_released = True
                        terminal_release_error = error
                    else:
                        self._lease_released = True
            except BaseException:
                self._retain_failed_owner(self)
                raise
            with self._lifecycle:
                self._closed = True
                self._lifecycle.notify_all()
            type(self)._discard_exact_closed_owner(self)
        if terminal_release_error is not None:
            raise terminal_release_error

    def __enter__(self) -> Self:
        self._begin_operation()
        self._end_operation()
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        self.close()


__all__ = ["HermesDaemonRuntime"]


def _ipv4_loopback_endpoint(value: object) -> bool:
    if not isinstance(value, tuple) or len(value) < 2:
        return False
    host, port = value[:2]
    return (
        host == "127.0.0.1"
        and isinstance(port, int)
        and not isinstance(port, bool)
        and 1 <= port <= 65_535
    )


def _is_private_ipv4_stream(writer: asyncio.StreamWriter) -> bool:
    try:
        accepted_socket = writer.get_extra_info("socket")
        return (
            accepted_socket is not None
            and accepted_socket.family == socket.AF_INET
            and _ipv4_loopback_endpoint(writer.get_extra_info("sockname"))
            and _ipv4_loopback_endpoint(writer.get_extra_info("peername"))
        )
    except (AttributeError, OSError, TypeError, ValueError):
        return False


class PrivateDaemonServer:
    """Async loopback transport around independent per-connection services."""

    def __init__(
        self,
        runtime: HermesDaemonRuntime,
        *,
        abandonment_grace_seconds: float = 30.0,
    ) -> None:
        if not isinstance(runtime, HermesDaemonRuntime):
            raise TypeError("runtime must be a HermesDaemonRuntime instance")

        from alice_brain_hermes.protocol.service import ProtocolService

        abandonment_grace_seconds = _positive_seconds(
            abandonment_grace_seconds,
            name="abandonment_grace_seconds",
        )
        self.runtime = runtime
        self.abandonment_grace_seconds = float(abandonment_grace_seconds)
        self.service = ProtocolService(
            runtime,
            credential=runtime.credential.token,
            instance_nonce=runtime.lease.instance_nonce,
        )
        self._server: asyncio.AbstractServer | None = None
        self._listener_wait_task: asyncio.Task[None] | None = None
        self._shutdown = asyncio.Event()
        self._writers: set[asyncio.StreamWriter] = set()
        self._writer_wait_tasks: dict[asyncio.StreamWriter, asyncio.Task[None]] = {}
        self._handlers: set[asyncio.Task[None]] = set()
        self._max_concurrent_connections = (
            self.service.limits.max_concurrent_connections
        )
        self._unauthenticated_idle_timeout_seconds = (
            self.service.limits.unauthenticated_idle_timeout_ms / 1_000.0
        )
        self._active_connection_count = 0
        self._maintenance_ready = asyncio.Event()
        self._maintenance_enabled = asyncio.Event()
        self._maintenance_first_pass = asyncio.Event()
        self._maintenance_periodic_enabled = asyncio.Event()
        self._maintenance_first_cutoff: datetime | None = None
        self._maintenance_periodic_not_before: float | None = None
        self._maintenance_task: asyncio.Task[None] | None = None
        self._maintenance_error: BaseException | None = None
        self._fatal_error: BaseException | None = None
        self._transport_quiesced = False
        self._cleanup_lock = asyncio.Lock()
        runtime._install_transport_cleanup_owner(self)
        runtime._require_daemon_serving_boundary()

    def _consume_listener_wait(self, task: asyncio.Task[None]) -> None:
        if self._listener_wait_task is task:
            self._listener_wait_task = None
        with suppress(BaseException):
            task.result()

    async def _wait_listener_closed(self, *, deadline: float) -> BaseException | None:
        server = self._server
        if server is None:
            return None
        task = self._listener_wait_task
        if task is None:
            task = asyncio.create_task(server.wait_closed())
            self._listener_wait_task = task
            task.add_done_callback(self._consume_listener_wait)
        done, _pending = await asyncio.wait(
            (task,),
            timeout=max(0.0, deadline - asyncio.get_running_loop().time()),
        )
        if task not in done:
            task.cancel()
            return SchedulerShutdownError(
                "daemon listener did not close within its bound"
            )
        try:
            task.result()
        except BaseException as error:
            return error
        return None

    def _consume_writer_wait(
        self,
        writer: asyncio.StreamWriter,
        task: asyncio.Task[None],
    ) -> None:
        if self._writer_wait_tasks.get(writer) is task:
            del self._writer_wait_tasks[writer]
        with suppress(BaseException):
            task.result()

    async def _wait_writers_closed(
        self,
        writers: tuple[asyncio.StreamWriter, ...],
        *,
        deadline: float,
    ) -> dict[asyncio.StreamWriter, BaseException | None]:
        """Give every writer one shared shutdown deadline."""

        tasks: dict[asyncio.StreamWriter, asyncio.Task[None]] = {}
        for writer in writers:
            task = self._writer_wait_tasks.get(writer)
            if task is None:
                task = asyncio.create_task(writer.wait_closed())
                self._writer_wait_tasks[writer] = task
                task.add_done_callback(
                    lambda completed, owned_writer=writer: self._consume_writer_wait(
                        owned_writer, completed
                    )
                )
            tasks[writer] = task
        if not tasks:
            return {}
        _done, pending = await asyncio.wait(
            tuple(tasks.values()),
            timeout=max(0.0, deadline - asyncio.get_running_loop().time()),
        )
        for task in pending:
            task.cancel()
        results: dict[asyncio.StreamWriter, BaseException | None] = {}
        for writer, task in tasks.items():
            if task in pending:
                results[writer] = SchedulerShutdownError(
                    "daemon client writers did not close within the shared bound"
                )
                continue
            try:
                task.result()
            except (BrokenPipeError, ConnectionError):
                results[writer] = None
            except BaseException as error:
                results[writer] = error
            else:
                results[writer] = None
        return results

    @property
    def runtime_home(self) -> Path:
        return self.runtime.runtime_home

    @property
    def closed(self) -> bool:
        return self._transport_quiesced and self.runtime.closed

    def close(self) -> None:
        """Retry only after asynchronous transport shutdown was proven."""
        if not self._transport_quiesced:
            raise SchedulerShutdownError(
                "daemon transport authority is still quarantined"
            )
        HermesDaemonRuntime._claim_failed_owner(self)
        try:
            self.runtime.close()
        except BaseException:
            HermesDaemonRuntime._restore_claimed_failed_owner(self)
            raise
        HermesDaemonRuntime._discard_exact_closed_owner(self)

    async def _run_abandonment_pass(self) -> None:
        cutoff = self._maintenance_first_cutoff
        if cutoff is None:
            cutoff = datetime.now(UTC) - timedelta(
                seconds=self.abandonment_grace_seconds
            )
        else:
            self._maintenance_first_cutoff = None
        candidates = await asyncio.to_thread(
            self.runtime.ledger.list_abandonable_bridge_streams,
            last_seen_before=cutoff,
        )
        for bridge_instance_id, brain_id in candidates:
            if self._shutdown.is_set():
                return
            try:
                engine = await asyncio.to_thread(self.runtime.engine, brain_id)
                await asyncio.to_thread(
                    engine.abandon_bridge_stream,
                    bridge_instance_id,
                    last_seen_not_after=cutoff,
                )
            except (BridgeBindingError, BridgeClosedError, KeyError):
                # A reconnect or clean close may win after the candidate read.
                continue

    def _observe_runtime_fail_stop(self) -> bool:
        if not self.runtime.fail_stopped:
            return False
        if self._fatal_error is None:
            self._fatal_error = self.runtime.fail_stop_error or RuntimeError(
                "daemon runtime entered fail-stop"
            )
        self.service.begin_shutdown()
        self._shutdown.set()
        return True

    async def _maintain_abandoned_streams(self) -> None:
        interval = min(1.0, max(0.01, self.abandonment_grace_seconds / 4.0))
        try:
            self._maintenance_ready.set()
            await self._maintenance_enabled.wait()
            if self._shutdown.is_set():
                self._maintenance_first_pass.set()
                return
            await self._run_abandonment_pass()
            self._maintenance_first_pass.set()
            if self._observe_runtime_fail_stop():
                return
            # The startup task refreshes restart grace at the exact readiness
            # boundary before periodic abandonment is allowed to resume.
            await self._maintenance_periodic_enabled.wait()
            if self._shutdown.is_set():
                return
            not_before = self._maintenance_periodic_not_before
            if not_before is None:
                raise RuntimeError(
                    "periodic abandonment boundary is not initialized"
                )
            loop = asyncio.get_running_loop()
            while (remaining := not_before - loop.time()) > 0:
                try:
                    await asyncio.wait_for(self._shutdown.wait(), timeout=remaining)
                except TimeoutError:
                    continue
                else:
                    return
            self._maintenance_periodic_not_before = None
            await self._run_abandonment_pass()
            if self._observe_runtime_fail_stop():
                return
            while not self._shutdown.is_set():
                try:
                    await asyncio.wait_for(self._shutdown.wait(), timeout=interval)
                except TimeoutError:
                    await self._run_abandonment_pass()
                    if self._observe_runtime_fail_stop():
                        return
        except BaseException as error:
            if self.runtime.fail_stopped:
                self._fatal_error = self.runtime.fail_stop_error or error
            else:
                self._maintenance_error = error
            self._maintenance_ready.set()
            self._maintenance_first_pass.set()
            self.service.begin_shutdown()
            self._shutdown.set()

    async def _send(self, writer: asyncio.StreamWriter, response: bytes) -> bool:
        try:
            writer.write(response + b"\n")
            await writer.drain()
            return True
        except (BrokenPipeError, ConnectionError):
            return False

    async def _process_frame(
        self,
        connection: object,
        writer: asyncio.StreamWriter,
        frame: bytes,
    ) -> bool:
        response = await asyncio.to_thread(connection.handle_frame, frame)
        sent = await self._send(writer, response)
        if self._observe_runtime_fail_stop():
            return False
        if not sent:
            return False
        if connection.shutdown_requested:
            self.service.begin_shutdown()
            self._shutdown.set()
            return False
        return True

    async def _process_frame_with_auth_deadline(
        self,
        connection: object,
        writer: asyncio.StreamWriter,
        frame: bytes,
        *,
        authentication_deadline: float,
    ) -> bool:
        if getattr(connection, "authenticated", False):
            return await self._process_frame(connection, writer, frame)
        remaining = authentication_deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            return False
        try:
            return await asyncio.wait_for(
                self._process_frame(connection, writer, frame),
                timeout=remaining,
            )
        except TimeoutError:
            return False

    def _record_connection_cleanup_failure(self, error: BaseException) -> None:
        if self.runtime.fail_stopped:
            if self._fatal_error is None:
                self._fatal_error = self.runtime.fail_stop_error or error
            self.service.begin_shutdown()
            self._shutdown.set()
            return
        if self._maintenance_error is None:
            self._maintenance_error = error
        self.service.begin_shutdown()
        self._shutdown.set()

    async def _handle_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        task = asyncio.current_task()
        if task is not None:
            self._handlers.add(task)
        self._writers.add(writer)
        connection = None
        admitted = False
        try:
            if not _is_private_ipv4_stream(writer):
                return
            if self._active_connection_count >= self._max_concurrent_connections:
                return
            self._active_connection_count += 1
            admitted = True
            connection = self.service.new_connection()
            buffer = bytearray()
            discarding = False
            maximum = self.service.limits.max_request_bytes
            keep_running = True
            authentication_deadline = (
                asyncio.get_running_loop().time()
                + self._unauthenticated_idle_timeout_seconds
            )
            while keep_running and not self.service.shutting_down:
                if getattr(connection, "authenticated", False):
                    chunk = await reader.read(16_384)
                else:
                    remaining = (
                        authentication_deadline - asyncio.get_running_loop().time()
                    )
                    if remaining <= 0:
                        break
                    try:
                        chunk = await asyncio.wait_for(
                            reader.read(16_384), timeout=remaining
                        )
                    except TimeoutError:
                        break
                if not chunk:
                    if discarding:
                        oversized = b"x" * (maximum + 1)
                        await self._process_frame_with_auth_deadline(
                            connection,
                            writer,
                            oversized,
                            authentication_deadline=authentication_deadline,
                        )
                    elif buffer:
                        await self._process_frame_with_auth_deadline(
                            connection,
                            writer,
                            bytes(buffer),
                            authentication_deadline=authentication_deadline,
                        )
                    break
                cursor = 0
                while keep_running:
                    newline = chunk.find(b"\n", cursor)
                    if newline < 0:
                        tail = chunk[cursor:]
                        if not discarding:
                            buffer.extend(tail)
                            if len(buffer) > maximum:
                                buffer.clear()
                                discarding = True
                        break
                    fragment = chunk[cursor:newline]
                    cursor = newline + 1
                    if discarding:
                        discarding = False
                        oversized = b"x" * (maximum + 1)
                        keep_running = await self._process_frame_with_auth_deadline(
                            connection,
                            writer,
                            oversized,
                            authentication_deadline=authentication_deadline,
                        )
                        continue
                    buffer.extend(fragment)
                    if len(buffer) > maximum:
                        frame = b"x" * (maximum + 1)
                    else:
                        frame = bytes(buffer)
                        if frame.endswith(b"\r"):
                            frame = frame[:-1]
                    buffer.clear()
                    keep_running = await self._process_frame_with_auth_deadline(
                        connection,
                        writer,
                        frame,
                        authentication_deadline=authentication_deadline,
                    )
                    if cursor >= len(chunk):
                        break
        except (ConnectionError, asyncio.CancelledError):
            pass
        except BaseException as error:
            self._record_connection_cleanup_failure(error)
        finally:
            try:
                if connection is not None:
                    try:
                        await asyncio.to_thread(connection.close)
                    except BaseException as error:
                        self._record_connection_cleanup_failure(error)
            finally:
                writer_quiesced = True
                try:
                    writer.close()
                except BaseException as error:
                    writer_quiesced = False
                    self._record_connection_cleanup_failure(error)
                try:
                    with suppress(BrokenPipeError, ConnectionError):
                        await writer.wait_closed()
                except BaseException as error:
                    writer_quiesced = False
                    self._record_connection_cleanup_failure(error)
                finally:
                    if admitted:
                        self._active_connection_count -= 1
                    if writer_quiesced:
                        self._writers.discard(writer)
                    if task is not None:
                        self._handlers.discard(task)

    async def start(self) -> DaemonDiscoveryV2:
        self._server = await asyncio.start_server(
            self._handle_client,
            host="127.0.0.1",
            port=0,
            family=socket.AF_INET,
            start_serving=True,
            backlog=self._max_concurrent_connections,
        )
        sockets = self._server.sockets or ()
        if len(sockets) != 1:
            raise RuntimeError("daemon did not bind exactly one loopback socket")
        host, port = sockets[0].getsockname()[:2]
        if host != "127.0.0.1":
            raise RuntimeError("daemon socket is not numeric IPv4 loopback")
        record = DaemonDiscoveryV2(
            pid=os.getpid(),
            process_marker=self.runtime.lease.process_marker,
            instance_nonce=self.runtime.lease.instance_nonce,
            launch_nonce=self.runtime.lease.launch_nonce,
            endpoint=LoopbackEndpointV1(host="127.0.0.1", port=port),
            package_version=__version__,
            credential_ref=self.runtime.credential.path.name,
        )
        self.runtime.lease.assert_authority()
        publish_discovery(self.runtime.lease, record)
        self.runtime.lease.assert_authority()
        return record

    async def prove_readiness(self, record: DaemonDiscoveryV2) -> None:
        from alice_brain_hermes.protocol.client import DaemonClient

        client = await asyncio.to_thread(
            DaemonClient.connect,
            self.runtime.runtime_home,
            initialize=False,
        )
        try:
            health = await asyncio.to_thread(client.health)
            if health.get("instance_nonce") != record.instance_nonce:
                raise RuntimeError("authenticated health nonce mismatch")
            if health.get("launch_nonce") != record.launch_nonce:
                raise RuntimeError("authenticated health launch nonce mismatch")
            writers_ready = (
                health.get("brain_count") == health.get("engine_count")
                and health.get("brain_count") == health.get("scheduler_count")
                and health.get("brain_count")
                == health.get("running_scheduler_count")
            )
            if not writers_ready or health.get("runtime_ready") is not False:
                raise RuntimeError(
                    "authenticated health did not prove pre-serve readiness"
                )
        finally:
            await asyncio.to_thread(client.close)

    async def _cleanup_after_run(self) -> BaseException | None:
        async with self._cleanup_lock:
            return await self._cleanup_after_run_locked()

    async def _cleanup_after_run_locked(self) -> BaseException | None:
        HermesDaemonRuntime._claim_failed_owner(self)
        failure: BaseException | None = None
        deadline = asyncio.get_running_loop().time() + _SHUTDOWN_DRAIN_TIMEOUT_SECONDS
        self.runtime._leave_daemon_serving_boundary()
        self.service.begin_shutdown()
        self._shutdown.set()
        self._maintenance_enabled.set()
        self._maintenance_periodic_enabled.set()

        listener_quiesced = self._server is None
        if self._server is not None:
            try:
                self._server.close()
            except BaseException as error:
                failure = error
            else:
                error = await self._wait_listener_closed(deadline=deadline)
                if error is not None:
                    if failure is None:
                        failure = error
                else:
                    listener_quiesced = True

        maintenance_quiesced = self._maintenance_task is None
        if self._maintenance_task is not None:
            done, _pending = await asyncio.wait(
                (self._maintenance_task,),
                timeout=max(0.0, deadline - asyncio.get_running_loop().time()),
            )
            maintenance_quiesced = self._maintenance_task in done
            if maintenance_quiesced:
                try:
                    self._maintenance_task.result()
                except BaseException as error:
                    if failure is None:
                        failure = error
            elif failure is None:
                failure = SchedulerShutdownError(
                    "daemon maintenance did not stop within its bound"
                )

        writers_quiesced = listener_quiesced
        handlers_quiesced = listener_quiesced
        if listener_quiesced:
            writers = tuple(self._writers)
            unproven_writers: set[asyncio.StreamWriter] = set()
            handlers = tuple(
                task for task in self._handlers if task is not asyncio.current_task()
            )
            for writer in writers:
                try:
                    writer.close()
                except BaseException as error:
                    unproven_writers.add(writer)
                    if failure is None:
                        failure = error
            writer_results = await self._wait_writers_closed(
                writers,
                deadline=deadline,
            )
            for writer in writers:
                error = writer_results[writer]
                if error is not None:
                    unproven_writers.add(writer)
                    if failure is None:
                        failure = error
                else:
                    if writer not in unproven_writers:
                        self._writers.discard(writer)
            writers_quiesced = not unproven_writers

            if handlers:
                done, pending = await asyncio.wait(
                    handlers,
                    timeout=max(0.0, deadline - asyncio.get_running_loop().time()),
                )
                handlers_quiesced = not pending
                for task in done:
                    try:
                        task.result()
                    except (ConnectionError, asyncio.CancelledError):
                        pass
                    except BaseException as error:
                        if failure is None:
                            failure = error
                if pending and failure is None:
                    failure = SchedulerShutdownError(
                        "daemon client handlers did not drain within their bound"
                    )

        self._transport_quiesced = (
            listener_quiesced
            and maintenance_quiesced
            and writers_quiesced
            and handlers_quiesced
        )
        if self._transport_quiesced:
            try:
                await asyncio.to_thread(self.runtime.close)
            except BaseException as error:
                if failure is None:
                    failure = error
        else:
            try:
                HermesDaemonRuntime._retain_failed_owner(self)
            except BaseException as error:
                if failure is None:
                    failure = error
            if failure is None:
                failure = SchedulerShutdownError(
                    "daemon transport shutdown could not be proven"
                )
        if self.closed:
            HermesDaemonRuntime._discard_exact_closed_owner(self)
        return failure

    async def _cleanup_uninterruptibly(self) -> BaseException | None:
        cleanup = asyncio.create_task(
            self._cleanup_after_run(),
            name="alice-brain-hermes-daemon-cleanup",
        )
        while True:
            try:
                failure = await asyncio.shield(cleanup)
                break
            except asyncio.CancelledError:
                # Preserve fail-stop cleanup even when shutdown is cancelled
                # repeatedly. The original cancellation remains the run error.
                continue
        if not self._transport_quiesced:
            if isinstance(failure, SchedulerShutdownError):
                return failure
            shutdown_failure = SchedulerShutdownError(
                "daemon transport shutdown could not be proven"
            )
            if failure is not None:
                shutdown_failure.__cause__ = failure
            return shutdown_failure
        return failure

    async def _serve_until_shutdown(self) -> None:
        if self._shutdown.is_set():
            return
        self.runtime._enter_daemon_serving_boundary()
        try:
            await self._shutdown.wait()
        finally:
            self.runtime._leave_daemon_serving_boundary()

    async def run(self) -> None:
        failure: BaseException | None = None
        try:
            self._maintenance_task = asyncio.create_task(
                self._maintain_abandoned_streams(),
                name="alice-brain-hermes-bridge-maintenance",
            )
            await self._maintenance_ready.wait()
            if self._fatal_error is not None:
                raise self._fatal_error
            if self._maintenance_error is not None:
                raise self._maintenance_error
            record = await self.start()
            await self.prove_readiness(record)
            await asyncio.to_thread(self.runtime.ledger.refresh_daemon_restart_grace)
            self._maintenance_first_cutoff = datetime.min.replace(tzinfo=UTC)
            self._maintenance_enabled.set()
            await self._maintenance_first_pass.wait()
            if self._fatal_error is not None:
                raise self._fatal_error
            if self._maintenance_error is not None:
                raise self._maintenance_error
            # Startup work and the mandatory first maintenance pass must not
            # consume a prior stream's user-visible reconnect window. Refresh
            # once more at the final readiness boundary so the advertised
            # daemon always grants the complete configured grace interval.
            await asyncio.to_thread(self.runtime.ledger.refresh_daemon_restart_grace)
            self._maintenance_periodic_not_before = (
                asyncio.get_running_loop().time() + self.abandonment_grace_seconds
            )
            self._maintenance_periodic_enabled.set()
            await self._serve_until_shutdown()
            if self._fatal_error is not None:
                raise self._fatal_error
            if self._maintenance_error is not None:
                raise self._maintenance_error
        except BaseException as error:
            failure = error
        finally:
            cleanup_failure = await self._cleanup_uninterruptibly()
            if isinstance(cleanup_failure, SchedulerShutdownError):
                if failure is not None and cleanup_failure.__cause__ is None:
                    cleanup_failure.__cause__ = failure
                failure = cleanup_failure
            elif failure is None:
                failure = cleanup_failure

        if failure is not None:
            raise failure


async def _run_daemon(
    runtime_home: Path,
    *,
    launch_nonce: str | None = None,
    scheduler_interval_seconds: float,
    abandonment_grace_seconds: float,
) -> None:
    scheduler_interval_seconds = _positive_seconds(
        scheduler_interval_seconds,
        name="scheduler_interval_seconds",
    )
    abandonment_grace_seconds = _positive_seconds(
        abandonment_grace_seconds,
        name="abandonment_grace_seconds",
    )
    await HermesDaemonRuntime.retry_failed_cleanup_async(runtime_home)
    runtime = await asyncio.to_thread(
        HermesDaemonRuntime.open,
        runtime_home,
        scheduler_interval_seconds=scheduler_interval_seconds,
        launch_nonce=launch_nonce,
    )
    try:
        daemon = PrivateDaemonServer(
            runtime=runtime,
            abandonment_grace_seconds=abandonment_grace_seconds,
        )
    except BaseException:
        await asyncio.to_thread(runtime.close)
        raise
    # From this point the daemon owns transport quiescence and the runtime.
    # Its fail-stop cleanup may intentionally retain both; an outer close would
    # invalidate that authority boundary.
    await daemon.run()


def _run_private_daemon_loop(
    coroutine: Coroutine[object, object, None],
) -> None:
    """Run one daemon coroutine without joining quarantined executor work.

    A normal return and ordinary failures retain ``asyncio.run``'s graceful
    executor shutdown.  ``SchedulerShutdownError`` means a writer or transport
    worker could not be proven stopped; joining that worker would make the CLI's
    shutdown bound fictional.  In that fail-stop case the caller must terminate
    the process rather than continue using quarantined runtime authority.
    """

    loop = asyncio.new_event_loop()
    fail_stop = False
    try:
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(coroutine)
        except SchedulerShutdownError:
            fail_stop = True
            raise
        finally:
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
                if fail_stop:
                    # A fail-stop return is followed immediately by os._exit.
                    # Running coroutine finalizers here can enqueue another
                    # non-cooperative executor call and make that hard exit
                    # unreachable.  These tasks are deliberately quarantined,
                    # so their normal "destroyed pending" diagnostic would be
                    # misleading during direct helper tests.
                    task._log_destroy_pending = False
            if pending and not fail_stop:
                loop.run_until_complete(
                    asyncio.gather(*pending, return_exceptions=True)
                )
            if not fail_stop:
                loop.run_until_complete(loop.shutdown_asyncgens())
                loop.run_until_complete(loop.shutdown_default_executor())
    finally:
        asyncio.set_event_loop(None)
        loop.close()


def run_private_daemon(
    runtime_home: str | Path,
    *,
    launch_nonce: str | None = None,
    scheduler_interval_seconds: float = 1.0,
    abandonment_grace_seconds: float = 30.0,
) -> None:
    _run_private_daemon_loop(
        _run_daemon(
            Path(runtime_home),
            launch_nonce=launch_nonce,
            scheduler_interval_seconds=scheduler_interval_seconds,
            abandonment_grace_seconds=abandonment_grace_seconds,
        )
    )


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--runtime-home", required=True)
    parser.add_argument("--launch-nonce")
    parser.add_argument("--scheduler-interval", type=float, default=1.0)
    parser.add_argument("--abandonment-grace", type=float, default=30.0)
    arguments = parser.parse_args(argv)
    try:
        run_private_daemon(
            arguments.runtime_home,
            launch_nonce=arguments.launch_nonce,
            scheduler_interval_seconds=arguments.scheduler_interval,
            abandonment_grace_seconds=arguments.abandonment_grace,
        )
        return 0
    except RuntimeOwnedError:
        return 2
    except SchedulerShutdownError:
        # The event loop deliberately did not join executor workers whose
        # termination could not be proven.  Returning to Python would allow
        # those workers to retain runtime authority, and interpreter shutdown
        # could wait forever.  The daemon CLI therefore fails closed here.
        os._exit(3)
    except Exception as error:
        print(
            f"alice-brain-hermes daemon failed: {type(error).__name__}",
            file=sys.stderr,
            flush=True,
        )
        return 1


if __name__ == "__main__":
    sys.exit(_main())
