from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from alice_brain_hermes import errors
from alice_brain_hermes.core import state as state_module
from alice_brain_hermes.core.cognition import LocalCognitionPort, result_payload
from alice_brain_hermes.core.events import new_event
from alice_brain_hermes.core.reducer import reduce_state
from alice_brain_hermes.core.state import BrainState
from alice_brain_hermes.ids import new_id
from alice_brain_hermes.protocol.models import CoverageV1, validate_observation
from alice_brain_hermes.runtime.engine import ConsciousEngine
from alice_brain_hermes.runtime.store import SQLiteLedger


def _next(
    state: BrainState,
    event_type: str,
    payload: dict[str, object],
) -> BrainState:
    return reduce_state(
        state,
        new_event(
            event_type,
            state.brain_id,
            state.brain_id,
            payload,
            sequence=state.last_sequence + 1,
        ),
    )


def _complete_action(state: BrainState, action_id: str) -> BrainState:
    for event_type, payload in (
        ("action.proposed", {"action_id": action_id, "intent": {}}),
        (
            "action.energy_assessed",
            {
                "action_id": action_id,
                "salience": 0.5,
                "urgency": 0.5,
                "valence": 0.0,
                "arousal": 0.0,
                "control": 0.5,
                "resources": 0.5,
                "cost": 0.5,
                "personality_relevance": 0.5,
            },
        ),
        ("action.prepared", {"action_id": action_id}),
        ("action.dispatched", {"action_id": action_id}),
        ("action.receipt", {"action_id": action_id, "status": "success"}),
        ("action.reconstructed", {"action_id": action_id}),
    ):
        state = _next(state, event_type, payload)
    return state


def test_cognition_rejects_duplicate_alternative_branch_ids() -> None:
    state = BrainState.genesis(new_id())
    payload = result_payload(LocalCognitionPort().reflect({"duplicate": True}))
    payload["alternatives"][1]["branch_id"] = payload["alternatives"][0]["branch_id"]

    with pytest.raises(errors.DomainInvariantError, match="invalid local cognition"):
        _next(state, "cognition.reflected", payload)


def test_terminal_action_details_evict_but_active_overflow_is_explicit() -> None:
    limit = state_module.MAX_ACTION_RECORDS
    state = BrainState.genesis(new_id())
    for index in range(limit + 3):
        state = _complete_action(state, f"terminal-{index:04d}")

    assert len(state.action_records) == limit
    assert state.action_records[0].action_id == "terminal-0003"
    assert state.working_set.action_records.total == limit + 3
    assert state.working_set.action_records.evicted == 3
    assert len(state.energy_records) == state_module.MAX_ENERGY_RECORDS
    assert state.working_set.energy_records.total == limit + 3
    assert state.working_set.energy_records.evicted == 3
    assert BrainState.model_validate_json(state.canonical_json()) == state

    active = BrainState.genesis(new_id())
    for index in range(limit):
        active = _next(
            active,
            "action.proposed",
            {"action_id": f"active-{index:04d}", "intent": {}},
        )
    capacity_error = errors.DomainCapacityError
    with pytest.raises(capacity_error, match="active action capacity"):
        _next(
            active,
            "action.proposed",
            {"action_id": "active-overflow", "intent": {}},
        )
    assert len(active.action_records) == limit

    mixed = BrainState.genesis(new_id())
    mixed = _next(
        mixed,
        "action.proposed",
        {"action_id": "mixed-oldest-active", "intent": {}},
    )
    mixed = _complete_action(mixed, "mixed-terminal")
    for index in range(limit - 2):
        mixed = _next(
            mixed,
            "action.proposed",
            {"action_id": f"mixed-active-{index:04d}", "intent": {}},
        )
    mixed = _next(
        mixed,
        "action.proposed",
        {"action_id": "mixed-new", "intent": {}},
    )
    mixed_ids = {item.action_id for item in mixed.action_records}
    assert "mixed-oldest-active" in mixed_ids
    assert "mixed-terminal" not in mixed_ids
    assert "mixed-new" in mixed_ids

    frame = SQLiteLedger._project_bridge_frame(state, through_capture_seq=0)
    for section, counter_name in (
        ("rd", "action_records"),
        ("energy", "energy_records"),
    ):
        omission = frame.omission_counts[section]
        counter = getattr(state.working_set, counter_name)
        assert omission["included"] + omission["omitted"] == counter.total


def test_every_accumulated_state_collection_has_a_visible_fixed_limit() -> None:
    world = BrainState.genesis(new_id())
    for index in range(state_module.MAX_WORLD_PROPOSITIONS_PER_LAYER + 2):
        world = _next(
            world,
            "observation.recorded",
            {"proposition_id": f"fact-{index:04d}", "content": {}},
        )
    assert len(world.world.observed) == (state_module.MAX_WORLD_PROPOSITIONS_PER_LAYER)
    assert world.working_set.world_observed.evicted == 2
    for event_type, field_name, counter_name in (
        ("belief.updated", "believed", "world_believed"),
        ("simulation.created", "simulated", "world_simulated"),
        ("ideal.updated", "ideal", "world_ideal"),
    ):
        layered = BrainState.genesis(new_id())
        for index in range(state_module.MAX_WORLD_PROPOSITIONS_PER_LAYER + 2):
            layered = _next(
                layered,
                event_type,
                {
                    "proposition_id": f"{field_name}-{index:04d}",
                    "content": {},
                },
            )
        assert len(getattr(layered.world, field_name)) == (
            state_module.MAX_WORLD_PROPOSITIONS_PER_LAYER
        )
        assert getattr(layered.working_set, counter_name).evicted == 2

    thought = BrainState.genesis(new_id())
    for index in range(state_module.MAX_THOUGHT_BRANCHES + 2):
        thought = _next(
            thought,
            "simulation.created",
            {
                "branch_id": f"branch-{index:04d}",
                "proposition_id": f"simulation-{index:04d}",
                "content": {},
            },
        )
    assert len(thought.thought_space) == state_module.MAX_THOUGHT_BRANCHES
    assert thought.working_set.thought_space.evicted == 2

    memory = BrainState.genesis(new_id())
    for index in range(state_module.MAX_MEMORY_RECORDS + 2):
        memory = _next(
            memory,
            "memory.recorded",
            {"memory_id": f"memory-{index:04d}", "content": {}},
        )
    assert len(memory.memories) == state_module.MAX_MEMORY_RECORDS
    assert memory.working_set.memories.evicted == 2

    cognition = BrainState.genesis(new_id())
    reflection = result_payload(LocalCognitionPort().reflect({"bounded": True}))
    for _ in range(state_module.MAX_COGNITION_REFLECTIONS + 2):
        cognition = _next(cognition, "cognition.reflected", reflection)
    assert len(cognition.cognition.reflections) == (
        state_module.MAX_COGNITION_REFLECTIONS
    )
    assert cognition.working_set.cognition_reflections.evicted == 2

    raw = BrainState.genesis(new_id())
    for index in range(state_module.MAX_RAW_LIFECYCLE_KEYS + 2):
        raw = _next(raw, f"future.lifecycle.{index:04d}", {})
    assert len(raw.raw_lifecycle_counts) == state_module.MAX_RAW_LIFECYCLE_KEYS
    assert raw.working_set.raw_lifecycle_counts.total == (
        state_module.MAX_RAW_LIFECYCLE_KEYS + 2
    )
    assert raw.working_set.raw_lifecycle_counts.evicted == 2
    assert (
        sum(raw.raw_lifecycle_counts.values())
        + (raw.working_set.raw_lifecycle_events_evicted)
        == raw.last_sequence
    )

    personality = BrainState.genesis(new_id())
    personality = _next(
        personality,
        "personality.revised",
        {
            "layer": "traits",
            "values": {
                f"trait-{index:04d}": 0.0
                for index in range(state_module.MAX_PERSONALITY_VALUES_PER_LAYER)
            },
        },
    )
    with pytest.raises(errors.DomainCapacityError, match="personality"):
        _next(
            personality,
            "personality.revised",
            {"layer": "traits", "values": {"overflow": 0.0}},
        )
    for layer in ("adaptations", "narrative_ideal"):
        layered_personality = BrainState.genesis(new_id())
        layered_personality = _next(
            layered_personality,
            "personality.revised",
            {
                "layer": layer,
                "values": {
                    f"{layer}-{index:04d}": 0.0
                    for index in range(state_module.MAX_PERSONALITY_VALUES_PER_LAYER)
                },
            },
        )
        assert len(getattr(layered_personality.personality, layer)) == (
            state_module.MAX_PERSONALITY_VALUES_PER_LAYER
        )
        with pytest.raises(errors.DomainCapacityError, match="personality"):
            _next(
                layered_personality,
                "personality.revised",
                {"layer": layer, "values": {"overflow": 0.0}},
            )

    capabilities = BrainState.genesis(new_id())
    capabilities = _next(
        capabilities,
        "capabilities.reported",
        {
            "capabilities": {
                f"capability-{index:04d}": True
                for index in range(state_module.MAX_CAPABILITIES)
            }
        },
    )
    assert len(capabilities.capabilities) == state_module.MAX_CAPABILITIES
    with pytest.raises(errors.DomainCapacityError, match="capabilities"):
        _next(
            capabilities,
            "capabilities.reported",
            {
                "capabilities": {
                    f"capability-{index:04d}": True
                    for index in range(state_module.MAX_CAPABILITIES + 1)
                }
            },
        )

    identity = BrainState.genesis(new_id())
    for _ in range(state_module.MAX_IDENTITY_ACTORS - 1):
        identity = _next(
            identity,
            "identity.actor_registered",
            {"actor_id": new_id(), "kind": "human"},
        )
    with pytest.raises(errors.DomainCapacityError, match="actor capacity"):
        _next(
            identity,
            "identity.actor_registered",
            {"actor_id": new_id(), "kind": "human"},
        )

    authorizations = BrainState.genesis(new_id())
    for _ in range(state_module.MAX_PROVENANCE_AUTHORIZATIONS):
        authorizations = _next(
            authorizations,
            "identity.provenance_authorized",
            {"actor_id": new_id(), "adapter_id": "bounded-adapter"},
        )
    with pytest.raises(errors.DomainCapacityError, match="authorization capacity"):
        _next(
            authorizations,
            "identity.provenance_authorized",
            {"actor_id": new_id(), "adapter_id": "bounded-adapter"},
        )

    expected_counter_fields = {
        "action_records",
        "energy_records",
        "thought_space",
        "memories",
        "cognition_reflections",
        "world_observed",
        "world_believed",
        "world_simulated",
        "world_ideal",
        "identity_actors",
        "provenance_authorizations",
        "personality_traits",
        "personality_adaptations",
        "personality_narrative_ideal",
        "raw_lifecycle_counts",
        "raw_lifecycle_events_evicted",
        "reduced_event_count",
        "capabilities",
    }
    assert set(type(world.working_set).model_fields) == expected_counter_fields

    for bounded_state, section, counter_name in (
        (world, "world", "world_observed"),
        (thought, "st", "thought_space"),
        (memory, "memory", "memories"),
    ):
        frame = SQLiteLedger._project_bridge_frame(bounded_state, through_capture_seq=0)
        counter = getattr(bounded_state.working_set, counter_name)
        if section == "world":
            omission = frame.omission_counts[section]["layers"]["observed"]
        elif section == "st":
            omission = frame.omission_counts[section]["branches"]
        else:
            omission = frame.omission_counts[section]
        assert omission["included"] + omission["omitted"] == counter.total


def test_fully_saturated_reducer_state_fits_the_composite_frame_budget() -> None:
    initial = BrainState.genesis(new_id())
    state = initial.model_copy(
        update={
            "workspace": initial.workspace.model_copy(
                update={"capacity": state_module.MAX_WORKSPACE_BROADCAST}
            )
        }
    ).revalidated()

    for index in range(state_module.MAX_ACTION_RECORDS):
        action_id = f"saturated-action-{index:04d}"
        state = _next(
            state,
            "action.proposed",
            {"action_id": action_id, "intent": {}},
        )
        state = _next(
            state,
            "action.energy_assessed",
            {
                "action_id": action_id,
                "salience": 0.5,
                "urgency": 0.5,
                "valence": 0.0,
                "arousal": 0.0,
                "control": 0.5,
                "resources": 0.5,
                "cost": 0.5,
                "personality_relevance": 0.5,
            },
        )

    for index in range(state_module.MAX_MEMORY_RECORDS):
        state = _next(
            state,
            "memory.recorded",
            {
                "memory_id": f"saturated-memory-{index:04d}",
                "content": {"index": index},
            },
        )

    for index in range(state_module.MAX_COGNITION_REFLECTIONS):
        state = _next(
            state,
            "cognition.reflected",
            result_payload(LocalCognitionPort().reflect({"saturated": index})),
        )

    for event_type, prefix in (
        ("observation.recorded", "observed"),
        ("belief.updated", "believed"),
        ("simulation.created", "simulated"),
        ("ideal.updated", "ideal"),
    ):
        for index in range(state_module.MAX_WORLD_PROPOSITIONS_PER_LAYER):
            payload: dict[str, object] = {
                "proposition_id": f"{prefix}-{index:04d}",
                "content": {"index": index},
            }
            if event_type == "simulation.created":
                payload["branch_id"] = f"world-branch-{index:04d}"
            state = _next(state, event_type, payload)

    registered_actor_ids = [state.brain_id]
    for _ in range(state_module.MAX_IDENTITY_ACTORS - 1):
        actor_id = new_id()
        registered_actor_ids.append(actor_id)
        state = _next(
            state,
            "identity.actor_registered",
            {"actor_id": actor_id, "kind": "human"},
        )
    for index, actor_id in enumerate(registered_actor_ids):
        state = _next(
            state,
            "identity.provenance_authorized",
            {
                "actor_id": actor_id,
                "adapter_id": f"saturated-adapter-{index:04d}",
            },
        )

    # Register these event types before filling the raw-event-key working set.
    state = _next(
        state,
        "personality.revised",
        {"layer": "traits", "values": {}},
    )
    state = _next(
        state,
        "capabilities.reported",
        {"capabilities": {}},
    )
    state = _next(
        state,
        "workspace.broadcast",
        {"cycle": 1, "candidates": []},
    )
    raw_index = 0
    while len(state.raw_lifecycle_counts) < state_module.MAX_RAW_LIFECYCLE_KEYS:
        state = _next(state, f"saturated.raw.{raw_index:04d}", {})
        raw_index += 1

    for layer in ("traits", "adaptations", "narrative_ideal"):
        state = _next(
            state,
            "personality.revised",
            {
                "layer": layer,
                "values": {
                    f"{layer}-{index:04d}": 0.0
                    for index in range(state_module.MAX_PERSONALITY_VALUES_PER_LAYER)
                },
            },
        )
    state = _next(
        state,
        "capabilities.reported",
        {
            "capabilities": {
                f"capability-{index:04d}": True
                for index in range(state_module.MAX_CAPABILITIES)
            }
        },
    )
    workspace_candidates = [
        {
            "candidate_id": f"workspace-{index:04d}",
            "specialist": "memory",
            "score": 0.5,
            "content": {"index": index},
            "source_ids": [],
            "cycle": 2,
        }
        for index in range(state_module.MAX_WORKSPACE_BROADCAST)
    ]
    state = _next(
        state,
        "workspace.broadcast",
        {"cycle": 2, "candidates": workspace_candidates},
    )

    retained_lengths = {
        "action_records": len(state.action_records),
        "energy_records": len(state.energy_records),
        "thought_space": len(state.thought_space),
        "memories": len(state.memories),
        "cognition_reflections": len(state.cognition.reflections),
        "world_observed": len(state.world.observed),
        "world_believed": len(state.world.believed),
        "world_simulated": len(state.world.simulated),
        "world_ideal": len(state.world.ideal),
        "identity_actors": len(state.identity.actors),
        "provenance_authorizations": len(state.identity.authorizations),
        "personality_traits": len(state.personality.traits),
        "personality_adaptations": len(state.personality.adaptations),
        "personality_narrative_ideal": len(state.personality.narrative_ideal),
        "raw_lifecycle_counts": len(state.raw_lifecycle_counts),
        "capabilities": len(state.capabilities),
    }
    expected_lengths = {
        "action_records": state_module.MAX_ACTION_RECORDS,
        "energy_records": state_module.MAX_ENERGY_RECORDS,
        "thought_space": state_module.MAX_THOUGHT_BRANCHES,
        "memories": state_module.MAX_MEMORY_RECORDS,
        "cognition_reflections": state_module.MAX_COGNITION_REFLECTIONS,
        "world_observed": state_module.MAX_WORLD_PROPOSITIONS_PER_LAYER,
        "world_believed": state_module.MAX_WORLD_PROPOSITIONS_PER_LAYER,
        "world_simulated": state_module.MAX_WORLD_PROPOSITIONS_PER_LAYER,
        "world_ideal": state_module.MAX_WORLD_PROPOSITIONS_PER_LAYER,
        "identity_actors": state_module.MAX_IDENTITY_ACTORS,
        "provenance_authorizations": (state_module.MAX_PROVENANCE_AUTHORIZATIONS),
        "personality_traits": state_module.MAX_PERSONALITY_VALUES_PER_LAYER,
        "personality_adaptations": (state_module.MAX_PERSONALITY_VALUES_PER_LAYER),
        "personality_narrative_ideal": (state_module.MAX_PERSONALITY_VALUES_PER_LAYER),
        "raw_lifecycle_counts": state_module.MAX_RAW_LIFECYCLE_KEYS,
        "capabilities": state_module.MAX_CAPABILITIES,
    }
    assert retained_lengths == expected_lengths
    assert len(state.workspace.broadcast) == state_module.MAX_WORKSPACE_BROADCAST
    assert BrainState.model_validate_json(state.canonical_json()) == state

    frame = SQLiteLedger._project_bridge_frame(state, through_capture_seq=0)
    omission = frame.omission_counts

    def exact(section: object, expected_total: int) -> None:
        assert isinstance(section, Mapping)
        assert section["included"] + section["omitted"] == expected_total

    personality_total = sum(
        getattr(state.working_set, f"personality_{layer}").total
        for layer in ("traits", "adaptations", "narrative_ideal")
    )
    exact(omission["pc"], personality_total)
    exact(omission["energy"], state.working_set.energy_records.total)
    exact(
        omission["st"],
        len(state.workspace.broadcast) + state.working_set.thought_space.total,
    )
    exact(omission["st"]["workspace"], len(state.workspace.broadcast))
    exact(
        omission["st"]["branches"],
        state.working_set.thought_space.total,
    )
    exact(omission["rd"], state.working_set.action_records.total)
    exact(omission["a"], state.working_set.action_records.total)
    world_total = 0
    for layer in ("observed", "believed", "simulated", "ideal"):
        counter = getattr(state.working_set, f"world_{layer}")
        exact(omission["world"]["layers"][layer], counter.total)
        world_total += counter.total
    exact(omission["world"], world_total)
    exact(
        omission["self_boundary"],
        state.working_set.identity_actors.total - 1,
    )
    exact(
        omission["self_boundary"]["fields"]["authorization_records"],
        state.working_set.provenance_authorizations.total,
    )
    exact(omission["memory"], state.working_set.memories.total)
    exact(omission["capabilities"], state.working_set.capabilities.total)
    exact(
        omission["cognition"]["fields"]["reflection_records"],
        state.working_set.cognition_reflections.total,
    )
    exact(
        omission["raw_lifecycle_counts"],
        state.working_set.raw_lifecycle_counts.total,
    )
    work = omission["working_set"]
    assert work["counters"] == state.working_set.model_dump(mode="json")
    assert work["projection_records_visited"] <= (
        state_module.FRAME_PROJECTION_RECORD_BUDGET
    )
    assert work["projection_records_visited"] == work["projection_record_budget"]


def test_working_set_counter_tampering_is_rejected() -> None:
    state = _next(BrainState.genesis(new_id()), "future.lifecycle.event", {})
    values = state.model_dump(mode="python")
    values["working_set"]["raw_lifecycle_counts"]["total"] += 1
    with pytest.raises(ValidationError, match="counter does not match"):
        BrainState.model_validate(values)

    values = state.model_dump(mode="python")
    values.pop("working_set")
    with pytest.raises(ValidationError, match="working_set"):
        BrainState.model_validate(values)


def _observation(instance: str):
    return validate_observation(
        {
            "bridge_instance_id": instance,
            "capture_seq": 1,
            "captured_at": datetime.now(UTC),
            "captured_monotonic_ns": 1,
            "hook": "post_api_request",
            "context": {
                "session_id": "session-1",
                "task_id": "task-1",
                "turn_id": "turn-1",
                "api_request_id": "request-1",
            },
            "payload": {
                "platform": "test",
                "model": "model-y",
                "provider": "provider-x",
                "base_url": None,
                "api_mode": "streaming",
                "api_call_count": 1,
                "api_duration": 0.25,
                "started_at": "start",
                "ended_at": "end",
                "finish_reason": "stop",
                "message_count": 1,
                "response_model": "model-y",
                "response": {},
                "usage": {},
                "assistant_message": {},
                "assistant_content_chars": 0,
                "assistant_tool_call_count": 0,
                "extensions": {},
            },
            "coverage": CoverageV1(
                policy_version="copy-v1",
                capture_coverage="host_sanitized",
            ),
        }
    )


def test_long_history_replay_and_frame_projection_have_fixed_work_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    history_length = 10_000
    brain_id = new_id()
    database = tmp_path / "history.db"
    with SQLiteLedger.open(database) as ledger:
        ledger.ensure_brain(brain_id)
        instance = new_id()
        ledger.attach_bridge_stream(
            instance,
            brain_id=brain_id,
            server_actor_id=brain_id,
            server_adapter_id="bounded-history-test",
            connected_nonce="connection",
            recovery_token="ab" * 32,
        )

        short_engine = ConsciousEngine(ledger, brain_id, actor_id=brain_id)
        short_statements: list[str] = []
        original_connection = ledger._connection

        class CountingConnection:
            @property
            def in_transaction(self) -> bool:
                return original_connection.in_transaction

            def execute(self, statement: str, parameters=()):
                short_statements.append(" ".join(statement.lower().split()))
                return original_connection.execute(statement, parameters)

            def commit(self) -> None:
                original_connection.commit()

            def rollback(self) -> None:
                original_connection.rollback()

            def close(self) -> None:
                original_connection.close()

        ledger._connection = CountingConnection()
        short_frame = ledger.project_bridge_frame(
            instance,
            expected_state=short_engine.state,
            scheduler_sample="stopped",
        )
        ledger._connection = original_connection

        with ledger._transaction(immediate=True):
            for sequence in range(1, history_length + 1):
                event = new_event(
                    "future.lifecycle.event",
                    brain_id,
                    brain_id,
                    {},
                    sequence=sequence,
                )
                ledger._connection.execute(
                    "INSERT INTO events("
                    "brain_id, sequence, event_id, body_fingerprint, "
                    "envelope_fingerprint, envelope_json"
                    ") VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        brain_id,
                        sequence,
                        event.event_id,
                        event.body_fingerprint(),
                        event.envelope_fingerprint(),
                        event.canonical_json(),
                    ),
                )
            ledger._connection.execute(
                "UPDATE brains SET next_sequence = ? WHERE brain_id = ?",
                (history_length + 1, brain_id),
            )

        engine = ConsciousEngine(ledger, brain_id, actor_id=brain_id)

        def forbid_replay(*_args, **_kwargs):
            raise AssertionError("hot path attempted historical replay")

        for helper in (
            "_full_replay_in_transaction",
            "_replay_target_states_in_transaction",
            "_validate_replay_and_snapshots_in_transaction",
        ):
            monkeypatch.setattr(ledger, helper, forbid_replay)

        commit_statements: list[str] = []
        long_statements: list[str] = []

        class LongCountingConnection(CountingConnection):
            def execute(self, statement: str, parameters=()):
                long_statements.append(" ".join(statement.lower().split()))
                return original_connection.execute(statement, parameters)

        class CommitCountingConnection(CountingConnection):
            def execute(self, statement: str, parameters=()):
                commit_statements.append(" ".join(statement.lower().split()))
                return original_connection.execute(statement, parameters)

        ledger._connection = CommitCountingConnection()
        acknowledgement = engine.commit_bridge_record(instance, _observation(instance))
        ledger._connection = original_connection
        assert acknowledgement.event_sequence == history_length + 1

        ledger._connection = LongCountingConnection()
        frame = ledger.project_bridge_frame(
            instance,
            expected_state=engine.state,
            scheduler_sample="stopped",
        )
        ledger._connection = original_connection

        assert engine.state.last_sequence == history_length + 1
        assert engine.state.raw_lifecycle_counts["future.lifecycle.event"] == (
            history_length
        )
        assert engine.state.working_set.raw_lifecycle_counts.total == 2
        assert (
            ledger._connection.execute(
                "SELECT COUNT(*) FROM events WHERE brain_id = ?", (brain_id,)
            ).fetchone()[0]
            == history_length + 1
        )
        work = frame.omission_counts["working_set"]
        assert work["projection_records_visited"] <= (work["projection_record_budget"])
        assert work["projection_record_budget"] == (
            state_module.FRAME_PROJECTION_RECORD_BUDGET
        )
        assert work["ledger_events_scanned"] == 0
        assert len(short_statements) <= 12
        assert len(long_statements) <= 12
        assert not any(" from events" in statement for statement in long_statements)
        assert len(commit_statements) <= 20
        assert not any(
            "order by event.sequence asc" in statement
            or "order by sequence asc" in statement
            for statement in commit_statements
        )
        plan = ledger._connection.execute(
            "EXPLAIN QUERY PLAN SELECT COALESCE(MAX(sequence), 0) "
            "FROM events WHERE brain_id = ?",
            (brain_id,),
        ).fetchall()
        assert any("primary key" in row[3].lower() for row in plan)
        boundary = ledger.list_events(
            brain_id, after_sequence=history_length - 1, limit=2
        )
        assert [item.sequence for item in boundary] == [
            history_length,
            history_length + 1,
        ]
        assert boundary[0].event_type == "future.lifecycle.event"
        assert boundary[1].event_type == "hermes.observer.post_api_request"
        assert (
            short_frame.omission_counts["working_set"]["projection_records_visited"]
            <= state_module.FRAME_PROJECTION_RECORD_BUDGET
        )
