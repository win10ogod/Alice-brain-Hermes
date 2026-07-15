"""Pure deterministic local cognition for off-turn structured reflection."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from alice_brain_hermes.core.events import FrozenJsonDict, thaw_json
from alice_brain_hermes.core.limits import MAX_COGNITION_REFLECTIONS

COGNITION_ALGORITHM_VERSION = "local-branches-v1"
COGNITION_CONFIG_VERSION = "default-v1"


class CognitionAlternative(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid", frozen=True, strict=True, validate_default=True
    )

    branch_id: str = Field(min_length=1, max_length=256)
    stance: Literal["proceed", "defer", "seek_observation"]
    content: FrozenJsonDict
    expected_consequences: tuple[FrozenJsonDict, ...]

    @field_validator("expected_consequences", mode="before")
    @classmethod
    def _json_consequences(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value


class CognitionResult(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(
        allow_inf_nan=False,
        extra="forbid",
        frozen=True,
        strict=True,
        validate_default=True,
    )

    cognition_mode: Literal["local"] = "local"
    provider_used: Literal[False] = False
    algorithm_version: str = COGNITION_ALGORITHM_VERSION
    config_version: str = COGNITION_CONFIG_VERSION
    source_ids: tuple[str, ...]
    input_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    alternatives: tuple[CognitionAlternative, ...]
    uncertainty: float = Field(ge=0.0, le=1.0)
    uncertainty_basis: Literal["deterministic_heuristic"] = "deterministic_heuristic"
    calibrated: Literal[False] = False
    reflection: FrozenJsonDict

    @field_validator("source_ids", "alternatives", mode="before")
    @classmethod
    def _json_arrays_to_tuples(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @field_validator("alternatives")
    @classmethod
    def _unique_alternative_branch_ids(
        cls, value: tuple[CognitionAlternative, ...]
    ) -> tuple[CognitionAlternative, ...]:
        branch_ids = tuple(item.branch_id for item in value)
        if len(set(branch_ids)) != len(branch_ids):
            raise ValueError("cognition alternative branch_id values must be unique")
        return value


class CognitionState(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid", frozen=True, strict=True, validate_default=True
    )

    cognition_mode: Literal["local"] = "local"
    algorithm_version: str = COGNITION_ALGORITHM_VERSION
    config_version: str = COGNITION_CONFIG_VERSION
    reflections: tuple[CognitionResult, ...] = ()

    @field_validator("reflections", mode="before")
    @classmethod
    def _json_reflections(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @field_validator("reflections")
    @classmethod
    def _bounded_reflections(
        cls, value: tuple[CognitionResult, ...]
    ) -> tuple[CognitionResult, ...]:
        if len(value) > MAX_COGNITION_REFLECTIONS:
            raise ValueError("cognition reflections exceed working-set capacity")
        return value


def _canonical_content(content: Mapping[str, Any]) -> tuple[FrozenJsonDict, str]:
    frozen = FrozenJsonDict(content)
    canonical = frozen.canonical_json()
    fingerprint = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return frozen, fingerprint


class LocalCognitionPort:
    """No-provider deterministic branch generation; never an external fallback."""

    cognition_mode = "local"
    provider_used = False
    algorithm_version = COGNITION_ALGORITHM_VERSION
    config_version = COGNITION_CONFIG_VERSION

    def reflect(
        self,
        ignited_content: Mapping[str, Any],
        *,
        source_ids: tuple[str, ...] = (),
    ) -> CognitionResult:
        frozen, fingerprint = _canonical_content(ignited_content)
        sources = tuple(sorted(set(source_ids)))
        seed = int(fingerprint[:8], 16)
        uncertainty = round(0.25 + (seed % 5000) / 10_000, 4)
        alternatives = tuple(
            CognitionAlternative(
                branch_id=f"local-{fingerprint[:16]}-{stance}",
                stance=stance,
                content=FrozenJsonDict(
                    {
                        "input": thaw_json(frozen),
                        "policy": stance,
                    }
                ),
                expected_consequences=(
                    FrozenJsonDict(
                        {
                            "kind": "local_projection",
                            "stance": stance,
                            "requires_external_confirmation": True,
                        }
                    ),
                ),
            )
            for stance in ("proceed", "defer", "seek_observation")
        )
        reflection = FrozenJsonDict(
            {
                "summary": "deterministic structured local reflection",
                "input_keys": sorted(frozen.keys()),
                "epistemic_status": "counterfactual_not_observed",
            }
        )
        return CognitionResult(
            source_ids=sources,
            input_fingerprint=fingerprint,
            alternatives=alternatives,
            uncertainty=uncertainty,
            reflection=reflection,
        )


def cognition_result_from_payload(payload: Mapping[str, Any]) -> CognitionResult:
    """Rebuild a strict result from event JSON containers."""
    alternatives = tuple(
        CognitionAlternative(
            branch_id=item["branch_id"],
            stance=item["stance"],
            content=item["content"],
            expected_consequences=tuple(
                FrozenJsonDict(value) for value in item["expected_consequences"]
            ),
        )
        for item in payload["alternatives"]
    )
    return CognitionResult(
        cognition_mode=payload.get("cognition_mode", "local"),
        provider_used=payload.get("provider_used", False),
        algorithm_version=payload["algorithm_version"],
        config_version=payload["config_version"],
        source_ids=tuple(payload["source_ids"]),
        input_fingerprint=payload["input_fingerprint"],
        alternatives=alternatives,
        uncertainty=float(payload["uncertainty"]),
        uncertainty_basis=payload.get("uncertainty_basis", "deterministic_heuristic"),
        calibrated=payload.get("calibrated", False),
        reflection=payload["reflection"],
    )


def result_payload(result: CognitionResult) -> dict[str, Any]:
    return json.loads(result.model_dump_json())


__all__ = [
    "COGNITION_ALGORITHM_VERSION",
    "COGNITION_CONFIG_VERSION",
    "CognitionAlternative",
    "CognitionResult",
    "CognitionState",
    "LocalCognitionPort",
    "cognition_result_from_payload",
    "result_payload",
]
