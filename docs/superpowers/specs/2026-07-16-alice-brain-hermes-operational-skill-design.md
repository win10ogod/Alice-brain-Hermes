# Alice-brain-Hermes Operational Skill Design

## Goal

Add one repository-owned Agent Skill that guides an agent through exactly five Alice-brain-Hermes operations: start, observe, diagnose, verify, and trace the decision process.

## Artifact

Create `.agents/skills/operating-alice-brain-hermes/` with only:

- `SKILL.md`
- `agents/openai.yaml`

Do not add scripts, references, assets, or auxiliary Skill documentation.

## Workflow Contract

The Skill must:

1. Start the plugin runtime and confirm readiness.
2. Observe current runtime and integration state without changing it.
3. Diagnose runtime health and preserve the command's actual exit result.
4. Verify claims with fresh status, diagnostic, and trace evidence.
5. Trace recent decision records and report identifiers needed for follow-up.

Use the repository's current native and Hermes CLI help as the source of truth. Never invent a command, imply that a read-only command starts the daemon, or treat a skipped real-host check as verification.

## Content Boundary

The Skill contains only the operational workflow above. It contains no consciousness theory, personality or trust narrative, policy discussion, project history, development workflow, release workflow, or general product documentation.

## Validation

- Record a fresh-agent baseline without the Skill.
- Validate the Skill structure and metadata.
- Forward-test the same operational request with the Skill.
- Confirm that the result stays within the five operations and uses valid Alice-brain-Hermes or Hermes-native commands.
