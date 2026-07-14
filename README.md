# Alice-brain-Hermes

Alice-brain-Hermes is a self-contained engineering-consciousness project for
Hermes Agent. It owns its package namespace and, as later implementation stages
land, will own its runtime, daemon, event ledger, credentials, state, and CLI.
It neither imports nor connects to the separate Alice-brain project.

This repository treats PC/E/ST/RD/A transitions, continuous processing,
self/world models, metacognition, and grounded action receipts as observable
engineering mechanisms. Passing their contracts is evidence about those
mechanisms; it is not presented as proof of phenomenal consciousness.

The current bootstrap exposes package identity and an executable independence
audit. Runtime and Hermes hook behavior are implemented in later, separately
tested stages.

## Development

```console
uv sync --extra dev
uv build
uv run python scripts/check_independence.py dist/*.whl
uv run pytest
```

Python 3.11 through 3.13 is supported.
