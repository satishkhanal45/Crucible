# Crucible

A co-evolutionary red-team loop for RAG systems. An attacker agent evolves
prompt-injection attacks; a defender agent evolves guardrail configurations; an
archive plus a held-out attack set stop either from cheating.

## Rules of engagement

Crucible attacks **its own application-layer guardrails**.

**In scope**

- Direct and indirect prompt injection against a RAG application we own and run locally.
- System-prompt extraction, retrieved-context poisoning, tool-call hijacking, output-format subversion.
- Measuring whether *our* defenses hold.

**Out of scope, enforced as constraints, not documentation**

- Any target not on `TARGET_ALLOWLIST`. The executor blocks all other network egress and raises `EgressViolation`.
- Eliciting genuinely harmful content as a success criterion. Success is canary leakage and unauthorized tool calls, which are content-neutral. The oracle design removes the temptation.
- Attacks on the model provider's own safety training. We test our application layer.
- Publishing a raw payload library. Payloads live behind `--include-payloads`.

## Status

Phase 0 — Foundation. See `CLAUDE.md` for conventions and the build order,
and `docs/spec.md` for the authoritative scope of this build.

## Quickstart

```bash
cp .env.example .env    # fill in provider API keys
make up                 # postgres + api
make migrate            # alembic upgrade head
curl localhost:8000/health
curl localhost:8000/ready
```

## Development

```bash
uv sync --all-groups
make lint typecheck test
```

## License

Apache-2.0.
