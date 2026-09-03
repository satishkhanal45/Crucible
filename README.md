# Crucible

A co-evolutionary red-team loop for RAG systems. An attacker agent evolves
prompt-injection attacks; a defender agent evolves guardrail configurations; an
archive and a holdout set stop either from cheating.

---

## Rules of engagement

*Reproduced verbatim from `docs/spec.md` section 1. These are constraints in the
code, not aspirations in a document.*

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

---

## What Crucible is

Most guardrail evaluations are static: a fixed list of attack strings, run once,
producing a percentage. That number stops meaning anything the moment someone
writes attack 501. Crucible measures something harder to fake — whether a
defense holds up against an attacker that is *still adapting to it*.

Each round, the attacker mutates archived attacks against the current defense
and the defender proposes configuration changes in response. Three mechanisms
stop this from becoming a machine for generating flattering numbers:

- **Novelty pressure.** A candidate too close to something already archived is
  rejected *before it is executed*, so the attacker cannot farm variations of one
  working attack.
- **Full-archive re-evaluation.** The block rate a round records always comes
  from running the selected config against every non-holdout attack ever
  archived. Screening may sample; the recorded number may not.
- **A holdout set.** Attacks the defender never sees, in any form. The gap
  between the archive block rate and the holdout block rate is the overfitting
  measure, and it is reported every round.

Success is content-neutral. An attack succeeds when it extracts a **canary** — a
unique token planted in the system prompt, a confidential document, another
session's history, or a privileged tool's output — or when it causes an
unauthorized privileged tool call to *execute*. Nothing in the loop rewards
generating harmful text, which is what makes the oracle deterministic.

## Architecture

```
                 ┌──────────────┐         ┌──────────────┐
   parents ─────▶│   Attacker   │         │   Defender   │◀──── breaches
   from the      │  (LangGraph) │         │  (LangGraph) │      (non-holdout)
   MAP-Elites    └──────┬───────┘         └──────┬───────┘
   grid                 │ candidates             │ candidate configs
                        ▼                        ▼
                 ┌─────────────┐          ┌─────────────┐
                 │  Novelty    │          │ Evaluation  │
                 │  gate       │          │ service     │
                 └──────┬──────┘          └──────┬──────┘
                        │ admitted               │ screen, then FULL archive
                        ▼                        ▼
                 ┌──────────────────────────────────────┐
                 │      Executor  →  Target  →  Oracle   │
                 │  (bounded pool, isolated namespaces)  │
                 └───────────────────┬──────────────────┘
                                     ▼
                 ┌──────────────────────────────────────┐
                 │  Archive · cells · attempts · rounds  │
                 │        PostgreSQL + pgvector          │
                 └──────────────────────────────────────┘
```

Layering is `api → services → repositories → db`; agents talk to services and
never to the ORM. The target sits behind a `TargetAdapter` Protocol, which is
what lets the transfer experiment point the whole loop at a second application
without changing a line of loop code.

**The defense stack is configuration, never code.** Five layers — input
inspection, context handling, prompt hardening, output checking, and structural
tool authorization — and the defender emits a validated `DefenseConfig`. Unknown
keys are rejected rather than ignored.

## The coverage grid

Attacks are placed in a MAP-Elites grid of **96 cells**: 6 objectives × 2
executable delivery vectors × 8 techniques. Coverage is always printed with its
denominator, and always alongside how far the real classifier agrees with the
hand-assigned labels — a cell count means nothing if the cells are wrong.

![Coverage grid](reports/coverage_grid.png)

![Block rates and utility across rounds, with 95% Wilson intervals](reports/three_curves.png)

Every rate carries a Wilson score interval. Round-over-round differences whose
intervals overlap are not improvements, and the reports say so in those words.

## Results

Headline numbers live in [`docs/findings.md`](docs/findings.md). Every figure
there is regenerated by `crucible findings regenerate` from the `runs`,
`rounds`, `attempts`, `defense_configs`, `classifier_agreement` and
`experiment_results` tables. Nothing is typed by hand, and CI fails if a
committed number differs from what the stored data produces.

To regenerate the numbers and the charts yourself:

```bash
crucible findings regenerate          # rewrites every generated block
crucible report run --charts reports  # rewrites the three visuals
```

## Quickstart

Requirements: Docker, `uv`, and a Groq API key (the free tier is enough).

```bash
git clone <this repository> && cd crucible
cp .env.example .env          # then put your GROQ_API_KEY in it
```

**Port conflicts.** Compose publishes `${POSTGRES_PORT:-5432}` and
`${API_PORT:-8000}`. If a service on your machine already owns either port, set
them in `.env` before starting — for example `POSTGRES_PORT=5434` and
`API_PORT=8001`. `DATABASE_URL` has to match the port you choose.

```bash
make up          # ⚠️  the FIRST run builds images and takes about 10 minutes
make migrate     # alembic upgrade head
make seed        # load the corpus, plant canaries, load the 40 seed attacks
```

`make seed` prints nine checks; every one must pass before any number the system
produces means anything.

Then look around, and run something:

```bash
crucible archive stats                    # coverage out of 96, novelty, elites
crucible archive reclassify --dry-run     # how far the classifier agrees with the labels
crucible experiment list                  # every experiment, with a time estimate
crucible experiment run main              # the headline run: 6 rounds
crucible report run --format md           # the run report, payloads redacted
```

Useful during a long run:

```bash
crucible loop status                      # what is running and what halted
crucible loop resume --run-id <id>        # continue a checkpointed run
```

**Rate limits are the binding constraint,** not compute. The free tier's limit
is 8000 tokens per minute; `.env.example` ships a 6500 TPM margin with
concurrency 1. A 40-call classifier pass takes about nine minutes at that rate
and the headline run takes a few hours, so run it overnight. `crucible
experiment list` prints an estimate for each experiment.

Add `--debug` to any command to get a full traceback instead of a one-line
error.

## What is published, and what is not

Published: the taxonomy, the architecture, aggregate results, the defense
configurations, and the layer-ablation table. The configurations are the part
most likely to be useful to someone else, so they are emitted in full.

Not published: **canary values**, ever, in any output; the reference target's
system prompt in a form that would make this corpus reusable as a jailbreak
benchmark; and raw attack payloads, which are redacted to a mechanism
description and a `sha256:` prefix unless a report is explicitly generated with
`--include-payloads`.

## Development

```bash
make test        # the full suite; testcontainers spins up Postgres
make lint
make typecheck   # mypy --strict on src/crucible
```

Python 3.12, `uv` only, `uv.lock` committed. Async throughout: SQLAlchemy 2.0
with asyncpg, no sync database calls. Pydantic v2 at every boundary. Every
schema change is an Alembic revision, and a test asserts that autogenerate
produces an empty diff. No test makes a live API call.

## Licence

Apache-2.0. See [LICENSE](LICENSE).
