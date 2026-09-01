# AgroCopilot documentation

Technical documentation for AgroCopilot `0.1.0`.

## Documentation map

| Document | Audience | Purpose |
| --- | --- | --- |
| [Architecture](architecture.md) | Technical reviewers and developers | Components, request lifecycle, data boundaries and deployment model |
| [API reference](api.md) | API consumers and frontend developers | HTTP endpoints, request contracts, SSE and error behaviour |
| [Configuration](configuration.md) | Developers and operators | Environment variables, local profiles, persistence and external services |
| [Agent contracts](agent_prompt_contracts.md) | Agent and evaluation developers | Responsibilities, inputs, outputs, routing constraints and failure rules |
| [Prompting](prompting.md) | Agent developers | Prompt loading, context reduction, structured output and change procedure |
| [Runtime behaviour](multi-agent-runtime-notes.md) | Maintainers and reviewers | Fast path, concurrency, replanning, evidence handling and operational limits |
| [Evaluation](../evaluation/README.md) | Technical reviewers and evaluation developers | Final benchmark protocol, execution and statistical analysis |
| [Final benchmark configuration](../evaluation/configs/benchmark_final.json) | Technical reviewers | Six configurations, fixed corpus, judges and reproducibility parameters |
| [Benchmark statistics](../evaluation/analyze_benchmark_statistics.py) | Technical reviewers | Canonical inferential analysis for the dissertation benchmark |
| [Responsible use](../DISCLAIMER.md) | All users | Agronomic, regulatory and data-use limitations |

## Implementation references

| Area | Implementation |
| --- | --- |
| HTTP routes and request models | `backend/api.py` |
| Orchestration | `backend/services/chat_orchestrator.py` |
| Runtime schemas | `libs/schemas.py` |
| Runtime agent registry | `backend/api.py` |
| Agent catalogue and routing | `agents/organizer.py` |
| Settings and defaults | `backend/deps.py` |
| Example environment | `.env.example` |
| Frontend scripts | `ui-web/package.json` |
| CI commands | `.github/workflows/ci.yml` |

## Scope

AgroCopilot is a validated local prototype accompanying a Master's dissertation. The repository supports a FastAPI service, a React client, local persistence, optional model and search providers, meteorological context, STAC-based remote sensing, specialist attachment processing and a reproducible evaluation framework.

The runtime does not provide production-grade authentication, tenant isolation, shared attachment storage, a durable distributed event broker or operational service-level guarantees.

## Usage

- Paths are relative to the repository root unless stated otherwise.
- Examples assume commands are run from the repository root.
- Any example that can contact an external service is marked explicitly.

## Validation

```bash
python -m pytest tests tests_evaluation -q

cd ui-web
npm test
npm run typecheck
npm run build
```

The canonical dissertation benchmark is defined by `evaluation/configs/benchmark_final.json` and analysed by `evaluation/analyze_benchmark_statistics.py`. Route, schema, setting and agent changes require matching documentation updates.
