# Multi-agent runtime behaviour

## Execution model

`ChatOrchestratorService` receives a validated request and an agent registry. The organiser returns an `AgentPlan` containing steps, missions, dependencies, run counts and policy. The orchestrator validates the names against the registry before scheduling work.

An agent instance becomes runnable when all declared upstream dependencies have completed. Independent ready agents may run concurrently up to `AGENT_CONCURRENCY_LIMIT`, which defaults to four.

The runtime records each instance separately and then derives an aggregate level for the agent. A run can be usable, partial or failed. The evidence bundle and final execution metadata retain that classification.

## Single-agent fast path

Simple questions may use the planner-facing `writer` step without specialists. The registered `direct_writer` implementation handles this path.

Fast-path permissions are explicit:

- `plan.policy.fast_path.enabled`
- `allow_search`
- `disclose_search_use`
- `disclose_sources`
- `escalate_when_specialized`

The answer exposes the effective behaviour through `answer.fast_path` and the top-level response-path fields. A bounded search may run only when the policy allows it and the query meets the writer's search criteria.

Attachments, legal interpretation, remote sensing and temporal comparison are escalation signals. The direct writer must not simulate a specialist answer when those capabilities are required.

## Multi-agent synthesis

When specialists are selected, the writer receives an evidence bundle assembled by application code. Narrative generation and evidence assembly are separate:

1. specialist outputs, references and limitations are normalised;
2. execution state and partial failures are added;
3. case, memory and attachment context is compacted;
4. the effective policy and search trace are attached; and
5. the writer produces `FinalAnswer`.

The evidence bundle can be inspected independently of the final prose.

## Retries

Retry behaviour is plan-controlled through:

- `allow_retries`;
- `max_rounds`; and
- `retry_candidates`.

Retries are not a general loop around every failed agent. Only eligible agents may be retried, and the orchestrator preserves the failure history of earlier instances. A retry does not erase the original limitation.

## Replanning

Replanning is considered only when:

- `plan.allow_replan` is true;
- the organiser exposes a replan method; and
- accumulated cost is below `REPLAN_MAX_COST_USD`.

The organiser receives the current execution state and can add bounded steps or justify stopping. The runtime exposes replan metadata even when no step is added, including whether replanning was attempted, which diagnostics were returned and whether extra steps were applied.

If a model replan returns no useful step without a justified stop, a narrow heuristic may re-add an obvious specialist only when the execution report shows a real evidence gap. It does not expand an already adequate plan.

## Clarification

The organiser may return a `ClarificationRequest` instead of executable steps. The API then returns:

- the empty or non-executable plan;
- the clarification question and options; and
- the conversation identifier.

The frontend can submit an option's enriched query as a new chat request. Clarification is preferable to guessing a missing parcel, time period or decision objective.

## Attachment extraction and reuse

`AttachmentStore` persists the file, metadata and any local extraction. `metadata.extraction` can include extractor identity, confidence, warnings and page or format hints.

Downstream agents hydrate cached extraction lazily to avoid repeating OCR or document parsing:

- PDF, DOCX, text and supported image formats can produce local extraction;
- legacy `.doc` remains upload-compatible but is marked unsupported for local extraction;
- CSV and spreadsheet profiling use the tabular agent; and
- a failed model-enrichment step does not discard successful local extraction.

Attachment identifiers are resolved before orchestration. Missing requested identifiers fail the chat request rather than being silently ignored.

## Legal retrieval

The legal pipeline retains two evidence classes:

- authoritative references suitable for grounding; and
- supporting references useful for context or contrast.

Source status depends on currentness and an official-looking domain, not merely on a corpus label or the presence of a URL. Historical years alone do not force a live currentness check. Queries that explicitly require current rules, or evidence sets of insufficient quality, may trigger external verification when external calls are enabled.

If no authoritative source is found, the authoritative collection remains empty and the final answer must state the limitation.

## STAC and temporal comparison

STAC search produces a deduplicated candidate pool. Temporal pair selection evaluates that pool before downstream truncation, preserving an older baseline when it forms the better comparison.

`rs_analyst` interprets retrieved scenes only when the required inputs are compatible. Results preserve collection, dates, metric, confidence and limitations. Sentinel-1 radar signals and land-cover products are treated as contextual or auxiliary evidence rather than interchangeable optical time-series observations.

Remote-sensing memory reuse is typed as a hit, stale result or miss. Retrieval-only artefacts are not reused as if they contained completed temporal analysis.

## Case continuity

Case continuity uses two stages:

1. deterministic reduction creates `CaseStateDraft` from evidence summaries, temporal changes and continuity cues;
2. optional model refinement improves the draft.

The merge is a reconciliation, not replacement. Tasks are matched by title; stronger priority and blocked state survive; recommended inputs and blockers are normalised and combined.

The evidence ledger retains per-modality usable, partial, failed and missing counts. These counts prevent one summary string from masking a successful document result and a failed remote-sensing result.

Explicit cases, assertions, tasks and observations are stored transactionally in `data/cases.db`. Opt-in memory maintains a separate user-context model and legacy mirrors.

## Events and replay

`EventBroker` provides one process-local fan-out state per conversation and retains up to 100 recent event payloads in memory.

`SSE_LOG_MODE` controls auxiliary event persistence:

- `memory`: process-local replay only;
- `stdout`: emitted and delivered event records are written to standard output;
- `disk`: conversation events are appended to local JSONL files and can be replayed after the in-memory state is recreated.

The broker cleans a completed conversation after its last subscriber disconnects. Disk replay improves local resilience but does not make the broker safe for multiple API workers. A distributed deployment requires a shared stream or broker.

`SSE_TRACE=true` enables additional trace capture for emitted and delivered events. It should be treated as diagnostic data and reviewed for retention and privacy before non-local use.

## Cost tracking

Cost events are written to `data/costs.db` when `COST_TRACKING_ENABLED=true`. Estimates are derived from the configured pricing catalogue and recorded usage; they are not provider invoices.

The runtime uses cost thresholds for warnings and for the replanning boundary. Cost tracking does not enforce a hard provider budget.

## Failure semantics

The runtime favours explicit degradation:

- a missing attachment is an HTTP error;
- an unknown plan step is rejected;
- an agent exception becomes failed execution metadata;
- locally usable evidence survives optional enrichment failure;
- unrelated successful agents remain usable;
- a material gap is included in `limitations` or `missing_information`; and
- completion events are published after persistence and response assembly.

Successful HTTP completion therefore does not imply that every specialist succeeded. Consumers should inspect `answer.execution`, `answer.limitations` and `answer.evidence_ledger`.

## Operational constraints

The current implementation assumes:

- one trusted local user or a controlled demonstration environment;
- one API process for consistent SSE delivery;
- local filesystem access;
- SQLite-compatible write volume; and
- no hostile tenant sharing the runtime.

See [architecture.md](architecture.md) for the deployment boundary and [configuration.md](configuration.md) for runtime settings.
