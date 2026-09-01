# Agent contracts

AgroCopilot presents one product interface while retaining separate internal responsibilities. Users do not choose agents directly. `organizer` selects a route, the orchestrator enforces dependencies, and the writer produces the visible answer.

Executable schemas are defined in `libs/schemas.py`.

## Shared input and output

Most specialists implement `BaseAgent.run(AgentInput)` and return a model derived from `BaseAgentOutput`.

Common input includes:

- the user query and requested language;
- conversation and user identifiers;
- decision mode and the currently supported response mode;
- relevant attachment metadata;
- compact memory, case and execution context; and
- upstream agent outputs required by the dependency graph.

Common output includes:

- `summary`: a concise result;
- `refs`: traceable references;
- `limitations`: missing data, uncertainty and degradation;
- `trace`: execution metadata; and
- an agent-specific `data` payload.

An agent must return a limitation or typed failure when it cannot establish a result. It must not fill a missing field with an invented value merely to satisfy the schema.

## Runtime catalogue

| Agent | Role | Output | Must not |
| --- | --- | --- | --- |
| `organizer` | Selects the smallest valid plan, missions and dependencies | `AgentPlan` or `ClarificationRequest` | Route disabled agents, expose internal controls to the user, or add specialists without evidence need |
| `document_analyst` | Extracts and analyses PDF, DOCX, TXT and compatible document content | `DocumentAgentOutput` | Treat partial text as a complete document or claim unsupported verification |
| `spreadsheet_analyst` | Profiles CSV, XLS and XLSX attachments | `SpreadsheetAgentOutput` | Extrapolate a robust trend from insufficient rows or conceal parsing failures |
| `vision_ocr` | Extracts OCR text and bounded visual observations | `VisionAgentOutput` | Invent unreadable text or turn a visual signal into a definitive diagnosis |
| `legal` | Retrieves and synthesises regulatory evidence | legal findings, authoritative and supporting references | Declare definitive compliance or promote a non-official source to authoritative evidence |
| `stac` | Searches configured STAC collections and prepares scene data | `StacAgentOutput` / `StacResults` | Invent coordinates, acquisition dates, assets or catalogue results |
| `rs_analyst` | Interprets scenes, indices and temporal comparisons | remote-sensing findings and temporal comparison | Infer agronomic causality from spectral or radar evidence alone |
| `free` | Handles bounded general research outside specialist domains; may use configured web search for current or factual information | `FreeAgentOutput` | Override a specialist boundary, invent findings, or present unverified search material as established fact |
| `case_manager` | Reduces evidence into case state, tasks and blockers | `CaseManagerAgentOutput` / `CaseState` | Invent user history, field observations or completed work |
| `direct_writer` | Implements the visible single-agent response path | `WriterAgentOutput` / `FinalAnswer` | Use specialist claims without escalation or hide whether bounded search was used |
| `writer` | Planner-facing name for multi-agent synthesis | `WriterAgentOutput` / `FinalAnswer` | Discard material limitations, contradictions or partial failures |

`stac` and `rs_analyst` are available only when their corresponding settings are enabled. The orchestrator treats `organizer`, `writer` and `direct_writer` as internal coordination roles rather than ordinary parallel specialists.

### Free research agent

`free` is the general-purpose research route for questions that do not belong to a specialist domain. The organiser assigns it a bounded task; it can combine model reasoning with the configured web-search tool and returns findings, source references, confidence and limitations. It must not replace the legal, remote-sensing, document, spreadsheet or vision agents when those domains are relevant.

Web search is optional at configuration level. A live run requires an enabled external model, an API key for the selected model provider and `SEARCH_API_KEY` for the configured search provider. With external services disabled, the agent returns a typed low-confidence limitation rather than fabricating research results.

## Planning contract

`AgentPlan` contains:

- `steps`: ordered agent names;
- `missions`: per-agent instructions;
- `runs`: requested instance counts;
- `dependencies`: upstream requirements;
- `allow_replan`: whether one bounded evidence-gap review is permitted;
- `writer_mode`: `BRIEFING`, `STANDARD` or `DEEP_DIVE`;
- `response_mode`: `conversation` (the only currently supported API value);
- `policy`: retries, rounds and writer fast-path permissions;
- `diagnostics`: planner source and fallback rationale; and
- an optional clarification request.

The organiser should choose the minimum route that can answer safely:

- a simple non-specialist question may use the writer fast path;
- attachments require the matching extraction or analysis capability;
- legal or compliance questions require `legal`;
- satellite retrieval requires `stac`;
- satellite interpretation requires `rs_analyst` after `stac`; and
- case continuity uses `case_manager` when state should be created or updated.

An empty plan is valid only when it carries a clarification request or a justified stop condition.

## Dependency rules

Typical dependencies are:

```text
document_analyst ─┐
spreadsheet_analyst ─┼─> case_manager ─> writer
vision_ocr ───────┤
legal ────────────┤
stac ─> rs_analyst ┘
```

The actual graph is plan-specific. Independent ready nodes may run concurrently. Downstream agents receive only completed upstream context; failed upstream runs remain visible through the execution report and limitations.

## Attachment boundaries

The API accepts at most six files per upload request and a maximum of 10 MiB per file. Supported filename extensions are:

```text
.csv .doc .docx .jpeg .jpg .pdf .png .tif .tiff .txt .xls .xlsx
```

Format behaviour:

- legacy `.doc` is accepted at the API boundary but local extraction reports it as unsupported;
- document and OCR text may be cached in attachment metadata;
- CSV and spreadsheet analysis use a dedicated tabular path;
- local extraction can remain usable when optional LLM enrichment fails; and
- a requested attachment identifier that does not exist causes `POST /chat` to return HTTP 400.

## Legal evidence contract

Legal retrieval separates:

- **authoritative references**: current material from official-looking sources suitable for grounding; and
- **supporting references**: explanatory, historical or lower-authority material.

The legal agent must preserve this distinction. An empty authoritative set remains empty; supporting material is not silently promoted. Currentness checks may require external verification, so offline results must disclose that constraint.

## Remote-sensing contract

`stac` retrieves candidate scenes and assets. `rs_analyst` interprets compatible results. The analysis must preserve:

- collection and acquisition date;
- area or bounding-box limitations;
- cloud, mask or scene-quality constraints where available;
- the index or radar quantity used;
- whether a comparison is retrieval-only, partial or temporal; and
- the distinction between observed change and an agronomic explanation.

Temporal pair selection occurs before downstream result truncation so that the best eligible comparison is not discarded merely because it is older than the most recent scene.

## Case-state contract

`case_manager` starts from a deterministic `CaseStateDraft` produced from usable evidence and continuity cues. Optional model refinement may improve wording and structure, but merge logic preserves stronger task priority, blocked status, recommended inputs and limitations.

The resulting state may include:

- current summary;
- tasks and priorities;
- blockers;
- recommended inputs;
- field observations;
- evidence grouped by modality; and
- decisions or state changes.

Explicit cases in `data/cases.db` are authoritative. Opt-in Markdown memory is a separate compatibility and user-context mechanism.

## Writer contract

Every visible answer is represented by `FinalAnswer`. `message_md` contains the main narrative, while structured fields expose:

- executive summary;
- response path;
- search and escalation status;
- references and evidence summary;
- recommendations and next actions;
- missing information and required documents;
- limitations;
- case state and continuity;
- temporal comparison and remote-sensing data;
- execution and cost summaries; and
- optional `report_md` Markdown output. This field does not imply a separate
  selectable `report` request mode.

The writer must:

1. distinguish evidence, inference and recommendation;
2. retain material contradictions and partial failures;
3. state uncertainty at the point where it affects a decision;
4. avoid unsupported doses, diagnoses, legal conclusions and financial certainty;
5. cite sources actually present in the evidence bundle; and
6. escalate when a fast-path question crosses a specialist boundary.

## Failure and degradation

The runtime supports partial success. A specialist failure does not automatically invalidate unrelated evidence. The orchestrator records each instance as usable, partial or failed, and the writer explains material gaps.

Safe degradation means:

- no fabricated substitute result;
- no silent removal of a relevant failure;
- usable local extraction is retained;
- a missing specialist result becomes a limitation or requested input; and
- the final answer remains explicit about what could not be checked.

## Testing expectations

Agent changes should include:

- schema and strict-output tests;
- unit tests for deterministic behaviour;
- routing tests proving both selection and non-selection;
- orchestration tests for dependencies and partial failure;
- safety and limitation assertions; and
- at least one representative case in `evaluation/cases/`.

See [prompting.md](prompting.md) for prompt construction and [multi-agent-runtime-notes.md](multi-agent-runtime-notes.md) for execution behaviour.
