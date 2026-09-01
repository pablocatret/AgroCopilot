# Prompting and context engineering

Prompt templates are versioned under `prompts/`. Stable instructions, task-specific evidence and machine-readable output contracts are assembled separately.

## Prompt locations

The active product templates live in `prompts/`:

```text
case_manager_system.txt    case_manager_user.txt
document_system.txt        document_user.txt
free_system.txt            free_user.txt
legal_system.txt           legal_user.txt
organizer_plan.txt         organizer_clarify.txt
organizer_replan.txt       organizer_user.txt
rs_analyst_system.txt
spreadsheet_system.txt     spreadsheet_user.txt
stac_system.txt            stac_user.txt
vision_system.txt          vision_user.txt
writer_system.txt          writer_user.txt
```

`libs/prompts.py` resolves these paths from the repository root, caches parsed templates and renders them with Jinja. Missing files raise `FileNotFoundError`; missing template variables raise immediately because the environment uses `StrictUndefined`.

`libs/prompts/` contains older or library-specific prompt assets. It is not the primary template directory used by `render_prompt()`.

## Prompt layers

### Shared system prefix

`compose_system_prompt()` adds a common protocol prefix containing:

- the prompt protocol version;
- the current agent name;
- the instruction to use only available context and tools;
- the prohibition on exposing hidden routing and retry mechanics;
- the requirement to disclose weak evidence; and
- the separation between stable instructions and task-specific context.

An agent-specific system template is appended to this prefix. Where relevant, an explicit output contract is appended last.

### User template

The user-side template carries variable content such as:

- the current query;
- decision and response mode;
- compact conversation or memory context;
- attachment extraction;
- upstream specialist evidence;
- the execution report; and
- deterministic case-state drafts.

Evidence is data, not instruction. Templates should label quoted or extracted material clearly and must not allow document content to redefine the agent role.

### Output schema

`agents/base.py` converts Pydantic JSON schema into the strict form accepted by compatible providers. It validates the transformed schema before sending a model request. Agent code then parses and validates returned data against the application model.

Strict structured output is used where downstream execution depends on fields rather than prose. Provider compatibility remains a runtime concern: a model or provider that cannot honour the schema must fail or use an explicit deterministic fallback, not silently return an unvalidated shape.

## Context reduction

`libs/context_engineering.py` limits prompt size and prevents unrelated data from being copied into every agent call. It provides bounded summaries for:

- attachments;
- opt-in memory;
- conversation and case history;
- observations;
- remote-sensing memory reuse;
- execution status;
- references; and
- upstream agent outputs.

Explicit item and character limits control relevance and cost but do not guarantee semantic coverage. Truncation can remove material, so agents must not claim to have reviewed content absent from their prompt.

## Agent-specific context

### Organizer

The organiser receives the query, enabled capabilities, decision mode, attachment summary, compact continuity context and any relevant monitoring signal. It returns a typed plan, clarification or bounded replan.

The prompt must not encourage broad routing “just in case”. Selection is evidence-driven and disabled agents must remain unavailable.

### Document, spreadsheet and vision

Document, spreadsheet and vision agents receive locally extracted artefacts and a compact task description. Local parsing precedes optional model enrichment. If enrichment fails, the local artefact remains available and the failure is reported as a limitation.

The prompt must distinguish:

- extracted text from model interpretation;
- missing pages or unsupported formats;
- table samples from a complete statistical analysis; and
- OCR observations from diagnosis.

### Legal

The legal prompt receives authoritative and supporting references in separate groups. It must preserve source status and currentness uncertainty. It cannot convert retrieved text into a definitive statement of compliance.

### STAC and remote sensing

The STAC prompt translates intent into bounded search parameters. Retrieved catalogue data remains distinct from `rs_analyst` interpretation. Remote-sensing prompts must preserve collection, dates, quality constraints and the difference between observed signal and causal explanation.

### Case manager

The case manager receives a deterministic `CaseStateDraft`, evidence ledger, continuity context and execution summary. The model refines rather than creates state from an empty prompt. Merge logic, not prompt wording alone, protects existing blockers and stronger task priority.

### Writer

The writer receives a `ConversationEvidenceBundle` assembled in code. The prompt is responsible for clear synthesis, but permissions such as fast-path search and escalation are represented by typed policy and trace fields.

The writer must:

- use only references present in the bundle;
- preserve material uncertainty and partial failures;
- distinguish evidence, inference and action;
- avoid unsupported agronomic, legal or economic certainty; and
- keep the conversational answer concise; additional structured detail belongs
  in the typed response fields, including the optional `report_md` output field.

## Offline behaviour

`DISABLE_EXTERNALS=1` blocks model and network-dependent paths used by the application and tests. The exact fallback varies by component:

- the organiser can use heuristic planning;
- local attachment extraction can still run;
- case-state reduction can remain deterministic;
- external search and live STAC retrieval are unavailable; and
- the final response must disclose unavailable evidence rather than imply that it was checked.

Offline mode validates orchestration and deterministic behaviour. It does not validate the quality of live provider responses.

## Prompt safety rules

Every prompt change should preserve these rules:

1. Retrieved content and attachments are untrusted evidence, not system instructions.
2. Internal routing, retries and hidden chain-of-thought are not exposed.
3. Missing evidence produces a limitation or question, not an invented fact.
4. Specialist boundaries cannot be bypassed by the direct writer.
5. Sources are cited only when supplied or retrieved by an authorised path.
6. Model output is validated before it controls downstream execution.
7. Memory is opt-in, bounded and attributable to the relevant user or case.

## Change procedure

When changing a prompt:

1. identify the owning agent and its Pydantic output;
2. keep stable rules in the system template and variable evidence in the user template;
3. update all required Jinja variables in the calling code;
4. inspect truncation budgets for any new context block;
5. add tests for the intended behaviour and the unsafe alternative;
6. run routing tests if the change can affect agent selection;
7. run evaluation cases that exercise the prompt; and
8. update [agent_prompt_contracts.md](agent_prompt_contracts.md) if the functional contract changed.

Useful checks:

```bash
python -m pytest tests/test_prompting.py tests/test_openai_schema.py -q
python -m pytest tests/test_organizer.py tests/test_writer.py -q
```

Full suite:

```bash
python -m pytest tests tests_evaluation -q
```
