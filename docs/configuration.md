# Configuration

AgroCopilot reads backend settings through `pydantic-settings` in `backend/deps.py`. The backend loads `.env` from the repository root. The Vite client reads variables prefixed with `VITE_`.

Start from:

```bash
cp .env.example .env
```

On Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

Do not commit `.env` or real provider credentials.

## Recommended profiles

### Offline inspection and tests

```dotenv
DISABLE_EXTERNALS=1
OCR_BACKEND=none
VITE_API_BASE=http://localhost:8000
CORS_ORIGINS=http://localhost:5173
```

Use this profile for repository review. It exercises local parsing, persistence, orchestration fallbacks and the user interface without live model, search or STAC results.

### OpenAI-backed local run

```dotenv
DISABLE_EXTERNALS=0
LLM_PROVIDER=openai
OPENAI_API_KEY=replace-me
VITE_API_BASE=http://localhost:8000
CORS_ORIGINS=http://localhost:5173
```

Add search, STAC or geocoding configuration only when those capabilities are required.

### OpenRouter-backed local run

```dotenv
DISABLE_EXTERNALS=0
LLM_PROVIDER=openrouter
OPENROUTER_API_KEY=replace-me
OPENROUTER_APP_URL=http://localhost:5173
OPENROUTER_APP_TITLE=AgroCopilot
```

Per-agent provider overrides can mix OpenAI and OpenRouter, provided both required credentials are available.

## Model providers

| Variable | Default | Purpose |
| --- | --- | --- |
| `LLM_PROVIDER` | `openai` | Global provider: `openai` or `openrouter` |
| `OPENAI_API_KEY` | empty | Required for OpenAI calls |
| `OPENROUTER_API_KEY` | empty | Required for OpenRouter calls |
| `LLM_BASE_URL_OPENAI` | OpenAI API URL | OpenAI-compatible base URL |
| `LLM_BASE_URL_OPENROUTER` | OpenRouter API URL | OpenRouter base URL |
| `OPENROUTER_APP_URL` | empty | Optional attribution header |
| `OPENROUTER_APP_TITLE` | empty | Optional attribution header |

Supported per-agent provider overrides are:

```text
LLM_PROVIDER_ORGANIZER
LLM_PROVIDER_WRITER
LLM_PROVIDER_LEGAL
LLM_PROVIDER_STAC
LLM_PROVIDER_CASE_MANAGER
LLM_PROVIDER_DOCUMENT_ANALYST
LLM_PROVIDER_SPREADSHEET_ANALYST
LLM_PROVIDER_VISION_OCR
LLM_PROVIDER_FREE
```

## Models

The active settings support:

| Variable | Default |
| --- | --- |
| `OPENAI_MODEL_BASELINE` | `gpt-5-nano` |
| `OPENAI_MODEL_ORGANIZER` | `gpt-5-mini` |
| `OPENAI_MODEL_WRITER` | `gpt-5-mini` |
| `OPENAI_MODEL_STAC` | `gpt-5-mini` |
| `OPENAI_MODEL_LEGAL_WRITER` | `gpt-5-mini` |
| `OPENAI_MODEL_LEGAL` | `gpt-5-mini` |
| `OPENAI_MODEL_CASE_MANAGER` | `gpt-5-mini` |
| `OPENAI_MODEL_DOCUMENT_ANALYST` | `gpt-5-nano` |
| `OPENAI_MODEL_SPREADSHEET_ANALYST` | `gpt-5-nano` |
| `OPENAI_MODEL_VISION` | `gpt-4o-mini` |
| `OPENAI_MODEL_VISION_OCR` | `gpt-4o-mini` |
| `OPENAI_MODEL_FREE` | `gpt-5-mini` |
| `OPENAI_MODEL_QUERY_REWRITER` | `gpt-4o-mini` |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` |

`OPENAI_MODEL_BASELINE` is only the legacy fallback used when an agent-specific
model is not set. It is not a separate evaluation baseline; the dissertation
benchmark configuration is defined in `evaluation/configs/benchmark_final.json`.

Model availability, pricing and provider compatibility can change. A configured model must support the operations and structured-output behaviour expected by its agent.

The `free` agent uses `LLM_PROVIDER_FREE` and `OPENAI_MODEL_FREE` for general research tasks. When its task needs current or factual external information, it can call the configured web-search provider. That path also requires `SEARCH_API_KEY`; it is not available in the supported offline profile (`DISABLE_EXTERNALS=1`). Search findings are returned with source references and limitations.

## Web search

| Variable | Default | Purpose |
| --- | --- | --- |
| `SEARCH_PROVIDER` | `serper` | Implemented backends are `serper` and `tavily`; other values fall back to `serper` |
| `SEARCH_API_KEY` | empty | Provider credential |
| `ALLOWED_DOMAINS` | `europa.eu,globalgap.org,boe.es` | Comma-separated source-domain allow-list |

Search is not guaranteed to run merely because a key is configured. The plan and writer policy must also permit it.

## RAG and legal retrieval

| Variable | Default | Purpose |
| --- | --- | --- |
| `VECTOR_BACKEND` | `sqlite` | `sqlite`, `pgvector` or `qdrant` |
| `DATABASE_URL` | `sqlite:///./data.db` | SQLite vector path or database connection used by the vector layer |
| `QDRANT_PATH` | `./qdrant_data` | Embedded Qdrant storage |
| `QDRANT_COLLECTION` | `legal_chunks` | Qdrant collection |
| `LEGAL_RAG_STRATEGY` | `hybrid` | `bm25`, `vector` or `hybrid` |
| `LEGALIZE_DATA_DIR` | `./data/legalize` | Local legal corpus |
| `LEGALIZE_DEFAULT_REPOS` | `es,eu` | Default corpus groups |
| `LEGALIZE_INGEST_PROFILE` | `agro` | Ingestion profile |

`DATABASE_URL` does not configure the explicit case, conversation or cost stores; those use their own local paths.

## Remote sensing

| Variable | Default | Purpose |
| --- | --- | --- |
| `ENABLE_STAC` | `true` | Makes the STAC agent available |
| `ENABLE_RS_ANALYST` | `true` | Makes remote-sensing interpretation available when STAC is also enabled |
| `STAC_API_URL` | Planetary Computer STAC | Catalogue endpoint |
| `STAC_SEARCH_MODE` | `direct` | `direct` or `semantic_service` |
| `STAC_SEMANTIC_URL` | empty | Optional semantic-search service |
| `STAC_DEFAULT_COLLECTIONS` | empty | Comma-separated default collections |
| `STAC_MAX_ITEMS` | `12` | Maximum scenes returned downstream |
| `STAC_CONCURRENCY_LIMIT` | `4` | Concurrent catalogue operations |

An empty collection list allows query-specific selection. Remote-sensing output still depends on coordinates, dates, catalogue availability and compatible assets.

### Meteorological context

The remote-sensing workflow can query the historical Open-Meteo archive for the representative coordinate and acquisition interval of the selected scenes. It derives descriptive precipitation and temperature summaries for context; it does not reproduce a climatological SPI calculation and does not replace field measurements. The provider is part of the STAC-to-analysis route rather than a separately configurable application service. Disable external calls with `DISABLE_EXTERNALS=1` for offline inspection.

## Geocoding

| Variable | Default |
| --- | --- |
| `GEOCODER` | `NOMINATIM` |
| `GEOCODER_URL` | `https://nominatim.openstreetmap.org/search` |
| `GEOCODER_EMAIL` | empty |
| `GEOCODER_COUNTRY_BIAS` | `ES` |
| `GEOCODER_VIEWBOX` | empty |
| `GEOCODER_MAPBOX_TOKEN` | empty |
| `GEOCODER_GOOGLE_KEY` | empty |

When using a public geocoder, comply with its usage policy and provide any required identification. Do not send sensitive farm data to a third party without a reviewed data basis.

## OCR and attachments

| Variable | Default | Purpose |
| --- | --- | --- |
| `OCR_BACKEND` | `tesseract` | `tesseract` or `none` |
| `ATTACHMENTS_DIR` | `./backend/attachments` | File and metadata storage |

Tesseract must be installed separately and available to the process. Setting `OCR_BACKEND=none` disables local OCR but does not alter attachment size and type validation.

The API limit is six files per request and 10 MiB per file.

## Persistence

| Variable | Default | Data |
| --- | --- | --- |
| `CONVERSATIONS_DB_PATH` | `./data/conversations.db` | Conversations and messages |
| `COST_DB_PATH` | `./data/costs.db` | Usage and cost events |
| `MEMORY_DIR` | `./backend/memory` | Opt-in user memory |

The explicit case store currently uses the fixed local path `./data/cases.db`.

All paths are resolved relative to the process working directory. Run the API from the repository root unless you deliberately provide absolute paths.

## Events and logging

| Variable | Default | Purpose |
| --- | --- | --- |
| `LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING` or `ERROR` |
| `SSE_LOG_MODE` | `memory` | `memory`, `stdout` or `disk` |
| `SSE_TRACE` | `false` | Captures additional emitted/delivered trace data |
| `CORS_ORIGINS` | `http://localhost:5173` | Comma-separated browser origins |

`SSE_LOG_MODE=disk` stores conversation JSONL under `backend/logs/conversations/`. Disk mode provides local replay only; it is not a distributed event system.

Trace and log output can contain query and execution context. Apply appropriate retention and access controls outside a local demonstration.

## Cost tracking

| Variable | Default |
| --- | --- |
| `COST_TRACKING_ENABLED` | `true` |
| `COST_PRICING_MODE` | `standard` |
| `COST_WARN_USD_PER_CONVERSATION` | `0.25` |
| `COST_WARN_USD_PER_DAY` | `2.0` |
| `WEB_SEARCH_COST_USD_PER_1K` | `0.0` |
| `OPENAI_WEB_SEARCH_COST_USD_PER_1K` | `10.0` |

Recorded values are local estimates. Warning thresholds do not prevent provider spend.

## Orchestration controls

| Variable | Default | Purpose |
| --- | --- | --- |
| `AGENT_CONCURRENCY_LIMIT` | `4` | Maximum concurrently running agent instances |
| `REPLAN_MAX_COST_USD` | `2.0` | Disables optional replanning above the accumulated estimate |
| `MEMORY_REMOTE_SENSING_TTL_DAYS` | `21` | Freshness window for reusable remote-sensing memory |

These settings bound runtime behaviour but are not a substitute for process-level resource limits.

## Frontend

| Variable | Default in examples | Purpose |
| --- | --- | --- |
| `VITE_API_BASE` | `http://localhost:8000` | API base URL embedded by Vite |

Vite variables are available to browser code. Never place secrets in a `VITE_` variable.

## Docker Compose

The supplied Compose file:

- builds the API from `Dockerfile`;
- starts Uvicorn on port `8000`;
- starts a Node 20 Vite development server on port `5173`;
- mounts the repository into both containers; and
- passes `.env` to the API.

It is a development convenience. It uses `npm install`, a live source mount and the Vite development server; it is not a hardened production image.

## Security checklist

Before enabling external calls:

- keep `.env` outside version control;
- restrict `CORS_ORIGINS`;
- review provider data-processing terms;
- use scoped credentials;
- inspect log and trace retention;
- avoid real personal or commercially sensitive data in the prototype;
- verify official regulatory sources at decision time; and
- monitor provider usage independently of local cost estimates.
