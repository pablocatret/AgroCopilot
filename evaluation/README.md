# AgroCopilot evaluation

This directory contains the reproducible evaluation framework used for the
master's dissertation. It is written for technical reviewers: the final
benchmark is the reference point, while older exploratory profiles and
baseline terminology are not part of the dissertation experiment.

## Final benchmark

The canonical configuration is
[`evaluation/configs/benchmark_final.json`](configs/benchmark_final.json). It
evaluates six model configurations on the same AgroCopilot multi-agent
workflow. The final experiment does **not** include a separate monolithic
baseline; the models are compared as alternative backends for the same system.

The corpus contains 57 cases:

- 48 seed cases covering general assistance, decision support, diagnosis,
  compliance and attachment analysis;
- 9 adversarial cases covering ambiguity, unsupported claims, legal or safety
  boundaries, evidence gaps, continuity and attachment handling.

Each case is executed once for each configuration at temperature zero: 57 × 6
= 342 system executions. Three LLM judge configurations then assess every
execution, producing 1,026 judge assessments. The ten semantic dimensions are
factual correctness, domain accuracy, responsible action quality,
actionability, decision-support quality, evidence utilisation, transparent
confidence, case personalisation, practical value and overall quality.

The benchmark also records operational and contract-level signals, including
answer completeness, forbidden-claim rate, overclaim count, actionability,
routing precision/recall/order/score, execution success, parser status,
latency, token usage and captured cost. Cost estimates are provider-dependent;
the dissertation does not claim a complete total cost because web-search and
embedding prices are not consistently available.

The three judge configurations in the final profile are `mimo-v2.5`,
`hy3-preview` and `step-3.7-flash`; all three are accessed through OpenRouter.
The six system configurations are:

| Configuration | Model backend |
|---|---|
| `openai_barata` | OpenAI `gpt-5-mini` |
| `openai_buena` | OpenAI `gpt-5.6-luna` |
| `china_barata` | OpenRouter `deepseek/deepseek-v4-flash` |
| `china_top` | OpenRouter `deepseek/deepseek-v4-pro` |
| `hiper_rapida` | OpenRouter `openai/gpt-oss-120b:nitro` |
| `hiper_pequena` | OpenRouter `mistralai/ministral-8b-2512` |

The manuscript reports the following mean overall-quality scores:

| Model | Mean score |
|---|---:|
| GPT-5 mini | 4.304 |
| GPT-5.6 Luna | 4.246 |
| DeepSeek V4 Flash | 3.807 |
| GPT-OSS 120B Nitro | 3.795 |
| DeepSeek V4 Pro | 3.760 |
| Ministral 8B | 3.374 |

The top two configurations do not show a detected statistically significant
difference under the dissertation protocol. Evidence utilisation is the
weakest semantic dimension overall, and the middle configurations remain
unresolved rather than supporting a strong ranking claim.

## Statistical analysis

The canonical inferential analysis is
[`analyze_benchmark_statistics.py`](analyze_benchmark_statistics.py). It uses
paired within-case comparisons, 10,000-case bootstrap confidence intervals,
the Friedman global test, paired Wilcoxon tests with Holm correction, paired
effect sizes, Wilson intervals, ICC(2,1) and Kendall's W. The default seed is
`20260724` so the exported intervals are reproducible.

[`stats.py`](stats.py) contains auxiliary descriptive, correlation and Pareto
helpers used by the reporting layer and compatibility tests. Its older
paired-t-test and bootstrap defaults are not the dissertation's canonical
inferential results.

## Running the benchmark

Run the preflight first. It validates the corpus, attachments, basic
configuration and provider credentials without making LLM calls. The
`compare --dry-run` command prints the cost estimate and configured budget:

```bash
python -m evaluation.cli preflight --config evaluation/configs/benchmark_final.json
```

On Windows, use the project virtual environment when available:

```powershell
.\.venv\Scripts\python.exe -m evaluation.cli preflight --config evaluation/configs/benchmark_final.json
```

A full run can be started with:

```bash
EVALUATION_ENABLE_LLM=1 python -m evaluation.cli compare \
  --config evaluation/configs/benchmark_final.json \
  --output evaluation/results/benchmark_final \
  --max-concurrent 2
```

The Windows overnight helper runs the same 57 cases in seven resumable ranges
(`1:5`, `6:12`, `13:20`, `21:30`, `31:40`, `41:50`, `51:57`):

```powershell
.\evaluation\run_benchmark_overnight.ps1
```

It starts fresh ranges in the canonical output directory and does not depend
on batch identifiers from another checkout. Set provider credentials and
confirm the budget before enabling external calls.

To inspect or resume a generated batch:

Replace `<batch_id>` with the directory created by `compare` under
`evaluation/results/benchmark_final/`.

```bash
python -m evaluation.cli status --batch evaluation/results/benchmark_final/<batch_id>
python -m evaluation.cli resume --batch evaluation/results/benchmark_final/<batch_id>
python -m evaluation.cli repair-broken --batch evaluation/results/benchmark_final/<batch_id>
```

Generate a descriptive report and run the dissertation statistics with:

```bash
python -m evaluation.cli report \
  --results-dir evaluation/results/benchmark_final/<batch_id> \
  --output evaluation/report/<batch_id>
python evaluation/analyze_benchmark_statistics.py \
  --input evaluation/results/benchmark_final \
  --bootstrap 10000 \
  --seed 20260724
```

The report command is descriptive. The statistical script is the source of
the benchmark's inferential tables.

## Evaluation runtime controls

The final benchmark requires an explicit LLM opt-in and provider credentials:

| Variable | Default | Effect |
|---|---:|---|
| `EVALUATION_ENABLE_LLM` | unset/off | Must be `1`, `true`, `yes` or `on` to allow model calls; credentials are still required |
| `EVALUATION_INITIAL_CONCURRENT` | unset | Overrides the initial concurrency per provider and role |
| `EVALUATION_REQUEST_TIMEOUT_SECONDS` | `900` | Timeout for one model request |
| `EVALUATION_CATALOG_TIMEOUT_SECONDS` | `30` | Timeout for the OpenRouter model-capability catalogue request |
| `EVALUATION_TASK_TIMEOUT_SECONDS` | `7200` | Timeout for one complete case/configuration task, including judge calls |

`--max-concurrent` controls the configured initial concurrency for `compare`;
`EVALUATION_INITIAL_CONCURRENT`, when set, takes precedence. These controls are
operational safeguards and do not change the canonical corpus, models, judges,
temperature or statistical analysis.

## Additional CLI commands

The canonical reproduction path is `compare` with
`evaluation/configs/benchmark_final.json`. The CLI also exposes the following
inspection and maintenance commands:

```bash
python -m evaluation.cli audit --results-dir evaluation/results/benchmark_final/<batch_id>
python -m evaluation.cli routing --results-dir evaluation/results/benchmark_final/<batch_id>
python -m evaluation.cli rebase --batch evaluation/results/benchmark_final/<batch_id>
python -m evaluation.cli init-config --output eval_config.json
python -m evaluation.cli list-packs
python -m evaluation.cli list-judges
```

`run` remains available for compatibility and filtered exploratory runs; it is
not the canonical command for the dissertation benchmark.

## Configuration and artefacts

`eval_config.json` mirrors the final benchmark as a convenient repository-root
entry point. The specialised profiles under `evaluation/configs/` include the
canonical final profile and exploratory or smoke-test configurations; only
`benchmark_final.json` should be used to reproduce the dissertation results.

Generated `evaluation/results/` and `evaluation/report/` directories are
local artefacts and are excluded from Git. The case corpus, attachments,
configuration and analysis code are versioned. The numeric summary above is
the frozen result reported in the manuscript; a generated result export is not
currently committed to the repository.

## Interpretation limits

The results are scoped to this corpus, workflow, prompt version, provider
availability and execution protocol. They are not expert agronomic or legal
validation, and one run per case/configuration cannot characterise all model
variance. The manuscript also reports incomplete provider pricing and does not
support production-quality claims from these benchmark results alone.

The Spanish-language case texts and fixtures are intentional evaluation inputs,
not stale documentation. Do not translate or rewrite them when reproducing
the benchmark.

Run the framework tests with:

```bash
python -m pytest tests_evaluation -q
```
