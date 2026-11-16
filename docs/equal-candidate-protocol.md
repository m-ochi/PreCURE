# Equal-candidate protocol

`precure --protocol equal_candidates --method all` implements the Task C
experimental design in the current `paper/sections/results.tex` and
`paper/appendix_tex/equal45_protocol.tex` on publication-safe inputs.
It uses the public prompts and fidelity gate; it does not reproduce the
private X corpus, historical prompts, or saved provider outputs. Tasks A/B
on observed replies and histories remain outside this package.

## Offline validation

```bash
python -m pip install -e '.[gepa]'
precure --protocol equal_candidates --method all \
  --run-id offline-control-flow-check --output-dir outputs/equal-candidates
```

No API key or network access is needed to run the mock backend. The mock uses
a finite paraphrase fixture to exercise the 45-candidate budget. Its distinct
strings are not evidence of materially distinct designs or actual UGC quality.
The bundled six campaigns and first fictional persona per campaign test the
execution path, rather than recreate the paper's 11 observed campaigns.

## Search and selection

Defaults follow the paper:

1. Keep one persona fixed per campaign. Keep offer, period, eligibility and
   source facts fixed; edit only `creative_hint`.
2. Screen Original and proposals until exactly 45 unique valid expressions
   have been evaluated. Canonical identity is sorted JSON of all expression
   fields, matching the private runner; internal whitespace is preserved.
3. Use five simulated responses per candidate, with the same sample keys
   across candidates and methods. Preserve repeated response texts as separate
   generation attempts. Rescore the generated text independently of its
   predicted tendency vector.
4. Return the evaluated archive and recent rejection reasons to every proposer.
   Duplicates, empty strings and gate failures count as attempts, not candidates.
   Stop at 360 proposals if the target has not been reached, marking the cell
   `incomplete_candidate_budget`. No confirmation or endpoint is produced for
   that cell. The command exits with status 2 after writing the completed cells
   and incomplete records; it does not infer results from a smaller budget.
5. Confirm Original and the top three revisions with 18 new paired samples.
   Compute each revision's paired CPS difference from Original, then subtract
   `1.699 * sample_standard_deviation / sqrt(18)` from its mean gain. Select the
   largest strictly positive adjusted gain; otherwise retain Original. Ties
   retain the first candidate in screening order.
6. Freeze that expression, then evaluate it and Original with 18 more paired
   samples from a separate common seed. When Original wins, reuse its final
   evaluation exactly. The multi-method CLI also shares the final Original
   evaluation across methods to avoid relying on provider seed determinism.
   Neither confirmation nor final samples enter search.

`sequential_cps` revises the current valid candidate from reply/CPS feedback.
`ipc_cps` performs explicit error analysis and uses recent history, replacing
it with highest-scoring history every third analysis. Its equal-candidate mode
continues to the candidate target rather than stopping on demo patience.
`gepa_cps` uses upstream GEPA 0.1.1 Pareto parent selection and reflective
minibatches over the five screening instances. Every reflected proposal is
fully screened, including proposals later rejected by GEPA's minibatch test.
A native stop callback ends search when the candidate target is reached.
`prefpo_cps` contrasts the highest-CPS pool member with a rotating distinct pool
member and revises the nonpreferred invitation. The first comparison uses
Original on both sides to initialize the pool. These public search adaptations
share the paper's operator definitions, but are not byte-identical copies of
the historical private implementation.

## Outputs and provenance

Use an empty output directory for every run. The runner writes `run_config.json`
**before** model calls, recording a stable `run_id`, Git commit, hashes of the
actual Python source files (including uncommitted edits), input hashes, model,
prompt/protocol version, seeds, objective and budgets. Register reported
experiments in the control repository before using their results in the paper.

`protocol_results.jsonl` stores a flushed record per completed or incomplete
cell: candidate order, attempts, rejection reasons, checkpoints at 5/10/20/45,
optimizer feedback, per-response scores and seeds, confirmation comparisons,
frozen selection, independent final scores, requests, provider tokens/cost and
wall time. `summary.csv` distinguishes complete and incomplete cells. A runtime
provider error marks the run config failed and preserves prior flushed cells;
the interrupted cell is not resumable and is not included as an endpoint.

For at least two campaigns, a complete matrix also produces `comparison.json`:
paired campaign-bootstrap percentile intervals with 10,000 shared resamples
(seed 20261113), exact two-sided sign-flip tests versus Original and Holm
correction across the requested methods. Components include CPS, Q, burden,
and fidelity risk. Incomplete or duplicate cells cannot enter inference. Exact
sign-flip enumeration supports up to 20 campaigns; the paper uses 11.

`--candidate-budget`, `--proposal-ceiling`, `--samples-per-persona`,
`--confirmation-samples`, `--final-samples`, `--analysis-seed` and
`--bootstrap-resamples` support smaller checks and are recorded explicitly.
Changing them changes the experimental design. `--iterations` and
`--ipc-patience` apply only to the demo protocol; equal-candidate search uses
its candidate budget and proposal ceiling. `--restarts` must be 1 and
`--personas-per-campaign` must be 1 in equal-candidate mode.

LLM execution is opt-in via `--backend openrouter`. The global
`--max-cost-usd` applies across all cells and stages. Actual paid experiments
were not run as part of implementing this protocol.
