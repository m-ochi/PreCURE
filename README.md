# PreCURE campaign-expression optimizer

Publication-safe reference implementation for the campaign-optimization core of:

> **PreCURE: A Pre-Publication Decision Framework for Richer UGC in Brand-Owned Social Media Promotions**
>
> Masanao Ochi and Takeshi Sakaki.

PreCURE keeps source-grounded campaign facts fixed and searches only the editable participation expression. Candidate expressions are compared using persona-conditioned synthetic responses, a four-axis UGC quality score, a response-burden penalty, a factual-drift penalty, and a deterministic fidelity gate.

This repository intentionally contains no observed X replies, user histories, account identifiers, participant data, or private model traces.

## Run the offline demo

Python 3.10 or newer is the only requirement. The fastest path from a fresh clone is:

```bash
./scripts/run_demo.sh
```

The script runs all six fictional stress-test campaigns with three fictional personas each. It uses the deterministic `mock` backend, makes no network requests, needs no API key, and writes:

- `outputs/demo/summary.csv`: selected expression and CPS change by campaign;
- `outputs/demo/trajectories.jsonl`: every evaluated candidate, response, score, risk, and gate decision;
- `outputs/demo/run_config.json`: parameters and SHA-256 hashes of the inputs.

The mock backend tests the full control flow and makes expected failure modes easy to inspect. Its scores are transparent heuristics; they are not the LLM scores or numerical results reported in the paper.

To install the command-line entry point and run the tests:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m unittest discover -s tests -v
precure --backend mock --method prefpo_cps --iterations 15
```

## Method implemented here

For response-axis means \(\bar{s}=(a,d_{cov},d_{spec},d_{rea})\), the default quality term is:

```text
Q = 0.505 a + 0.225 d_cov + 0.150 d_spec + 0.150 d_rea
```

The Campaign Proposal Score (CPS) is:

```text
J_plan = Q - 0.10 C_B - 0.02 C_F
```

where `C_B` is response burden and `C_F` is factual-drift risk. A candidate that fails the deterministic source gate is ineligible regardless of its score. Persona and campaign-conditioned response tendencies are combined with the paper's `eta=0.25` calibration before response generation.

Two search rules are available:

- `fixed_cps`: the paper's auditable score-feedback loop (Original plus up to four rewrites by default), returning the highest-CPS valid candidate;
- `prefpo_cps`: a compact pairwise candidate-pool adaptation that contrasts the current candidate with the preferred candidate and improves the nonpreferred expression.

The public runner conservatively freezes `offer`, `period`, and `eligibility` and edits only `creative_hint`. This is the search scope used by the paper's PrefPO-CPS condition and prevents protected facts from silently entering the rewrite space. See [docs/method.md](docs/method.md) for the code-to-paper map and current release boundary.

## Run with an LLM through OpenRouter

LLM execution is opt-in and may incur provider charges. Start with one campaign, one persona, one sample, and two iterations:

```bash
export OPENROUTER_API_KEY='...'
precure \
  --backend openrouter \
  --model deepseek/deepseek-v4-flash \
  --max-cost-usd 0.25 \
  --method fixed_cps \
  --limit-campaigns 1 \
  --personas-per-campaign 1 \
  --samples-per-persona 1 \
  --iterations 2 \
  --output-dir outputs/openrouter-smoke
```

The runner requests zero-data-retention routing and denies provider data collection. It stops before starting another request after recorded provider cost reaches `--max-cost-usd`; one in-flight request can make the final total slightly exceed the cap. Provider availability and structured-output support can change; override `--model` when necessary. Only the bundled fictional data should be used with this public runner.

The full stress test used larger, registered budgets: five iterations with three personas and three responses per persona for fixed CPS; the PrefPO-CPS experiment additionally used restarts, screening, fresh-seed confirmation, and a paired lower-confidence-bound selection rule. The compact public `prefpo_cps` runner exposes the optimization mechanism but does not claim to reproduce those exact API results or budgets.

## Synthetic stress-test data

[`data/synthetic_stress_test/`](data/synthetic_stress_test/) contains the exact six fictional campaign and persona inputs registered for the paper's difficult-campaign stress test. They deliberately combine multi-part instructions, long-answer requirements, photo or location requests, friend tagging, and unsupported benefit claims. All brands, post IDs, URLs, and personas are fictional.

The source hashes and disclosure statement are recorded in [`data/synthetic_stress_test/README.md`](data/synthetic_stress_test/README.md). Do not replace these fixtures with observed UGC or real-user histories.

## Repository layout

```text
src/precure/                 objective, gate, backends, and search loops
data/synthetic_stress_test/  exact fictional test inputs
scripts/run_demo.sh          no-key, one-command smoke run
tests/                       unit and end-to-end tests
docs/method.md               implementation map and limitations
```

## Reproducibility and privacy boundary

The complete paper also evaluates observed Japanese X campaigns and user-conditioned response generation. Those raw replies and user histories are excluded because free text can be searchable back to an author and redistribution is not required to exercise the optimization method. Aggregate, disclosure-reviewed tables will be released separately after their audit.

Never place API keys in the repository. `.env`, private data paths, generated outputs, key files, and staging directories are ignored by default.

## License and citation

The source code is released under the MIT License; see `LICENSE`. Citation metadata is provided in `CITATION.cff`.
