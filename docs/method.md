# Implementation map

This release isolates the publication-safe campaign-expression optimization core. It is not a redistribution of the private X corpus or a claim that deterministic mock scores equal the paper's model judgments.

## Code-to-method map

| Paper component | Public implementation |
|---|---|
| Fixed campaign core \(\phi\) and editable expression \(\theta\) | `Campaign` and `Expression` in `src/precure/models.py` |
| User/persona and campaign-conditioned tendencies | `Backend.score_persona` and `Backend.score_campaign` |
| \(\tilde{s}=\eta\mu+(1-\eta)\nu\), \(\eta=0.25\) | `calibrated_target` in `src/precure/objective.py` |
| Synthetic response generator \(G\) | `Backend.generate_response` |
| Four-axis response rubric | `Backend.score_response` and `AxisScores` |
| Quality \(Q=\omega^T\bar{s}\) | `quality_term` |
| Burden and fidelity-risk penalties | `Backend.score_risk` and `RiskScores` |
| Hard constraint \(H(\phi,\theta)\) | `validate_expression` in `src/precure/fidelity.py` |
| CPS \(J=Q-\lambda_B C_B-\lambda_F C_F\) | `campaign_proposal_score` |
| Fixed five-step score-feedback search | `fixed_cps_search` |
| Pairwise preference candidate-pool search | `prefpo_cps_search` |

Only `Q` is computed from generated responses. `C_B` and `C_F` are scored directly from the campaign core and candidate expression, so detailed synthetic responses cannot hide a burdensome or fact-drifting instruction. The gate is evaluated before model-based scoring; an invalid candidate receives no selectable CPS.

## Conservative public search space

The full experiments contain conditions where all expression fields can be proposed and checked. This public reference runner defaults to the narrower creative-only condition:

- `product` and category are outside the candidate object;
- `offer`, `period`, and `eligibility` are held fixed by construction, and the
  hard-fidelity gate additionally checks that the *facts* they encode
  (amounts, dates, negations, and the required entry action) are unchanged —
  a rewording that preserves the same facts passes, but a drift in an amount,
  deadline, or entry action does not;
- only the user-facing `creative_hint` is rewritten;
- encoded required terms, hashtags, and forbidden claims are checked deterministically.

This makes the source-protection boundary inspectable without publishing campaign-specific private review material. The deterministic gate does not replace open-ended human review of tone, product meaning, legal claims, or deployment safety.

## Backends

`MockBackend` is deterministic and network-free. It makes overloaded fictional starts produce low-quality refusal responses and assigns transparent penalties to long-answer, photo, tagging, location, purchase-history, and unsupported-guarantee requests. It exists for testing and teaching.

`OpenRouterBackend` runs the same control flow with structured model calls for persona scoring, campaign-conditioned scoring, response generation, response scoring, risk scoring, and refinement. The seed is derived from the run seed, campaign, iteration, persona, and sample so candidates receive matched, auditable rollout keys.

## Difference from the registered PrefPO-CPS experiment

The public pairwise runner retains the essential adaptation: evaluate candidates under CPS, contrast a preferred and nonpreferred expression, rewrite the nonpreferred expression, add nonduplicate candidates to a pool, preserve Original on ties, and exclude hard-gate failures.

The registered paper run additionally used three restarts, five-response screening, an 18-response fresh-seed confirmation stage, and a paired lower-confidence-bound rule. Those orchestration and budget controls are not yet included in this minimal release. Consequently, the public runner reproduces the method's executable mechanism and input contract, not the paper's saved provider outputs.

## Additional optimizer implementations

- `engine.gepa_cps_search`: upstream GEPA 0.1.1 `optimize_anything`, Pareto
  candidate selection, independent persona/sample examples, reflection on
  scored response evidence, no merge, cached screening and restart support.
  The per-example optimizer reward is `(CPS + 0.12) / 1.15`; this positive
  affine transformation preserves CPS ordering. Invalid candidates receive
  zero. The final public selection uses mean CPS over the complete screen.
- `engine.ipc_cps_search`: explicit error-analysis call followed by a rewrite
  call using scored expression/analysis history. Recent history is replaced
  with highest-scoring history every third step. Restart, patience and history
  length are configurable. Invalid rewrites do not replace the current state.
- `Backend.optimizer_text`: dedicated optimizer prompts, implemented by both
  OpenRouter and the deterministic offline mock. Mock text is a control-flow
  fixture, not an approximation of LLM reflection quality.

Both optimizers freeze the three protected fields structurally and apply the
same public fidelity gate before generating responses. Their release prompt
version and search options are recorded in `run_config.json`. Trajectories
include all screened expressions, scores, responses and gate failures.
The public gate and prompt set are the reference implementation described
above; historical experiment-specific checks, provider traces and the final
held-out confirmation/LCB stage are outside this release. Therefore this
release enables running both search algorithms, not numerical replication of
the paper's full experiment pipeline.

Upstream GEPA: https://github.com/gepa-ai/gepa (pinned dependency: 0.1.1).
