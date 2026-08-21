# Eliciting Richer User-Generated Responses to Brand-Owned Promotional Posts: User-Conditioned Language-Model Refinement - PreCURE Reproduction Package

Masanao Ochi (Oita University), Takeshi Sakaki (HottoLink, Inc.) — IEEE BigData 2026.

## Status

This repository was reset on 2026-08-21. Its previous contents implemented an
earlier pilot pipeline (an English Bluesky campaign corpus, Prolific human
evaluation, and GLM-4.7-Flash served via Ollama) that does not correspond to
the experiments reported in the submitted paper. Rather than leave a
reproduction package that silently mismatched the paper, the repository was
cleared and will be repopulated with the code and artifacts that actually
produced the submitted results.

**This repository currently contains no code or data.** It is public now so
that the code-availability statement in the paper resolves to a real
repository; the reproduction package itself is still being assembled from the
submission-time codebase and will be pushed here as a tagged release.

## What the completed package will contain

The submitted paper evaluates PreCURE on a Japanese-language X corpus (11
campaigns, four brands) using four main pipeline/scorer models (DeepSeek V4
Flash, GPT-5.6 Luna, Mistral Small 2603, Gemini 2.5 Flash-Lite) and a
seven-model extension for Task C (adding Claude Opus 5 Fast, GPT-5.6 Sol, and
Gemini 3.7 Flash). The release will include:

- Task A, B, and C runners, including the CPS refinement loop and the GEPA,
  AutoPrompt/IPC, and PrefPO-CPS optimizer adaptations.
- The deterministic hard-fidelity gate and primary-source semantic checks,
  with their unit tests.
- Prompts, model/provider configuration, and the leave-one-brand-out /
  leave-one-campaign-out fold definitions.
- De-identified, disclosure-reviewed aggregate result tables (scores,
  correlations, CPS deltas, blind-judgment win/loss/tie/unstable counts) that
  reproduce every table and figure in the paper, each tied to a run ID, model
  ID, and SHA-256 hash.

## What will not be included, and why

- **Raw X UGC replies, brand-post reply threads, or user posting histories.**
  Even with account handles removed, free-text replies are frequently
  searchable back to the original post and author. Redistributing collected
  platform content is also restricted by X's terms of service. The paper's
  statistics are reproducible from the aggregate tables above without the raw
  text.
- **Any participant-level identifiers or free-text responses from the prior
  Prolific human-evaluation round.** That round is superseded and is not part
  of the submitted paper's primary evidence; its data will not be
  redistributed here.
- **Full LLM traces that embed source UGC or user context.**

## For reviewers

Every quantitative claim in the paper is backed by an aggregate artifact with
a recorded run ID and hash; row-level social-media or participant data is
intentionally out of scope for redistribution rather than missing by
oversight. `LICENSE` and `CITATION.cff` are pending and will be added together
with the first populated release, before any aggregate artifact is published
here.

## Contact

Questions about reproduction should go to the corresponding author listed in
the paper.
