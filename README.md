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

## Source campaigns

The 11 canonical campaigns are official, brand-owned promotional posts from
public company accounts. Unlike UGC replies, these posts are not personal
data, so the links themselves are listed here:

| Campaign ID | Brand | Official post |
|---|---|---|
| starbucks_feedback_20260527 | Starbucks | <https://x.com/i/web/status/2059439461390545211> |
| starbucks_mobile_order_20260529 | Starbucks | <https://x.com/i/web/status/2060164471969202577> |
| starbucks_choice_20260610 | Starbucks | <https://x.com/i/web/status/2064543324430823694> |
| stmarch_comment_20260520 | Saint Marc Cafe | <https://x.com/i/web/status/2057234917696963064> |
| stmarch_episode_comment_20260619 | Saint Marc Cafe | <https://x.com/i/web/status/2067789466962731128> |
| subway_new_sandwich_20260513 | Subway | <https://x.com/i/web/status/2054334554761027959> |
| tullys_choice_20260508 | Tully's Coffee | <https://x.com/i/web/status/2052554394445709713> |
| tullys_cookie_comment_20260509 | Tully's Coffee | <https://x.com/i/web/status/2052976953125372400> |
| tullys_choice_reply_20260512 | Tully's Coffee | <https://x.com/i/web/status/2054320782352998892> |
| tullys_choice_reply_20260520 | Tully's Coffee | <https://x.com/i/web/status/2056919108885839886> |
| tullys_scene_reply_20260526 | Tully's Coffee | <https://x.com/i/web/status/2059129923659571364> |

These links resolve to the brand's own post; they do not identify any UGC
author or reveal reply content.

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

Every quantitative claim in the paper is backed by an aggregate artifact with
a recorded run ID and hash; row-level social-media or participant data is
intentionally out of scope for redistribution rather than missing by
oversight.

## Licensing

`LICENSE` and `CITATION.cff` are pending and will be added together with the
first populated release, before any aggregate artifact is published here.

## Contact

Questions about reproduction should go to the corresponding author listed in
the paper.
