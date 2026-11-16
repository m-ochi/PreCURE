from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Sequence

from .engine import fixed_cps_search, prefpo_cps_search, gepa_cps_search, ipc_cps_search, OPTIMIZER_PROMPT_VERSION
from .io import load_campaigns, load_personas
from .mock_backend import MockBackend
from .openrouter_backend import OpenRouterBackend
from .protocol import METHODS, PROTOCOL_VERSION, ProtocolConfig, run_protocol


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "data" / "synthetic_stress_test"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optimize fictional campaign participation expressions with PreCURE."
    )
    parser.add_argument(
        "--campaigns", type=Path,
        default=DEFAULT_DATA / "task_c_facts.json",
    )
    parser.add_argument(
        "--personas", type=Path,
        default=DEFAULT_DATA / "task_c_personas.jsonl",
    )
    parser.add_argument("--backend", choices=("mock", "openrouter"), default="mock")
    parser.add_argument("--method", choices=("fixed_cps", *METHODS, "all"), default="fixed_cps")
    parser.add_argument("--protocol", choices=("demo", "equal_candidates"), default="demo")
    parser.add_argument("--candidate-budget", type=int, default=45)
    parser.add_argument("--proposal-ceiling", type=int, default=360)
    parser.add_argument("--confirmation-samples", type=int, default=18)
    parser.add_argument("--final-samples", type=int, default=18)
    parser.add_argument("--run-id", help="Stable identifier; register before reporting an experiment.")
    parser.add_argument("--analysis-seed", type=int, default=20261113)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--model", default="deepseek/deepseek-v4-flash")
    parser.add_argument(
        "--max-cost-usd", type=float, default=1.0,
        help="Stop OpenRouter execution after recorded cost reaches this cap.",
    )
    parser.add_argument("--restarts", type=int, default=1)
    parser.add_argument("--ipc-history-length", type=int, default=5)
    parser.add_argument("--ipc-patience", type=int, default=3)
    parser.add_argument("--reflection-minibatch-size", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--samples-per-persona", type=int)
    parser.add_argument("--personas-per-campaign", type=int)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--limit-campaigns", type=int)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "demo")
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.samples_per_persona is None:
        args.samples_per_persona = 5 if args.protocol == "equal_candidates" else 3
    if args.personas_per_campaign is None:
        args.personas_per_campaign = 1 if args.protocol == "equal_candidates" else 3
    if args.iterations < 1 or args.samples_per_persona < 1:
        raise SystemExit("iterations and samples-per-persona must be positive")
    if args.personas_per_campaign < 1:
        raise SystemExit("personas-per-campaign must be positive")
    if args.limit_campaigns is not None and args.limit_campaigns < 1:
        raise SystemExit("limit-campaigns must be positive")
    if args.protocol == "equal_candidates":
        _equal_candidates_main(args)
        return
    if args.method == "all":
        raise SystemExit("--method all requires --protocol equal_candidates")
    campaigns = load_campaigns(args.campaigns)
    if args.limit_campaigns is not None:
        campaigns = campaigns[:args.limit_campaigns]
    personas = load_personas(args.personas)
    backend = (
        MockBackend()
        if args.backend == "mock"
        else OpenRouterBackend(args.model, max_cost_usd=args.max_cost_usd)
    )
    search = {"fixed_cps": fixed_cps_search, "sequential_cps": fixed_cps_search, "prefpo_cps": prefpo_cps_search,
              "gepa_cps": gepa_cps_search, "ipc_cps": ipc_cps_search}[args.method]
    options = {}
    if args.method in {"gepa_cps", "ipc_cps"}:
        options["restarts"] = args.restarts
    if args.method == "ipc_cps":
        options.update(history_length=args.ipc_history_length, patience=args.ipc_patience)
    if args.method == "gepa_cps":
        options["reflection_minibatch_size"] = args.reflection_minibatch_size
    results = []
    for campaign in campaigns:
        selected_personas = personas.get(campaign.campaign_id, [])[:args.personas_per_campaign]
        if not selected_personas:
            raise SystemExit(f"no personas for {campaign.campaign_id}")
        results.append(search(
            backend,
            campaign,
            selected_personas,
            iterations=args.iterations,
            samples_per_persona=args.samples_per_persona,
            seed=args.seed,
            **options,
        ))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trajectory_path = args.output_dir / "trajectories.jsonl"
    with trajectory_path.open("w", encoding="utf-8") as handle:
        for result in results:
            for iteration, row in enumerate(result.trajectory):
                handle.write(json.dumps({
                    "campaign_id": result.campaign_id,
                    "method": result.method,
                    "iteration": iteration,
                    "selected": iteration == result.selected_iteration,
                    **row.to_dict(),
                }, ensure_ascii=False, sort_keys=True) + "\n")

    with (args.output_dir / "optimizer_trace.jsonl").open("w", encoding="utf-8") as handle:
        for result in results:
            for event in result.optimizer_trace:
                handle.write(json.dumps({"campaign_id": result.campaign_id,
                                         "method": result.method, **event},
                                        ensure_ascii=False, sort_keys=True) + "\n")

    summary_path = args.output_dir / "summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "campaign_id", "method", "selected_iteration", "original_cps",
            "selected_cps", "delta_cps", "selected_c_b", "selected_c_f",
            "selected_creative_hint", "fidelity_passed",
        ])
        writer.writeheader()
        for result in results:
            original = result.trajectory[0]
            selected = result.selected
            writer.writerow({
                "campaign_id": result.campaign_id,
                "method": result.method,
                "selected_iteration": result.selected_iteration,
                "original_cps": original.cps,
                "selected_cps": selected.cps,
                "delta_cps": float(selected.cps or 0) - float(original.cps or 0),
                "selected_c_b": selected.risks.c_b,
                "selected_c_f": selected.risks.c_f,
                "selected_creative_hint": selected.expression.creative_hint,
                "fidelity_passed": selected.fidelity.passed,
            })

    config = {
        "optimizer_prompt_version": OPTIMIZER_PROMPT_VERSION,
        "search_options": options,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "backend": backend.name,
        "method": args.method,
        "seed": args.seed,
        "iterations": args.iterations,
        "samples_per_persona": args.samples_per_persona,
        "personas_per_campaign": args.personas_per_campaign,
        "campaign_count": len(results),
        "max_cost_usd": args.max_cost_usd,
        "objective": {
            "omega": {"a": 0.505, "d_cov": 0.225, "d_spec": 0.150, "d_rea": 0.150},
            "lambda_b": 0.10,
            "lambda_f": 0.02,
            "eta": 0.25,
        },
        "inputs": {
            str(args.campaigns): _sha256(args.campaigns),
            str(args.personas): _sha256(args.personas),
        },
        "openrouter": {
            "request_count": getattr(backend, "request_count", 0),
            "cost_usd": getattr(backend, "cost_usd", 0.0),
        },
    }
    (args.output_dir / "run_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {summary_path}")
    for result in results:
        print(
            f"{result.campaign_id}: t={result.selected_iteration} "
            f"CPS {result.trajectory[0].cps:.3f} -> {result.selected.cps:.3f}"
        )


def _equal_candidates_main(args):
    import subprocess
    if args.personas_per_campaign != 1:
        raise SystemExit("equal_candidates requires --personas-per-campaign 1")
    if args.restarts != 1:
        raise SystemExit("equal_candidates uses one archive per cell; --restarts must be 1")
    if args.bootstrap_resamples < 1:
        raise SystemExit("bootstrap-resamples must be positive")
    try:
        config = ProtocolConfig(
            candidate_budget=args.candidate_budget, proposal_ceiling=args.proposal_ceiling,
            screening_samples=args.samples_per_persona,
            confirmation_samples=args.confirmation_samples, final_samples=args.final_samples,
            history_length=args.ipc_history_length,
            reflection_minibatch_size=args.reflection_minibatch_size,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    methods = METHODS if args.method == "all" else (
        "sequential_cps" if args.method == "fixed_cps" else args.method,
    )
    if "gepa_cps" in methods:
        try:
            import gepa.optimize_anything
        except ImportError as exc:
            raise SystemExit("GEPA-CPS requires pip install -e '.[gepa]'") from exc
    campaigns = load_campaigns(args.campaigns)
    if args.limit_campaigns:
        campaigns = campaigns[:args.limit_campaigns]
    if not campaigns:
        raise SystemExit("no campaigns")
    personas = load_personas(args.personas)
    for campaign in campaigns:
        if not personas.get(campaign.campaign_id):
            raise SystemExit(f"no personas for {campaign.campaign_id}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.output_dir / "run_config.json"
    if any(args.output_dir.iterdir()):
        raise SystemExit("equal_candidates requires an empty output directory to preserve run provenance")
    from dataclasses import asdict
    try:
        commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unavailable"
    source_hashes = {str(path.relative_to(ROOT)): _sha256(path)
                     for path in sorted((ROOT / "src" / "precure").glob("*.py"))}
    run_config = {
        "run_id": args.run_id or "public-equal-candidates-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"),
        "protocol_version": PROTOCOL_VERSION, "optimizer_prompt_version": PROTOCOL_VERSION,
        "code_commit": commit, "source_sha256": source_hashes,
        "created_at": datetime.now(timezone.utc).isoformat(), "status": "running",
        "backend": args.backend, "model": args.model if args.backend == "openrouter" else "mock-v1",
        "seed": args.seed, "methods": list(methods), "parameters": asdict(config),
        "persona_policy": "first fixed persona per campaign",
        "campaign_ids": [campaign.campaign_id for campaign in campaigns],
        "inputs": {str(args.campaigns): _sha256(args.campaigns), str(args.personas): _sha256(args.personas)},
        "max_cost_usd": args.max_cost_usd,
        "analysis_seed": args.analysis_seed, "bootstrap_resamples": args.bootstrap_resamples,
        "objective": {"omega": [.505, .225, .150, .150], "lambda_b": .10, "lambda_f": .02, "eta": .25},
    }
    def write_config():
        config_path.write_text(json.dumps(run_config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_config()  # Freeze inputs/settings before any provider call.
    backend = MockBackend() if args.backend == "mock" else OpenRouterBackend(args.model, max_cost_usd=args.max_cost_usd)
    results = []
    final_original_cache = {}
    try:
        with (args.output_dir / "protocol_results.jsonl").open("w", encoding="utf-8") as handle:
            for campaign in campaigns:
                for method in methods:
                    result = run_protocol(backend, campaign, personas[campaign.campaign_id][:1],
                                          method=method, seed=args.seed, config=config,
                                          final_original_cache=final_original_cache)
                    handle.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
                    handle.flush()
                    results.append(result)
                    print(f"{campaign.campaign_id} {method}: {result['status']} "
                          f"({result['candidate_count']}/{config.candidate_budget})")
    except Exception as exc:
        run_config.update(status="failed", error=str(exc))
        write_config()
        raise
    incomplete = any(row["status"] != "complete" for row in results)
    run_config.update(status="incomplete_candidate_budget" if incomplete else "complete",
                      request_count=getattr(backend, "request_count", 0),
                      cost_usd=getattr(backend, "cost_usd", 0.0))
    write_config()
    with (args.output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["campaign_id", "method", "status", "candidate_count",
                                                   "proposal_attempts", "selected_index", "delta_cps"])
        writer.writeheader()
        for result in results:
            writer.writerow({key: result.get(key) for key in writer.fieldnames})
    if incomplete:
        raise SystemExit(2)
    from .analysis import summarize_comparison
    # A single campaign is useful for a smoke run, but not campaign-level inference.
    if len(campaigns) >= 2:
        analysis = summarize_comparison(results, tuple(methods), seed=args.analysis_seed,
                                        resamples=args.bootstrap_resamples)
        (args.output_dir / "comparison.json").write_text(
            json.dumps(analysis, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
