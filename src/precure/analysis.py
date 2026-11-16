"""Campaign-level inference for a complete equal-candidate comparison."""
from __future__ import annotations

from itertools import product
import random
from statistics import mean


def _percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def summarize_comparison(results: list[dict], methods: tuple[str, ...], *,
                         seed: int = 20261113, resamples: int = 10000) -> dict:
    if not results or not methods or resamples < 1:
        raise ValueError("nonempty results/methods and positive resamples required")
    if any(row["status"] != "complete" for row in results):
        raise ValueError("incomplete cells cannot enter endpoint inference")
    campaigns = sorted({row["campaign_id"] for row in results})
    if len(campaigns) > 20:
        raise ValueError("exact sign-flip implementation supports at most 20 campaigns")
    cells = {(row["campaign_id"], row["method"]): row for row in results}
    if len(cells) != len(results) or set(cells) != set(product(campaigns, methods)):
        raise ValueError("endpoint inference requires one complete cell per campaign and method")
    for campaign in campaigns:
        originals = [cells[campaign, method]["final_original"] for method in methods]
        if any(row != originals[0] for row in originals[1:]):
            raise ValueError("methods must share the same final Original evaluation")
    components = {"cps": lambda row: row["cps"], "q": lambda row: row["q"],
                  "c_b": lambda row: row["risks"]["c_b"],
                  "c_f": lambda row: row["risks"]["c_f"]}
    differences = {
        method: {component: [extract(cells[campaign, method]["final_selected"])
                              - extract(cells[campaign, method]["final_original"])
                              for campaign in campaigns]
                 for component, extract in components.items()}
        for method in methods}
    bootstrap = {method: {component: [] for component in components} for method in methods}
    rng = random.Random(seed)
    for _ in range(resamples):
        indices = [rng.randrange(len(campaigns)) for _ in campaigns]
        for method in methods:
            for component in components:
                bootstrap[method][component].append(mean(differences[method][component][i] for i in indices))
    output = {}
    for method in methods:
        deltas = differences[method]["cps"]
        observed = abs(sum(deltas))
        extreme = sum(abs(sum(sign * delta for sign, delta in zip(signs, deltas))) >= observed - 1e-12
                      for signs in product((-1, 1), repeat=len(campaigns)))
        output[method] = {
            "components": {component: {"mean_delta": mean(differences[method][component]),
                                       "ci95": [_percentile(bootstrap[method][component], .025),
                                                _percentile(bootstrap[method][component], .975)]}
                           for component in components},
            "p_exact_two_sided": extreme / (2 ** len(campaigns)),
        }
    running = 0.0
    for rank, method in enumerate(sorted(methods, key=lambda name: output[name]["p_exact_two_sided"])):
        running = max(running, min(1.0, (len(methods) - rank) * output[method]["p_exact_two_sided"]))
        output[method]["p_holm"] = running
    return {"independent_unit": "campaign", "campaign_count": len(campaigns),
            "bootstrap_seed": seed, "bootstrap_resamples": resamples,
            "multiplicity_family": list(methods), "methods": output}
