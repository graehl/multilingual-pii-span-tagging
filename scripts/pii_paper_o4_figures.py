#!/usr/bin/env python3
"""Render O4 paper comparisons and paired uncertainty from saved row counts."""

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from pii_paper_pooled_eval import EVIDENCE, ROOT, load_module, metrics, read_json, sha

sys.path.insert(0, str(Path.home() / "agents"))
import acli

DEST = EVIDENCE / "paper-o4-v1"
SOURCE = DEST / "scores.json.gz"
PRESIDIO_SWEEP = DEST / "presidio-threshold-v1/scores.json.gz"
OPERATING = DEST / "operating-points.json"
OPERATING_TRUST_REGION = DEST / "operating-points-trust-region.json"
REFINED = DEST / "refined-grid-v1/scores.json.gz"
FINE_BIAS = DEST / "fine-bias-v1/scores.json.gz"
# O2's rerun did not reproduce its saved sweeps, so it keeps the saved grid.
REFINED_SOURCES = {
    REFINED: ("gliner2", "gliner2-o4", "presidio"),
    FINE_BIAS: ("o4", "o3", "opf-openmed-multi2"),
}
TRUST_REGION_SEED = 20261009
TRUST_REGION_RESAMPLES = 10000
LANGS = ["en", "de", "es", "fr", "ar", "pt", "zh"]
# Reader-facing names of the scored populations. Silver-dev selected O4 and
# GL4's checkpoint; Silver-test never selected anything.
SELECTION = "ont3"
EVALUATIONS = {
    "gold7": ("human", None),
    "silver_test": ("heldout", None),
    "silver_test_seven": ("heldout", set(LANGS)),
}
# Swept systems in the main comparison, with their measured default point.
MAIN_SYSTEMS = ("o4", "o3", "ont2", "gliner2", "gliner2-o4", "presidio", "opf-openmed-multi2")


def items(point, coverage=80, typed=False):
    return point["typed"]["per_input"] if typed else point["regions"][str(coverage)]["per_input"]


def total(rows):
    return metrics([sum(row["counts"][i] for row in rows) for i in range(3)])


def view(report, population, panels):
    answer = {"N": report["populations"][population]["languages"], "scores": {}}
    for coverage in (80, 100):
        answer["scores"][str(coverage)] = {
            model: [
                dict(
                    threshold=p["threshold"],
                    panels={
                        key: total([r for r in items(p, coverage) if predicate(r)])
                        for key, predicate in panels.items()
                    },
                )
                for p in populations[population]["points"]
            ]
            for model, populations in report["systems"].items()
        }
    return answer


def fixed(report, model, population):
    threshold = 0.5 if model.startswith("gliner2") else 0
    return next(p for p in report["systems"][model][population]["points"] if p["threshold"] == threshold)


def merge_presidio_sweep(report):
    """Replace Presidio's fixed point by its score-threshold sweep.

    The sweep's threshold-0 point must reproduce the fixed point exactly,
    row by row, before it is accepted.
    """
    sweep = read_json(PRESIDIO_SWEEP)
    if sweep["status"] != "complete":
        raise ValueError("Complete Presidio sweep required")
    for population, data in report["systems"]["presidio"].items():
        points = sweep["systems"]["presidio"][population]["points"]
        zero = next(p for p in points if p["threshold"] == 0)
        for coverage in ("80", "100"):
            if zero["regions"][coverage] != data["points"][0]["regions"][coverage]:
                raise ValueError(f"Presidio sweep threshold 0 differs from the fixed point: {population}")
        data["points"] = points
    return report


def default_threshold(model):
    return 0.5 if model.startswith("gliner2") else 0


def merge_refined_grid(report):
    """Replace curves by their refined grids: halved confidence/score steps
    for GLiNER2, GL4 and Presidio, quarter O-bias steps for O3, O4 and PF.

    Each refined curve must reproduce every previously measured threshold
    row by row before it is accepted.
    """
    for path, models in REFINED_SOURCES.items():
        refined = read_json(path)
        if refined["status"] != "complete":
            raise ValueError(f"Complete refined-grid scores required: {path}")
        for model in models:
            merge_refined_system(report, refined, model)
    return report


def merge_refined_system(report, refined, model):
    for population, data in report["systems"][model].items():
        points = refined["systems"][model][population]["points"]
        by_threshold = {p["threshold"]: p for p in points}
        for old in data["points"]:
            new = by_threshold[old["threshold"]]
            for coverage in ("80", "100"):
                if new["regions"][coverage] != old["regions"][coverage]:
                    raise ValueError(f"Refined grid differs at {model}/{population}/{old['threshold']}")
        data["points"] = sorted(points, key=lambda p: p["threshold"])


def trust_region(groups, curve_counts, argmax_index):
    """Contiguous grid points around the argmax not reliably worse than it.

    `curve_counts` is (groups, grid points, 3) Silver-dev counts by source
    document; resample weights are shared by every system.
    """

    def f1(v):
        return np.divide(
            2 * v[..., 0],
            v[..., 1] + v[..., 2],
            out=np.zeros_like(v[..., 0]),
            where=v[..., 1] + v[..., 2] != 0,
        )

    resampled = np.einsum("bg,gkc->bkc", groups, curve_counts)
    delta = f1(resampled) - f1(resampled)[:, [argmax_index]]
    low, high = np.quantile(delta, [0.025, 0.975], axis=0)
    inside = (low <= 0) & (high >= 0)
    first = last = argmax_index
    while first > 0 and inside[first - 1]:
        first -= 1
    while last < len(inside) - 1 and inside[last + 1]:
        last += 1
    region = list(range(first, last + 1))
    middle = len(region) // 2
    if len(region) % 2 == 0:
        candidates = (region[middle - 1], region[middle])
        selected = min(candidates, key=lambda index: abs(index - argmax_index))
    else:
        selected = region[middle]
    return selected, (first, last), low, high


def resample_weights(groups):
    """Bootstrap counts over `groups` source documents, shared by every system scored on them."""
    rng = np.random.default_rng(TRUST_REGION_SEED)
    draws = rng.integers(0, groups, size=(TRUST_REGION_RESAMPLES, groups))
    return np.stack([np.bincount(draw, minlength=groups) for draw in draws]).astype(float)


def select_threshold(points, default, weights=None, group_index=None):
    """One system's operating point from its development curve `points`.

    The argmax of pooled 80%-overlap region F1, ties to the threshold nearest
    `default`; with bootstrap `weights` over the source groups numbered by
    `group_index`, the median of its trust region instead. Returns the chosen
    threshold, the argmax, the trust-region record (None for the argmax rule)
    and the development curve.
    """
    development = {p["threshold"]: total(items(p, 80)) for p in points}
    argmax = max(development, key=lambda t: (development[t]["F1"], -abs(t - default)))
    if weights is None:
        return argmax, argmax, None, development
    grid = sorted(development)
    by_threshold = {p["threshold"]: p for p in points}
    counts = np.zeros((len(group_index), len(grid), 3))
    for k, threshold in enumerate(grid):
        for row in items(by_threshold[threshold], 80):
            counts[group_index[row["group"]], k] += row["counts"]
    selected, (first, last), low, high = trust_region(weights, counts, grid.index(argmax))
    region = dict(
        argmax_threshold=argmax,
        bounds=[grid[first], grid[last]],
        width=grid[last] - grid[first],
        grid_points=last - first + 1,
        grid=grid,
        difference_from_argmax_ci95=[
            dict(threshold=t, low=float(lo), high=float(hi))
            for t, lo, hi in zip(grid, low, high, strict=True)
        ],
    )
    return grid[selected], argmax, region, development


def operating_point_sources(rule):
    """Paths and hashes of the score archives `operating_points` reads under `rule`."""
    archives = {"scores": SOURCE, "presidio_sweep": PRESIDIO_SWEEP}
    if rule == "trust-region":
        archives.update(refined_grid=REFINED, fine_bias=FINE_BIAS)
    return {key: {"path": str(path.relative_to(ROOT)), "sha256": sha(path)} for key, path in archives.items()}


def operating_points(report, rule="argmax", sources=None):
    """Fix each system's threshold on Silver-dev alone, then score it elsewhere.

    `argmax` maximizes pooled 80%-overlap region F1 on Silver-dev; ties go to
    the threshold nearest the system's default. `trust-region` bootstraps
    source documents to find the contiguous run of grid points around that
    argmax whose F1 difference interval includes zero, then takes its median
    grid point (the one nearer the argmax for an even count). No graphed
    population takes part in either choice. `sources`, when given, records
    the archives `report` came from; selection never reads them, so it can
    run on counts expanded from text-free receipts.
    """

    def scored(model, population, subset, coverage):
        return {
            p["threshold"]: total([r for r in items(p, coverage) if subset is None or r["lang"] in subset])
            for p in report["systems"][model][population]["points"]
        }

    weights = index = None
    if rule == "trust-region":
        group_ids = sorted({r["group"] for r in items(report["systems"]["o4"][SELECTION]["points"][0], 80)})
        index = {group: i for i, group in enumerate(group_ids)}
        weights = resample_weights(len(group_ids))

    systems = {}
    # O1 appears only in the pooled-Silver appendix figure; it is fitted but
    # takes no part in the paired comparisons.
    for model in (*MAIN_SYSTEMS, "ont1") if rule == "trust-region" else MAIN_SYSTEMS:
        default = default_threshold(model)
        chosen, argmax, region, development = select_threshold(
            report["systems"][model][SELECTION]["points"], default, weights, index
        )
        entry = dict(
            default_threshold=default,
            selected_threshold=chosen,
            selection_curve=[dict(threshold=t, **m) for t, m in development.items()],
            evaluations={},
        )
        if region is not None:
            entry["trust_region"] = region
        for name, (population, subset) in {"silver_dev": (SELECTION, None), **EVALUATIONS}.items():
            entry["evaluations"][name] = {}
            for coverage in (80, 100):
                curve = scored(model, population, subset, coverage)
                best = max(curve, key=lambda t: curve[t]["F1"])
                entry["evaluations"][name][str(coverage)] = dict(
                    selected=curve[chosen],
                    default=curve[default],
                    best_on_curve=dict(threshold=best, **curve[best]),
                    **({"argmax": curve[argmax]} if region is not None else {}),
                )
        systems[model] = entry
        print(
            f"[operating-point] {model} threshold={chosen:g} "
            + " ".join(
                f"{name}={entry['evaluations'][name]['80']['selected']['F1'] * 100:.2f}"
                for name in ("silver_dev", *EVALUATIONS)
            ),
            flush=True,
        )
    populations = {}
    for name, (population, subset) in {"silver_dev": (SELECTION, None), **EVALUATIONS}.items():
        languages = {
            lang: n
            for lang, n in report["populations"][population]["languages"].items()
            if subset is None or lang in subset
        }
        populations[name] = dict(source_population=population, N=sum(languages.values()), languages=languages)

    def comparisons(thresholds):
        return [
            dict(
                evaluation=name,
                **paired(report, "o4", other, [population], coverage=80, thresholds=thresholds),
            )
            for name, population in (("gold7", "human"), ("silver_test", "heldout"))
            for other in MAIN_SYSTEMS
            if other != "o4"
        ]

    rule_text = (
        "Per system, the outside-label bias (O-family, PF) or confidence/score threshold "
        "(GLiNER2, Presidio) maximizing pooled untyped-region F1 at 80% mutual overlap on "
        "Silver-dev alone; ties go to the threshold nearest the default. The fixed threshold "
        "is then scored unchanged on Gold-7 and Silver-test."
    )
    extra = {}
    if rule == "trust-region":
        rule_text = (
            "Per system, find the Silver-dev argmax of pooled untyped-region F1 at 80% mutual overlap "
            "(ties to the default). Bootstrap Silver-dev source documents "
            f"({TRUST_REGION_RESAMPLES:,} resamples, seed {TRUST_REGION_SEED}, weights shared by all "
            "systems) for F1(t) - F1(argmax) at every grid point. The trust region is the contiguous run "
            "of grid points around the argmax whose 95% percentile interval includes zero; the fixed "
            "threshold is its median grid point, the one nearer the argmax for an even count. "
            "GLiNER2 and Presidio use halved-step grids emulated from saved outputs. O3, O4 and PF use "
            "quarter-step O biases over -4..4 from a rerun that reproduced every saved sweep point row by "
            "row through the serving postprocessors. O2 keeps its saved grid: its rerun differed from the "
            "saved predictions on about 1% of rows, so the two were not mixed."
        )
        extra["paired_at_argmax"] = comparisons(
            {model: entry["trust_region"]["argmax_threshold"] for model, entry in systems.items()}
        )
    return dict(
        schema="pii-paper-o4-operating-points/v1",
        **({} if sources is None else {"sources": sources}),
        rule=rule_text,
        names={
            "silver_dev": "Silver-dev: 659 LLM-annotated Ont3 rows that also selected O4 and GL4's checkpoint",
            "gold7": "Gold-7: 1,283 human-annotated publisher-test segments in seven languages",
            "silver_test": "Silver-test: 542 single-teacher Ont3 rows, never used for any selection",
            "silver_test_seven": "Silver-test rows in Gold-7's seven languages",
        },
        metric_note="F1, P and R are fractions from counts pooled over rows; 100 is exact-region overlap",
        populations=populations,
        systems=systems,
        paired_at_selected=comparisons(
            {model: entry["selected_threshold"] for model, entry in systems.items()}
        ),
        **extra,
    )


def pool_ont3(report):
    """Combine collection batches at common measured thresholds, before F1."""
    batches = ("ont3", "heldout")
    languages = Counter()
    for batch in batches:
        languages.update(report["populations"][batch]["languages"])
    expected = sum(languages.values())
    if expected != 1201:
        raise ValueError(f"Unexpected Ont3 pool size: {expected}")
    for model, populations in report["systems"].items():
        sources = [{p["threshold"]: p for p in populations[b]["points"]} for b in batches]
        thresholds = sorted(sources[0].keys() & sources[1].keys())
        if not thresholds:
            raise ValueError(f"No shared thresholds for {model}")
        points = []
        for threshold in thresholds:
            point = dict(threshold=threshold, regions={})
            for coverage in (80, 100):
                records = [r for source in sources for r in items(source[threshold], coverage)]
                if len(records) != expected or len({r["id"] for r in records}) != expected:
                    raise ValueError(f"Ont3 membership mismatch: {model}/{threshold}")
                point["regions"][str(coverage)] = dict(per_input=records, metrics=total(records))
            if all("typed" in source[threshold] for source in sources):
                records = [r for source in sources for r in items(source[threshold], typed=True)]
                point["typed"] = dict(per_input=records, metrics=total(records))
            points.append(point)
        populations["ont3"] = dict(points=points)
        del populations["heldout"]
    report["populations"]["ont3"] = dict(
        N=expected, languages=dict(languages), split="reused development; pooled annotation collections"
    )
    del report["populations"]["heldout"]
    return report


def paired(report, candidate, control, populations, typed=False, *, coverage=100, thresholds=None):
    """Source-group bootstrap of an F1 difference; `thresholds` maps each
    system to a measured point, defaulting to its fixed default point."""

    def at(model, population):
        if thresholds is None:
            return fixed(report, model, population)
        return next(
            p for p in report["systems"][model][population]["points"] if p["threshold"] == thresholds[model]
        )

    rng = np.random.default_rng(20260929)
    totals, boot = np.zeros((2, 3)), np.zeros((10000, 2, 3))
    group_count = 0
    for population in populations:
        left = {r["id"]: r for r in items(at(candidate, population), coverage, typed)}
        right = {r["id"]: r for r in items(at(control, population), coverage, typed)}
        if left.keys() != right.keys():
            raise ValueError("Paired input mismatch")
        groups = defaultdict(lambda: np.zeros((2, 3)))
        for key, row in left.items():
            if row["group"] != right[key]["group"]:
                raise ValueError("Paired source-group mismatch")
            groups[row["group"]] += np.array([row["counts"], right[key]["counts"]])
        array = np.array(list(groups.values()))
        totals += array.sum(axis=0)
        group_count += len(array)
        for start in range(0, 10000, 100):
            indices = rng.integers(0, len(array), size=(100, len(array)))
            boot[start : start + 100] += array[indices].sum(axis=1)

    def f1(v):
        return np.divide(
            2 * v[..., 0],
            v[..., 1] + v[..., 2],
            out=np.zeros_like(v[..., 0]),
            where=v[..., 1] + v[..., 2] != 0,
        )

    estimates = f1(totals)
    delta = f1(boot)[:, 0] - f1(boot)[:, 1]
    return dict(
        candidate=candidate,
        control=control,
        populations=populations,
        metric="exact typed"
        if typed
        else "exact regions"
        if coverage == 100
        else f"{coverage}% overlap regions",
        **({} if thresholds is None else {"thresholds": {m: thresholds[m] for m in (candidate, control)}}),
        source_groups=group_count,
        candidate_f1=float(estimates[0]),
        control_f1=float(estimates[1]),
        delta=float(estimates[0] - estimates[1]),
        ci95=np.quantile(delta, [0.025, 0.975]).tolist(),
        p_two_sided=float(
            min(1, 2 * min((np.sum(delta <= 0) + 1) / 10001, (np.sum(delta >= 0) + 1) / 10001))
        ),
        win_rate=float(np.mean(delta > 0)),
        resamples=10000,
        selection_caveat="Conditional on development selection; no correction for recipe search",
    )


def main():
    parser = acli.argument_parser(description=__doc__, capabilities=("complete",))
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument(
        "--main-only",
        action="store_true",
        help="Write only the operating points and the main comparison figure family; "
        "leave summary.json, the Ont3 aggregate and gold-share outputs untouched",
    )
    parser.add_argument(
        "--operating-point-rule",
        choices=("argmax", "trust-region"),
        default="argmax",
        help="How each system's threshold is fixed on Silver-dev. trust-region also uses the "
        "refined GLiNER2/Presidio grids, writes operating-points-trust-region.json and the -tr and "
        "-compare figure variants, and implies --main-only",
    )
    acli.add_standard_args(parser)
    acli.maybe_complete(parser)
    args = parser.parse_args()
    trust = args.operating_point_rule == "trust-region"
    report = merge_presidio_sweep(read_json(SOURCE))
    if report["status"] != "complete":
        raise ValueError("Complete matrix required")
    if trust:
        report = merge_refined_grid(report)
        args.main_only = True
    operating = operating_points(
        report, args.operating_point_rule, sources=operating_point_sources(args.operating_point_rule)
    )
    (OPERATING_TRUST_REGION if trust else OPERATING).write_text(json.dumps(operating, indent=2) + "\n")
    gold7 = view(
        report,
        "human",
        {"overall": lambda r: True, **{lang: lambda r, lang=lang: r["lang"] == lang for lang in LANGS}},
    )
    silver_test = view(report, "heldout", {"all": lambda r: True, "seven": lambda r: r["lang"] in LANGS})
    if not args.main_only:
        report = pool_ont3(report)
        write_summary(report)
    if args.no_render:
        return
    plot = load_module("o4_render", EVIDENCE / "priority9-shared-v1/render-comparison.py")
    plot.CODES.update(o4="O4", o3="O3", **{"gliner2-o4": "GL4", "opf-openmed-multi2": "PF"})
    models = {
        "o4": ("XLM-R", "#009E73", "-", "v"),
        "o3": ("XLM-R", "#56B4E9", "--", "^"),
        "ont2": ("XLM-R, Ont2", "#0072B2", "-.", "o"),
        "gliner2": ("Stock GLiNER2", "#777777", "--", "s"),
        "gliner2-o4": ("Adapted GLiNER2", "#CC79A7", ":", "D"),
        "presidio": ("Presidio", "#D55E00", (0, (3, 1, 1, 1, 1, 1)), "X"),
        "opf-openmed-multi2": ("OpenMed PF v2", "#AA3377", (0, (1, 1)), "p"),
    }
    sizes = operating["populations"]
    gold7["aggregate_label"] = f"Gold-7 · seven languages · N = {sizes['gold7']['N']:,}"
    layout = dict(
        prefix="o4-human-ont3",
        models=models,
        languages=LANGS,
        recall_floor=0.4,
        extra_row=(
            silver_test,
            [
                ("seven", f"Silver-test · seven languages\nN = {sizes['silver_test_seven']['N']:,}"),
                (
                    "all",
                    f"Silver-test · {len(sizes['silver_test']['languages'])} languages\n"
                    f"N = {sizes['silver_test']['N']:,}",
                ),
            ],
        ),
    )
    selected = {model: operating["systems"][model]["selected_threshold"] for model in models}
    plot.render(
        gold7,
        **layout,
        coverages=(80, 100),
        operating_points=selected,
        operating_label="Fixed on Silver-dev",
        stem_suffix="-tr" if trust else "",
    )
    if trust:
        # Author-review comparison only: argmax rings, trust-region squares
        # and shaded trust-region extents on the same curves.
        regions = {model: operating["systems"][model]["trust_region"] for model in models}
        plot.render(
            gold7,
            **layout,
            coverages=(80,),
            operating_points={model: region["argmax_threshold"] for model, region in regions.items()},
            operating_label="Argmax",
            secondary_points=selected,
            secondary_label="Trust region",
            extents={model: tuple(region["bounds"]) for model, region in regions.items()},
            stem_suffix="-compare",
        )
    all_models = {**models, "ont1": plot.MODELS["ont1"]}
    if trust:
        # Appendix pooled-Silver figure, same layout as o4-ont3-all, with
        # swept Presidio, refined grids and trust-region rings.
        report = pool_ont3(report)
        data = view(report, "ont3", {"overall": lambda r: True})
        data["aggregate_label"] = (
            f"Silver-dev + Silver-test · 35 languages · N = {report['populations']['ont3']['N']:,}"
        )
        plot.render(
            data,
            prefix="o4-ont3-all",
            models=all_models,
            aggregate_only=True,
            recall_floor=0.4,
            coverages=(80,),
            operating_points={
                model: operating["systems"][model]["selected_threshold"] for model in all_models
            },
            operating_label="Fixed on Silver-dev",
            stem_suffix="-tr",
        )
    if args.main_only:
        return
    for population in ("ont3",):
        data = view(report, population, {"overall": lambda r: True})
        data["aggregate_label"] = f"Ont3 · 35 languages · N = {report['populations'][population]['N']:,}"
        plot.render(
            data,
            prefix=f"o4-{population}-all",
            models=all_models,
            aggregate_only=True,
            recall_floor=0.4,
            coverages=(80, 100),
        )
    # render() configured matplotlib (backend, rcParams); it imports pyplot lazily.
    import matplotlib.pyplot as plt

    gold_share(plt)


def write_summary(report):
    summary = dict(source_sha256=sha(SOURCE), populations=report["populations"], systems={}, paired=[])
    for model, populations in report["systems"].items():
        summary["systems"][model] = {}
        for population, data in populations.items():
            summary["systems"][model][population] = {
                str(coverage): {
                    "maximum": max(
                        (
                            dict(threshold=p["threshold"], **p["regions"][str(coverage)]["metrics"])
                            for p in data["points"]
                        ),
                        key=lambda p: p["F1"],
                    ),
                    "fixed": fixed(report, model, population)["regions"][str(coverage)]["metrics"],
                }
                for coverage in (80, 100)
            }
            if "typed" in data["points"][0]:
                summary["systems"][model][population]["typed_fixed"] = fixed(report, model, population)[
                    "typed"
                ]["metrics"]
        print(
            f"[summary] {model} "
            + " ".join(
                f"{p}={summary['systems'][model][p]['80']['maximum']['F1'] * 100:.2f}" for p in populations
            ),
            flush=True,
        )
    for populations, typed in (
        (["human"], False),
        (["ont3"], False),
        (["ont3"], True),
    ):
        summary["paired"].append(paired(report, "o4", "o3", populations, typed))
    for population in ("human", "ont3"):
        summary["paired"].append(paired(report, "gliner2-o4", "gliner2", [population]))
    (DEST / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def gold_share(plt):
    path = EVIDENCE / "tag-status-replay-v2/leaderboard-v6-20260928.md"
    data = []
    pattern = re.compile(r"gs30-titles-v6(?:-g(30|40|45|50|60))?(?:-seed([23]))?-4000-none")
    for line in path.read_text().splitlines():
        fields = [f.strip() for f in line.split("|")]
        if len(fields) != 8:
            continue
        match = pattern.fullmatch(fields[2].strip("`"))
        if match:
            data.append(
                dict(
                    share=int(match[1] or 35),
                    seed={None: 173, "2": 20260927, "3": 20260928}[match[2]],
                    weighted=float(fields[3]),
                    typed=float(fields[4]),
                    regions=float(fields[5]),
                    human=float(fields[6]),
                )
            )
    counts = {share: sum(r["share"] == share for r in data) for share in (30, 35, 40, 45, 50, 60)}
    if counts != {30: 2, 35: 3, 40: 3, 45: 2, 50: 3, 60: 1}:
        raise ValueError(f"Unexpected gold-share conditions: {counts}")
    (DEST / "gold-share.json").write_text(
        json.dumps(dict(source=str(path), sha256=sha(path), rows=data), indent=2) + "\n"
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), layout="constrained")
    for ax, keys in zip(axes, (("weighted",), ("typed", "human")), strict=True):
        for key in keys:
            color = {"weighted": "#009E73", "typed": "#0072B2", "human": "#CC79A7"}[key]
            for row in data:
                ax.plot(row["share"], row[key], ".", color=color, alpha=0.6, markersize=5)
            xs = sorted(counts)
            ys = [np.mean([r[key] for r in data if r["share"] == x]) for x in xs]
            ax.plot(
                xs,
                ys,
                "o-",
                color=color,
                markersize=3,
                label={"weighted": "Selection criterion", "typed": "Typed Ont3", "human": "Human gold"}[key],
            )
        ax.set(xlabel="Human-gold sampling (%)", ylabel="F1 (%)", xticks=[30, 40, 50, 60])
        ax.grid(alpha=0.18)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=9)
    for suffix in ("svg", "pdf", "png"):
        fig.savefig(EVIDENCE.parent / f"figures/o4-gold-share.{suffix}", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
