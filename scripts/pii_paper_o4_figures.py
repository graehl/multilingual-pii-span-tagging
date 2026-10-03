#!/usr/bin/env python3
"""Render O4 paper comparisons and paired uncertainty from saved row counts."""

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from pii_paper_pooled_eval import EVIDENCE, load_module, metrics, read_json, sha

sys.path.insert(0, str(Path.home() / "agents"))
import acli

DEST = EVIDENCE / "paper-o4-v1"
SOURCE = DEST / "scores.json.gz"
LANGS = ["en", "de", "es", "fr", "ar", "pt", "zh"]


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


def paired(report, candidate, control, populations, typed=False):
    rng = np.random.default_rng(20260929)
    totals, boot = np.zeros((2, 3)), np.zeros((10000, 2, 3))
    group_count = 0
    for population in populations:
        left = {r["id"]: r for r in items(fixed(report, candidate, population), 100, typed)}
        right = {r["id"]: r for r in items(fixed(report, control, population), 100, typed)}
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
        metric="exact typed" if typed else "exact regions",
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
    acli.add_standard_args(parser)
    acli.maybe_complete(parser)
    args = parser.parse_args()
    report = pool_ont3(read_json(SOURCE))
    if report["status"] != "complete":
        raise ValueError("Complete matrix required")
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
        "presidio": ("Presidio", "#D55E00", "None", "X"),
        "opf-openmed-multi2": ("OpenMed PF v2", "#AA3377", (0, (1, 1)), "p"),
    }
    human = view(
        report,
        "human",
        {"overall": lambda r: True, **{lang: lambda r, lang=lang: r["lang"] == lang for lang in LANGS}},
    )
    human["aggregate_label"] = "Human gold · seven languages · N = 1,283"
    extra = view(report, "ont3", {"dev": lambda r: True, "seven": lambda r: r["lang"] in LANGS})
    plot.render(
        human,
        prefix="o4-human-ont3",
        models=models,
        languages=LANGS,
        recall_floor=0.4,
        extra_row=(
            extra,
            [
                ("seven", "Ont3 · seven languages\nN = 302"),
                ("dev", "Ont3 · 35 languages\nN = 1,201"),
            ],
        ),
        coverages=(80, 100),
    )
    all_models = {**models, "ont1": plot.MODELS["ont1"]}
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
    gold_share(plot.plt)


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
