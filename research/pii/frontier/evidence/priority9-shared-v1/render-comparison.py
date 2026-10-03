"""Reproduce Figure 1 from the five saved sweeps: python render-comparison.py.

Uses the existing redaction-region projection and maximum-cardinality matcher.
No model inference, gold-driven threshold selection, or interpolated scores.
"""

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
sys.path.insert(0, str(ROOT / "scripts"))
from pii_eval import _maximum_matches, merge_regions

MODELS = {
    "gliner2": ("GLiNER2", "#777777", "--", "s"),
    "gliner2-tuned": ("GL Ont3-tuned", "#CC79A7", ":", "D"),
    "ont1": ("XLM-R · Ont1", "#E69F00", "-.", "^"),
    "ont2": ("XLM-R · Ont2", "#0072B2", "-", "o"),
    "ont3": ("XLM-R · Ont3", "#009E73", (0, (5, 2, 1, 2, 1, 2)), "v"),
}
LANGUAGES = {
    "en": "English",
    "de": "German",
    "es": "Spanish",
    "fr": "French",
    "ar": "Arabic",
    "pt": "Portuguese",
    "ko": "Korean",
    "vi": "Vietnamese",
    "zh": "Chinese",
}
REFERENCES = {"person_reference", "organization_reference"}
CODES = {"gliner2": "GL", "gliner2-tuned": "GL3", "ont1": "O1", "ont2": "O2", "ont3": "O3", "presidio": "PR"}


def regions(spans, text):
    for span in spans:
        assert 0 <= span["start"] < len(text) and span["start"] < span["end"], span
    return merge_regions([s for s in spans if s.get("label", s.get("type")) not in REFERENCES], text)


def compatible(a, b, coverage):
    intersection = max(0, min(a["end"], b["end"]) - max(a["start"], b["start"]))
    return all(intersection * 100 >= coverage * (s["end"] - s["start"]) for s in (a, b))


def metrics(counts):
    tp, pred, gold = counts
    p, r = tp / pred if pred else 0, tp / gold if gold else 0
    return {
        "tp": tp,
        "predicted": pred,
        "gold": gold,
        "P": p,
        "R": r,
        "F1": 2 * p * r / (p + r) if p + r else 0,
    }


def main():
    gold_path = HERE / "evaluation.jsonl"
    gold_hash = hashlib.sha256(gold_path.read_bytes()).hexdigest()
    gold = [json.loads(line) for line in gold_path.read_text().splitlines()]
    assert len(gold) == len({r["id"] for r in gold}) == len({r["text"] for r in gold}) == 607
    counts = Counter(r["lang"] for r in gold)
    assert set(counts) == set(LANGUAGES)
    assert not any(s["type"] in REFERENCES for r in gold for s in r["spans"])
    # The source gold has no optional-reference targets. Reference predictions
    # are excluded; there are consequently no neutral named-on-reference cases.
    evidence = {
        "input_sha256": gold_hash,
        "N": dict(counts),
        "split": "reused development",
        "projection": "untyped maximal redaction regions; whitespace gaps merged",
        "reference_policy": "optional",
        "aggregate": "pooled counts across 9 languages",
        "models": {},
        "scores": {},
        "clipped_endpoints": {},
    }
    for model in MODELS:
        name = "ont3-context" if model == "ont3" else model
        path = HERE / f"paper-priority9-{name}-sweep.json"
        sweep = json.loads(path.read_text())
        assert sweep["input_sha256"] == gold_hash and sweep["rows"] == len(gold)
        evidence["models"][model] = {
            "path": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "checkpoint": sweep["model_path"],
            "context": sweep.get("context", "complete input; no external neighboring text"),
        }
        clipped = set()
        for predictions in sweep["points"].values():
            for row, prediction in zip(gold, predictions, strict=True):
                for span in prediction["preds"]:
                    if span["end"] > len(row["text"]):
                        clipped.add((row["id"], span["start"], span["end"], len(row["text"])))
        evidence["clipped_endpoints"][model] = sorted(clipped)
        for coverage in (80, 90, 100):
            scored = []
            for threshold, predictions in sorted(sweep["points"].items(), key=lambda item: float(item[0])):
                assert len(predictions) == len(gold)
                totals = {lang: [0, 0, 0] for lang in LANGUAGES}
                per_input = []
                for row, prediction in zip(gold, predictions, strict=True):
                    assert row["id"] == prediction["id"]
                    expected = regions(row["spans"], row["text"])
                    actual = regions(prediction["preds"], row["text"])
                    tp = _maximum_matches(expected, actual, lambda a, b: compatible(a, b, coverage))
                    triple = [tp, len(actual), len(expected)]
                    totals[row["lang"]] = [a + b for a, b in zip(totals[row["lang"]], triple)]
                    per_input.append({"id": row["id"], "counts": triple})
                pooled = [sum(values[i] for values in totals.values()) for i in range(3)]
                scored.append(
                    {
                        "threshold": float(threshold),
                        "per_input": per_input,
                        "panels": {
                            **{lang: metrics(v) for lang, v in totals.items()},
                            "overall": metrics(pooled),
                        },
                    }
                )
            evidence["scores"].setdefault(str(coverage), {})[model] = scored
    (HERE / "scores.json").write_text(json.dumps(evidence, separators=(",", ":")) + "\n")
    render(evidence)


def curve_coordinates(points):
    """Break long terminal backward strokes, retaining every measured marker."""
    distinct = []
    for point in points:
        coordinate = (point["R"], point["P"])
        if not distinct or coordinate != distinct[-1]:
            distinct.append(coordinate)
    coordinates = []
    for index, point in enumerate(distinct):
        if index and index in (1, len(distinct) - 1):
            previous = distinct[index - 1]
            dr, dp = point[0] - previous[0], point[1] - previous[1]
            if abs(dr) > 0.2 and dr * dp > 0:
                coordinates.append((float("nan"), float("nan")))
        coordinates.append(point)
    return coordinates


def render(
    evidence,
    prefix="priority9-five-model",
    *,
    models=None,
    aggregate_only=False,
    languages=None,
    precision_floor=0.4,
    recall_floor=0.0,
    extra_row=None,
    coverages=(80, 90, 100),
):
    """Render one comparison family. `extra_row` = (evidence, [(panel key,
    title), ...]) adds one row of three aggregate panels below a
    seven-language view, sharing its legend."""
    # Plotting libraries load here so scoring callers need no plotting stack.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from adjustText import adjust_text
    from matplotlib.font_manager import FontProperties
    from matplotlib.lines import Line2D
    from matplotlib.transforms import ScaledTranslation

    models = MODELS if models is None else models
    languages = LANGUAGES if languages is None else languages
    counts = evidence["N"]
    language_labels = evidence.get("language_labels", LANGUAGES)
    seven_panels = not aggregate_only and len(languages) == 7
    count = sum(counts.values())
    output = ROOT / "research/pii/frontier/figures"
    plt.rcParams.update(
        {"font.family": "DejaVu Sans", "font.size": 10, "svg.hashsalt": "priority9-five-model-v1"}
    )
    # Pooled panel spans two cells beside the first language, then two full
    # rows of three languages.
    cells = [(0, 2)] + [(1 + i // 3, i % 3) for i in range(6)]
    for coverage in coverages:
        if extra_row is not None:
            row_evidence, row_panels = extra_row
            if not seven_panels or len(row_panels) not in (2, 3):
                raise ValueError("An extra row needs a seven-language view and two or three panels")
            fig = plt.figure(figsize=(7.2, 7.4))
            outer = fig.add_gridspec(
                2, 1, height_ratios=(3, 1.05), left=0.075, right=0.985, bottom=0.12, top=0.965, hspace=0.3
            )
            grid = outer[0].subgridspec(3, 3, wspace=0.45, hspace=0.62)
            panels = [(fig.add_subplot(grid[0, :2]), evidence, "overall", evidence["aggregate_label"], 11)]
            for lang, (row, column) in zip(languages, cells, strict=True):
                title = f"{language_labels[lang]} · N = {counts[lang]}"
                panels.append((fig.add_subplot(grid[row, column]), evidence, lang, title, 9.5))
            grid = outer[1].subgridspec(1, 3, wspace=0.45)
            for column, (key, title) in enumerate(row_panels):
                cell = (
                    (grid[0, :2] if column == 0 else grid[0, 2]) if len(row_panels) == 2 else grid[0, column]
                )
                panels.append((fig.add_subplot(cell), row_evidence, key, title, 9.5))
        elif aggregate_only:
            fig = plt.figure(figsize=(7.2, 3.8))
            legend_rows = (len(models) + 2) // 3
            grid = fig.add_gridspec(1, 3, left=0.075, right=0.985, bottom=0.12 + 0.06 * legend_rows, top=0.90)
            axes = {"overall": fig.add_subplot(grid[0, :])}
        elif seven_panels:
            fig = plt.figure(figsize=(7.2, 5.9))
            grid = fig.add_gridspec(
                3, 3, left=0.075, right=0.985, bottom=0.16, top=0.95, wspace=0.45, hspace=0.5
            )
            axes = {"overall": fig.add_subplot(grid[0, :2])}
            for lang, (row, column) in zip(languages, cells, strict=True):
                axes[lang] = fig.add_subplot(grid[row, column])
        else:
            fig = plt.figure(figsize=(7.2, 7.8))
            grid = fig.add_gridspec(
                4,
                3,
                height_ratios=(1.35, 1, 1, 1),
                left=0.075,
                right=0.985,
                bottom=0.13,
                top=0.95,
                wspace=0.45,
                hspace=0.5,
            )
            axes = {"overall": fig.add_subplot(grid[0, :])}
            for i, lang in enumerate(languages):
                axes[lang] = fig.add_subplot(grid[1 + i // 3, i % 3])
        if extra_row is None:
            overall_title = evidence.get(
                "aggregate_label",
                (
                    f"35-language Ont3 collection · N = {count}"
                    if aggregate_only
                    else f"{len(languages)} languages · N = {count}"
                ),
            )
            panels = [
                (ax, evidence, lang, overall_title, 12)
                if lang == "overall"
                else (ax, evidence, lang, f"{language_labels[lang]} · N = {counts[lang]}", 10)
                for lang, ax in axes.items()
            ]
        for ax, view, lang, title, title_size in panels:
            # Small panels get lighter strokes so curves stay separable.
            small = lang != "overall"
            linewidth, markersize, point_size = (1.0, 1.9, 3.6) if small else (1.6, 3, 5)
            labels, targets, occupied, fixed_labels = [], [], [], []
            outside_view = 0
            for model, (label, color, style, marker) in models.items():
                points = [p["panels"][lang] for p in view["scores"][str(coverage)][model]]
                # Precision is undefined when no region is predicted. Keep the
                # measured counts in evidence, but do not draw a fabricated
                # segment to (recall=0, precision=0).
                points = [p for p in points if p["predicted"] > 0]
                if not points:
                    ax.text(
                        0.03,
                        0.06 + 0.10 * outside_view,
                        f"{CODES[model]}: no predictions",
                        transform=ax.transAxes,
                        fontsize=7,
                        color=color,
                    )
                    outside_view += 1
                    continue
                coordinates = curve_coordinates(points)
                ax.plot(
                    [p[0] for p in coordinates],
                    [p[1] for p in coordinates],
                    color=color,
                    linestyle=style,
                    marker=marker,
                    markersize=point_size if model == "presidio" else markersize,
                    linewidth=linewidth,
                    alpha=0.88,
                    label=label,
                )
                visible = [p for p in points if p["P"] >= precision_floor and p["R"] >= recall_floor]
                if not visible:
                    if model == "presidio":
                        # Mark an off-scale fixed point just outside the axis it misses.
                        point = points[0]
                        edge = (
                            point["R"] if point["R"] >= recall_floor else recall_floor - 0.018,
                            point["P"] if point["P"] >= precision_floor else precision_floor - 0.018,
                        )
                        ax.plot(
                            [edge[0]],
                            [edge[1]],
                            linestyle="None",
                            marker=marker,
                            markersize=point_size,
                            color=color,
                            clip_on=False,
                        )
                        # Label it just inside the panel so the code stays visible.
                        targets.append(edge)
                        labels.append(
                            ax.text(
                                max(edge[0], recall_floor) + 0.03,
                                max(edge[1], precision_floor) + 0.04,
                                CODES[model],
                                fontsize=7.2,
                                color=color,
                                bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=0.3),
                            )
                        )
                        continue
                    if model.startswith("opf"):
                        continue
                    ax.text(
                        0.03,
                        0.06 + 0.10 * outside_view,
                        f"{CODES[model]}: precision < {precision_floor:.2f}"
                        if max(p["P"] for p in points) < precision_floor
                        else f"{CODES[model]}: outside view",
                        transform=ax.transAxes,
                        fontsize=7,
                        color=color,
                    )
                    outside_view += 1
                    continue
                neutral = 0.5 if model.startswith("gliner2") else 0
                anchor = next(
                    p["panels"][lang]
                    for p in view["scores"][str(coverage)][model]
                    if p["threshold"] == neutral
                )
                anchor = min(visible, key=lambda p: (p["P"] - anchor["P"]) ** 2 + (p["R"] - anchor["R"]) ** 2)
                if model == "presidio":
                    text = ax.annotate(
                        "PR",
                        (anchor["R"], anchor["P"]),
                        xytext=(5, 5),
                        textcoords="offset points",
                        color=color,
                        fontsize=7.2,
                    )
                    fixed_labels.append(text)
                    occupied.extend((p["R"], p["P"]) for p in points)
                    continue
                if model == "o4":
                    anchor = max(visible, key=lambda p: p["F1"])
                    text = ax.annotate(
                        "O4",
                        (anchor["R"], anchor["P"]),
                        xytext=(4, 4),
                        textcoords="offset points",
                        color=color,
                        fontsize=7.2,
                        bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=0.3),
                        arrowprops=dict(arrowstyle="-", color=color, lw=0.5),
                    )
                    fixed_labels.append(text)
                    occupied.extend((p["R"], p["P"]) for p in points)
                    continue
                if lang == "fr" and model == "gliner2" and "o4" in models:
                    anchor = max(visible, key=lambda p: p["P"])
                    text = ax.annotate(
                        "GL",
                        (anchor["R"], anchor["P"]),
                        xytext=(-5, 5),
                        textcoords="offset points",
                        ha="right",
                        va="bottom",
                        color=color,
                        fontsize=7.2,
                        bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=0.3),
                        arrowprops=dict(arrowstyle="-", color=color, lw=0.5),
                    )
                    fixed_labels.append(text)
                    occupied.extend((p["R"], p["P"]) for p in points)
                    continue
                if model.startswith("ont") or model == "o3":
                    # Label the separated high-recall tail, not the crowded knee.
                    candidates = list(visible)
                    recall_midpoint = (min(p["R"] for p in candidates) + max(p["R"] for p in candidates)) / 2
                    candidates = [p for p in candidates if p["R"] >= recall_midpoint]
                    precision_midpoint = (
                        min(p["P"] for p in candidates) + max(p["P"] for p in candidates)
                    ) / 2
                    candidates = [p for p in candidates if p["P"] <= precision_midpoint]
                    other_curves = []
                    for other in models:
                        if other == model:
                            continue
                        curve = [p["panels"][lang] for p in view["scores"][str(coverage)][other]]
                        for a, b in zip(curve, curve[1:]):
                            other_curves.extend(
                                (a["R"] + t * (b["R"] - a["R"]), 2 * (a["P"] + t * (b["P"] - a["P"])))
                                for t in np.linspace(0, 1, 40)
                            )
                    anchor = max(
                        candidates,
                        key=lambda p: (
                            min(np.hypot(p["R"] - r, 2 * p["P"] - precision) for r, precision in other_curves)
                            + 0.03 * p["R"]
                        ),
                    )
                if lang == "ar" and model == "o3":
                    text = ax.annotate(
                        "O3",
                        (anchor["R"], anchor["P"]),
                        xytext=(5, 0),
                        textcoords="offset points",
                        va="center",
                        color=color,
                        fontsize=7.2,
                        bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=0.3),
                        arrowprops=dict(arrowstyle="-", color=color, lw=0.5),
                    )
                    fixed_labels.append(text)
                    occupied.extend((p["R"], p["P"]) for p in points)
                    continue
                targets.append((anchor["R"], anchor["P"]))
                occupied.extend((p["R"], p["P"]) for p in points)
                # A fixed point often sits inside the curves' high-recall knee,
                # where the placer cannot move its label clear; start it farther
                # out so the label gets an arrow back to the marker.
                dr, dp = (0.14, 0.07) if model == "presidio" else (0.04, 0.025)
                labels.append(
                    ax.text(
                        max(anchor["R"] - dr, recall_floor + 0.02),
                        max(anchor["P"] - dp, precision_floor + 0.02),
                        CODES[model],
                        fontsize=7.2,
                        color=color,
                        bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=0.3),
                    )
                )
            ax.set(
                xlim=(recall_floor, 1.02),
                ylim=(precision_floor, 1.02),
                xticks=[recall_floor, (1 + recall_floor) / 2, 1],
                yticks=[precision_floor, (1 + precision_floor) / 2, 1],
            )
            ax.grid(alpha=0.18)
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_title(title, fontsize=title_size)
            ax.tick_params(labelsize=9)
            ax.set_xlabel("Recall", fontsize=9)
            ax.xaxis.set_label_coords(0.75, 0)
            ax.xaxis.label.set_transform(ax.transAxes + ScaledTranslation(0, -7 / 72, fig.dpi_scale_trans))
            ax.xaxis.label.set_verticalalignment("top")
            label_width = fig.canvas.get_renderer().get_text_width_height_descent(
                "Precision", FontProperties(size=9), False
            )[0]
            fits = label_width < ax.bbox.height / 2 - 18 / 72 * fig.dpi
            ax.set_ylabel("Precision" if fits else "P", fontsize=9, rotation=90 if fits else 0)
            ax.yaxis.set_label_coords(0, 0.75)
            ax.yaxis.label.set_transform(ax.transAxes + ScaledTranslation(-12 / 72, 0, fig.dpi_scale_trans))
            ax.yaxis.label.set_horizontalalignment("center")
            ax.yaxis.label.set_verticalalignment("center")
            np.random.seed(173)
            adjust_text(
                labels,
                objects=fixed_labels or None,
                x=[p[0] for p in occupied],
                y=[p[1] for p in occupied],
                target_x=[p[0] for p in targets],
                target_y=[p[1] for p in targets],
                ax=ax,
                iter_lim=300,
                expand=(1.15, 1.3),
                min_arrow_len=3,
                arrowprops=dict(arrowstyle="->", color="#555555", lw=0.5, mutation_scale=5),
            )
        for monochrome in (False, True):
            if monochrome:
                for ax, *_ in panels:
                    for text in ax.texts:
                        text.set_color("#111111")
                    for index, line in enumerate(ax.lines):
                        line.set_color("#111111")
                        line.set_alpha(1)
                        line.set_markerfacecolor("white" if index in (0, 1, 3) else "#111111")
            handles = [
                Line2D(
                    [],
                    [],
                    color="#111111" if monochrome else c,
                    linestyle=s,
                    marker=m,
                    markersize=4,
                    label=f"{CODES[model]} · {label}",
                    markerfacecolor="white"
                    if monochrome and i in (0, 1, 3)
                    else "#111111"
                    if monochrome
                    else c,
                )
                for i, (model, (label, c, s, m)) in enumerate(models.items())
            ]
            legend = fig.legend(
                handles=handles,
                loc="lower center",
                ncol=4 if len(models) == 7 else 3,
                frameon=False,
                fontsize=9 if len(models) == 7 else 10,
                handlelength=2.2 if len(models) == 7 else 3.2,
                bbox_to_anchor=(0.5, 0.005 if extra_row is not None else 0.015),
            )
            variant = "-bw" if monochrome else ""
            stem = output / f"{prefix}-overlap{coverage}{variant}-v1"
            for suffix in ("svg", "pdf", "png"):
                fig.savefig(stem.with_suffix("." + suffix), dpi=150)
            legend.remove()
            print(f"[figure] {stem}")
        plt.close(fig)
    print(f"[validated] {len(models)} models; {count} unique inputs; language N={dict(counts)}")


if __name__ == "__main__":
    main()
