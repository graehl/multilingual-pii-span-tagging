"""Task-level steps of the software supplement, run under one work directory.

Each step writes into `WORK/<step>` and logs to `WORK/logs/<step>` with a
receipt, so a reader can run the pipeline piecewise or all at once:

    data        fetch and convert the public corpora, assemble the base rows
    human-gold  rebuild the paper's 1,283-row public human-gold evaluation
    mixture     compile an O4-style training directory from public data
    evaluate    predict and score a checkpoint on the paper's human-gold view
    redact      tag or redact raw text with a checkpoint
    demo        all of the above at minutes scale on three small corpora
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "research/pii/frontier/evidence"
SOFTWARE = ROOT / "research/pii/frontier/software"

# Corpora O4's human-gold branch samples, and the base-branch sources whose
# natural realizations O4 also used (openner/aqmar natural come from the gold fetches).
GOLD_SOURCES = ("openner-commercial-core", "mapa", "aqmar-openner", "wojood-sample")
BASE_FETCH = ("hiner", "idner-news-2k")
BASE_ASSEMBLY = ("openner-commercial-core-natural", "aqmar-natural", "hiner-natural", "idner-news-natural")
DEMO_GOLD = ("aqmar-openner", "wojood-sample")
DEMO_FETCH = ("idner-news-2k",)
DEMO_ASSEMBLY = ("aqmar-natural", "idner-news-natural")
REFERENCE_TYPES = {"person_reference", "organization_reference"}
PAPER_O4_HUMAN = {"maximum_f1": 0.8883116883116883, "fixed_zero_bias_f1": 0.8816416269695859}


def driver():
    from scripts import pii_reproduction

    return pii_reproduction


def step(work: Path, name: str, command: list[str], settings: dict, environment=None) -> dict:
    # A rerun of a step keeps earlier logs: mixture, mixture-2, mixture-3, ...
    logs = work / "logs" / name
    serial = 1
    while logs.exists():
        serial += 1
        logs = work / "logs" / f"{name}-{serial}"
    return driver().run_logged(command, logs, settings=settings, environment_overrides=environment)


def python() -> str:
    return driver().workflow_python()


def require_licenses(sources) -> None:
    """Refuse to fetch a source whose license terms have not been approved."""
    import subprocess

    result = subprocess.run(
        [python(), "scripts/pii_software_sources.py", "check", *(f"--source={s}" for s in sources)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        message = result.stdout.strip() or result.stderr.strip()
        try:
            message = json.loads(message.splitlines()[-1])["error"]["message"]
        except (ValueError, KeyError, IndexError):
            pass
        raise ValueError(message)


def data_command(args) -> dict:
    work = args.work.resolve()
    gold, fetch, assembly = (
        (DEMO_GOLD, DEMO_FETCH, DEMO_ASSEMBLY)
        if args.demo
        else (
            GOLD_SOURCES,
            BASE_FETCH,
            BASE_ASSEMBLY,
        )
    )
    require_licenses([*gold, *fetch])
    done = []
    for source in (*gold, *fetch):
        upstream = work / "upstream" / source
        if not (work / "onboarded" / source / "manifest.json").is_file():
            for native, target in (("fetch", upstream), ("build", work / "onboarded")):
                step(
                    work,
                    f"{native}-{source}",
                    [
                        python(),
                        "scripts/pii_onboard_sources.py",
                        native,
                        source,
                        "--source-root",
                        str(upstream),
                        "--output-root",
                        str(work / "onboarded"),
                    ],
                    {"source": source, "phase": native, "target": str(target)},
                )
        done.append(source)
    assemble = step(
        work,
        "assemble-base",
        [
            python(),
            "scripts/pii_assemble_corpus.py",
            "build",
            "--out",
            str(work / "base"),
            "--min-node-count",
            "1",
            "--rule-completion-mode",
            "audit",
            "--seed",
            "20260927",
            *(value for source in assembly for value in ("--source", source)),
        ],
        {"sources": list(assembly)},
        {"PII_ONBOARDED_HOME": str(work / "onboarded"), "PII_EVAL_HOME": str(work / "logs/unused-eval-home")},
    )
    return {
        "ok": True,
        "onboarded": str(work / "onboarded"),
        "base": str(work / "base"),
        "sources": done,
        "base_sources": list(assembly),
        "assemble_receipt": assemble["receipt"],
    }


def human_gold_command(args) -> dict:
    work = args.work.resolve()
    command = [
        python(),
        "scripts/pii_public_gold.py",
        "eval-rebuild",
        "--onboarded",
        str(work / "onboarded"),
        "--out",
        str(work / "human-gold"),
    ]
    for source in DEMO_GOLD if args.demo else ():
        command.extend(("--source", source))
    receipt = step(work, "human-gold", command, {"subset": list(DEMO_GOLD) if args.demo else None})
    return {"ok": True, "human_gold": str(work / "human-gold"), "receipt": receipt["receipt"]}


def mixture_command(args) -> dict:
    work = args.work.resolve()
    out = (getattr(args, "out", None) or work / "mixture").resolve()
    command = [
        python(),
        "scripts/pii_public_mixture.py",
        "--onboarded",
        str(work / "onboarded"),
        "--base",
        str(work / "base"),
        "--evaluation",
        str(work / "human-gold/evaluation.jsonl"),
        "--gold-share",
        str(args.gold_share),
        "--out",
        str(out),
        "--screen",
        getattr(args, "screen", "overlap"),
    ]
    # Keep both of the paper's evaluations out of training: human gold and the Ont3 populations.
    for name in ("selection-inputs.jsonl", "heldout.jsonl"):
        command.extend(("--evaluation", str(ROOT / "data/ont3-evaluation" / name)))
    if getattr(args, "no_language_caps", False):
        command.append("--no-language-caps")
    for value in getattr(args, "language_cap", None) or ():
        command.extend(("--language-cap", value))
    for source in DEMO_GOLD if args.demo else ():
        command.extend(("--gold-corpus", source))
    if args.o4_membership:
        command.extend(("--o4-membership", str(SOFTWARE / "records/o4-training-membership.csv")))
    for path in getattr(args, "annotated", None) or ():
        command.extend(("--annotated", str(path.resolve())))
    receipt = step(
        work, "mixture", command, {"gold_share": args.gold_share, "o4_membership": args.o4_membership}
    )
    return {"ok": True, "mixture": str(out), "receipt": receipt["receipt"]}


def fetch_web_command(args) -> dict:
    """Recover the exact public web text of O4's teacher-annotated training rows."""
    work = args.work.resolve()
    require_licenses(["fineweb", "fineweb-2"])
    command = [
        python(),
        "scripts/pii_fetch_web.py",
        "--out",
        str(work / "web"),
        "--max-scan",
        str(args.max_scan),
    ]
    for language in args.language or ():
        command.extend(("--language", language))
    receipt = step(work, "fetch-web", command, {"languages": args.language, "max_scan": args.max_scan})
    summary = json.loads(Path(receipt["receipt"]).parent.joinpath("stdout.log").read_text().splitlines()[0])
    return {
        **summary,
        "rows": str(work / "web/rows.jsonl"),
        "web_receipt": str(work / "web/receipt.json"),
        "next": "annotate --input WORK/web/rows.jsonl --web-receipt WORK/web/receipt.json, "
        "then mixture --annotated OUT/training-rows.jsonl",
    }


NEEDLE_SET = "data/pii-needles/rare-type-needles-v1.json"
CUE_LEXICONS = "data/pii-annotations/final35/reference-cue-lexicon-v1"
# The paper's training draws: share of each language's sentences per stratum.
DRAW_STRATA = (("identifier", 0.1), ("reference", 0.2), ("ordinary", 0.7))
HUMAN_GOLD_DOMAIN = "human-gold"


def weighted_specs(values, *, special: dict[str, Path] | None = None) -> list[str]:
    """Flatten repeatable, comma-separated PATH[:WEIGHT] options into absolute, existing paths."""
    from scripts.pii_domain_select import parse_weighted

    specs = []
    for spec in (spec.strip() for value in values or () for spec in value.split(",")):
        if not spec:
            continue
        path, weight = parse_weighted(spec)
        path = (special or {}).get(str(path), path).resolve()
        if not path.is_file():
            raise ValueError(f"no such file: {path}")
        specs.append(f"{path}:{weight:g}")
    return specs


def weighted_names(values) -> set[str]:
    from scripts.pii_domain_select import parse_weighted

    return {str(parse_weighted(spec.strip())[0]) for value in values or () for spec in value.split(",")}


def merged_needle_set(specs: list[str], target: Path) -> Path:
    """Concatenate needle sets; a file's weight multiplies each of its needles' weights."""
    from scripts.pii_domain_select import parse_weighted

    needles, sources = [], []
    for spec in specs:
        path, weight = parse_weighted(spec)
        config = json.loads(path.read_text(encoding="utf-8"))
        if config.get("schema") != "pii-needle-set/v1":
            raise ValueError(f"{path}: not a pii-needle-set/v1 needle set")
        prefix = f"{path.stem}:" if len(specs) > 1 else ""
        needles += [
            {**needle, "name": prefix + needle["name"], "weight": needle["weight"] * weight}
            for needle in config["needles"]
        ]
        sources.append({"path": str(path), "weight": weight, "needles": len(config["needles"])})
    target.write_text(
        json.dumps({"schema": "pii-needle-set/v1", "needles": needles, "merged_from": sources}, indent=1)
        + "\n",
        encoding="utf-8",
    )
    return target


def select_command(args) -> dict:
    """Draw unlabeled candidate text from FineWeb / FineWeb-2 for screening and annotation."""
    work = args.work.resolve()
    languages = [code for value in args.language for code in value.split(",") if code]
    require_licenses(sorted({"fineweb" if code == "en" else "fineweb-2" for code in languages}))
    runs = work / "select"
    serial = 1
    while (runs / f"{serial:03d}").exists():
        serial += 1
    out = (args.out or runs / f"{serial:03d}").resolve()
    if out.exists():
        raise ValueError(f"{out} already exists; selections are immutable")
    needles = weighted_specs(args.needles, special={NEEDLE_SET: ROOT / NEEDLE_SET})
    if (
        HUMAN_GOLD_DOMAIN in weighted_names(args.domain)
        and not (work / "human-gold/evaluation.jsonl").is_file()
    ):
        raise ValueError("the default domain is WORK/human-gold/evaluation.jsonl; run human-gold first")
    domains = weighted_specs(args.domain, special={HUMAN_GOLD_DOMAIN: work / "human-gold/evaluation.jsonl"})
    if missing := [str(path) for path in args.exclude or () if not path.is_file()]:
        raise ValueError(f"no such --exclude file: {', '.join(missing)}")
    # Never select text already evaluated on, fetched or selected before.
    excludes = [work / "human-gold/evaluation.jsonl", work / "web/rows.jsonl"]
    excludes += sorted(runs.glob("*/rows.jsonl"))
    excludes = [path for path in excludes if path.is_file()] + [path.resolve() for path in args.exclude or ()]
    exclude_args = [arg for path in excludes for arg in ("--exclude-jsonl", str(path))]
    state = runs / "state"
    state.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True)
    settings = {
        "languages": languages,
        "rows": args.rows,
        "scan": args.scan,
        "excludes": [str(p) for p in excludes],
    }
    outputs = []
    if not needles and not domains:
        quotas = [f"training/{stratum}={max(1, round(args.rows * share))}" for stratum, share in DRAW_STRATA]
        command = [python(), "scripts/pii_final35_native_draw.py", "--languages", ",".join(languages)]
        command += ["--output-dir", str(out / "draw"), "--receipt", str(out / "draw/receipt.json")]
        command += ["--cue-lexicon-dir", str(ROOT / CUE_LEXICONS), "--seed", str(args.seed)]
        command += ["--max-documents-per-language", str(args.scan or 20000), *exclude_args]
        command += [arg for quota in quotas for arg in ("--quota", quota)]
        step(work, "select-draw", command, {**settings, "method": "draw", "quotas": quotas})
        outputs += [("draw", out / "draw" / f"{code}.jsonl") for code in languages]
    if needles:
        merged = merged_needle_set(needles, out / "needles.json")
        command = [python(), "scripts/pii_needle_select.py", "--languages", ",".join(languages)]
        command += ["--needles", str(merged), "--output-dir", str(out / "needles")]
        command += ["--receipt", str(out / "needles-receipt.json"), "--scan", str(args.scan or 2000)]
        command += [
            "--cursor",
            str(state / "cursor.json"),
            "--used-ledger",
            str(state / "used-regions.jsonl"),
        ]
        command += ["--threshold", "1", "--sentences", "--acli-quiet", *exclude_args]
        step(work, "select-needles", command, {**settings, "method": "needles", "needles": needles})
        receipt = json.loads((out / "needles-receipt.json").read_text(encoding="utf-8"))
        outputs += [("needles", Path(summary["sentences"]["path"])) for summary in receipt["languages"]]
    if domains:
        command = [python(), "scripts/pii_domain_select.py", "--languages", ",".join(languages)]
        command += [arg for spec in domains for arg in ("--domain", spec)]
        command += ["--output-dir", str(out / "domain"), "--receipt", str(out / "domain-receipt.json")]
        command += [
            "--cursor",
            str(state / "cursor.json"),
            "--used-ledger",
            str(state / "used-regions.jsonl"),
        ]
        command += ["--scan", str(args.scan or 3000), "--rows", str(args.rows), "--device", args.device]
        command += ["--acli-quiet", *exclude_args]
        step(work, "select-domain", command, {**settings, "method": "domain", "domains": domains})
        outputs += [("domain", out / "domain" / f"{code}.jsonl") for code in languages]
    counts, seen = {}, set()
    with (out / "rows.jsonl").open("x", encoding="utf-8") as sink:
        for method, path in outputs:
            for line in path.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                # Sentences of a needle paragraph that were already used elsewhere stay out.
                if row.get("previously_used") or row["id"] in seen:
                    continue
                seen.add(row["id"])
                sink.write(
                    json.dumps({**row, "selection": method}, ensure_ascii=False, sort_keys=True) + "\n"
                )
                counts[method] = counts.get(method, 0) + 1
    return {
        "ok": True,
        "rows": str(out / "rows.jsonl"),
        "rows_by_method": counts,
        "excluded": [str(path) for path in excludes],
        "next": f"pii-reproduce.py screen --work {args.work} --input {out / 'rows.jsonl'}",
    }


def screen_command(args) -> dict:
    """Screen your own candidate text against evaluation and training data before annotation."""
    work = args.work.resolve()
    out = (args.out or work / "screen" / args.input.stem).resolve()
    compares = list(args.compare or ())
    # The rebuilt human-gold evaluation is always something to keep out of training.
    evaluation = work / "human-gold/evaluation.jsonl"
    if not any("evaluation" in spec.partition("=")[0].split(",") for spec in compares):
        if not evaluation.is_file():
            raise ValueError(
                "no evaluation data to screen against: run human-gold first or pass "
                "--compare evaluation=NAME=PATH"
            )
        compares.append(f"evaluation=human-gold={evaluation}")
    command = [python(), "scripts/pii_software_screen.py", "--input", str(args.input.resolve())]
    command += ["--out", str(out), "--device", args.device, "--json"]
    for spec in compares:
        command.extend(("--compare", spec))
    receipt = step(work, "screen", command, {"input": str(args.input), "compare": compares})
    summary = json.loads(Path(receipt["receipt"]).parent.joinpath("stdout.log").read_text().splitlines()[-1])
    return {
        **summary,
        "dedup_receipt": str(out / "receipt.json"),
        "next": f"pii-reproduce.py annotate --input {out / 'retained.jsonl'} "
        f"--dedup-receipt {out / 'receipt.json'}"
        if summary.get("retained")
        else "No candidate row is new; nothing to annotate.",
    }


def refiner_partition(annotated: list[Path], out: Path, validation_share: float) -> dict:
    """Split complete Ont3 rows into refiner train/validation files by source group."""
    import hashlib

    rows = []
    for path in annotated:
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row.get("supervision") != "complete" or row.get("label_space") != "v2":
                raise ValueError(f"{path}: the refiner needs complete Ont3 rows (annotate or teacher output)")
            # The refiner moves primary-span endpoints only; references are never refined.
            row["spans"] = [span for span in row["spans"] if span[2] not in REFERENCE_TYPES]
            rows.append(row)
    if not rows:
        raise ValueError("no annotated rows")

    def group(row: dict) -> str:
        source = row.get("source") or {}
        return str(source.get("id") or row.get("source_group_id") or row["id"])

    def held_out(row: dict) -> bool:
        digest = hashlib.sha256(group(row).encode()).digest()
        return int.from_bytes(digest[:8], "big") / 2**64 < validation_share

    train = [row for row in rows if not held_out(row)]
    validation = [row for row in rows if held_out(row)]
    train_texts = {row["text"] for row in train}
    validation = [row for row in validation if row["text"] not in train_texts]
    out.mkdir(parents=True, exist_ok=False)
    files = {
        "native-v2-complete": (out / "native-v2-complete.jsonl", train),
        "view-native-v2-train": (out / "view-native-v2-train.jsonl", []),
        "validation": (out / "validation.jsonl", validation),
    }
    for path, items in files.values():
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in items), encoding="utf-8"
        )
    receipt = {
        "schema": "pii-v2-complete-calibration-corpus",
        "purpose": "reader boundary-refiner fit from complete teacher-annotated rows",
        "train_validation_text_overlap": len(train_texts & {row["text"] for row in validation}),
        "validation": {
            "rows": len(validation),
            "manifest_sha256": hashlib.sha256(
                "\n".join(sorted(r["id"] for r in validation)).encode()
            ).hexdigest(),
            "source_group_overlap": len({group(r) for r in train} & {group(r) for r in validation}),
        },
        "train_rows": len(train),
        "outputs": {
            name: {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for name, (path, _) in files.items()
        },
    }
    (out / "build.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def train_refiner_command(args) -> dict:
    """Fit a character-boundary refiner for your tagger from complete Ont3 rows."""
    work = args.work.resolve()
    data = work / "refiner-data"
    partition = refiner_partition([path.resolve() for path in args.annotated], data, args.validation_share)
    out = (args.out or work / "refiner").resolve()
    command = [
        python(),
        "scripts/pii_character_boundary_refiner.py",
        "fit",
        "--train",
        str(data / "native-v2-complete.jsonl"),
        "--train",
        str(data / "view-native-v2-train.jsonl"),
        "--validation",
        str(data / "validation.jsonl"),
        "--partition-receipt",
        str(data / "build.json"),
        "--model",
        str(args.checkpoint.resolve()),
        "--output",
        str(out),
        "--epochs",
        str(args.epochs),
    ]
    receipt = step(work, "train-refiner", command, {"partition": partition, "epochs": args.epochs})
    return {"ok": True, "refiner": str(out), "partition": partition, "receipt": receipt["receipt"]}


def names_command(args) -> dict:
    """Build the name-kind model from approved public lexicons into WORK/name-kind."""
    work = args.work.resolve()
    command = [python(), "scripts/pii_software_names.py", "--out", str(work / "name-kind")]
    command += ["--threads", str(args.threads)]
    receipt = step(work, "names", command, {"threads": args.threads})
    return {
        "ok": True,
        "bundle": str(work / "name-kind/bundle/name-kind.config.json"),
        "receipt": receipt["receipt"],
        "next": "redact --work WORK --serve now applies name-kind postprocessing",
    }


def licenses_command(args) -> dict:
    """List every data source's recorded license terms; optionally record approvals."""
    import subprocess

    command = [python(), "scripts/pii_software_sources.py", "licenses"]
    for source in args.approve or ():
        command.append(f"--approve={source}")
    if args.approve_all:
        command.append("--approve-all")
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    if result.returncode:
        raise ValueError(result.stdout.strip() or result.stderr.strip())
    return json.loads(result.stdout.splitlines()[0])


ONT3_DATA = ROOT / "data/ont3-evaluation"


def predicted_sweep(out: Path, name: str, inputs: Path, checkpoint: Path, raw: bool, work: Path):
    """The paper's O-bias prediction sweep on one population, served unless raw."""
    sweep = out / f"sweep-{name}.json"
    step(
        out,
        f"predict-{name}",
        [
            "nice",
            "-n",
            "10",
            python(),
            "research/pii/frontier/evidence/priority9-shared-v1/predict-sweeps.py",
            "ont3",
            str(inputs),
            str(sweep),
            "--model-path",
            str(checkpoint.resolve()),
            "--context-side",
            "none",
        ],
        {"checkpoint": str(checkpoint), "view": f"paper {name} view, no context, O-bias sweep"},
    )
    if raw:
        return sweep, []
    return served_sweep(work, out, inputs, sweep, name)


def evaluate_command(args) -> dict:
    """The paper's human-gold and Ont3 views: its bias sweep, prediction code, serving and scorer."""
    work = args.work.resolve()
    out = (args.out or work / "evaluation").resolve()
    raw = getattr(args, "raw", False)
    populations = getattr(args, "population", "all")
    out.mkdir(parents=True, exist_ok=False)
    result = {"ok": True, "serving": None, "control": "o4-unrefined" if raw else "o4"}
    if populations in ("all", "human"):
        result["human"] = evaluate_human(args, work, out, raw, result)
    if populations in ("all", "ont3"):
        result["ont3"] = evaluate_ont3(args, work, out, raw, result)
    result["serving"] = result["serving"] or "raw model output (--raw)"
    (out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def evaluate_ont3(args, work: Path, out: Path, raw: bool, result: dict) -> dict:
    """Both Ont3 collections, scored and paired against O4's receipts like the paper's numbers."""
    sweeps = {}
    for name, filename in (("selection", "selection-inputs.jsonl"), ("heldout", "heldout.jsonl")):
        sweeps[name], serving = predicted_sweep(out, name, ONT3_DATA / filename, args.checkpoint, raw, work)
        result["serving"] = result["serving"] or serving
    command = [python(), "scripts/pii_software_ont3.py", "--selection-sweep", str(sweeps["selection"])]
    command += ["--heldout-sweep", str(sweeps["heldout"]), "--model", args.name]
    command += ["--control", result["control"], "--out", str(out / "scores-ont3.json.gz")]
    scored = step(out, "score-ont3", command, {"control": result["control"]})
    summary = json.loads(Path(scored["receipt"]).parent.joinpath("stdout.log").read_text().splitlines()[0])
    return {
        "scores": str(out / "scores-ont3.json.gz"),
        "views": summary["summary"],
        "note": "Zero bias; F1 in percent: redaction regions at 80% and exact overlap, exact typed spans; "
        "selection = the 659 rows O4 was selected on, heldout = 542 never used for selection, pooled = both.",
    }


def evaluate_human(args, work: Path, out: Path, raw: bool, result: dict) -> dict:
    """The paper's public human-gold view, paired against O4's receipt."""
    human = (args.human_gold or work / "human-gold").resolve()
    sweep, serving = predicted_sweep(out, "human", human / "inputs.jsonl", args.checkpoint, raw, work)
    result["serving"] = result["serving"] or serving
    receipt_name = "o4-boundary.json.gz" if raw else "o4-comparison.json.gz"
    command = [
        python(),
        "scripts/pii_public_gold.py",
        "score",
        "--evaluation",
        str(human / "evaluation.jsonl"),
        "--sweep",
        str(sweep),
        "--model",
        args.name,
        "--out",
        str(out / "scores-human.json.gz"),
    ]
    sidecar = SOFTWARE / "records/title-extents-human-gold.jsonl"
    if sidecar.is_file():
        command.extend(("--title-sidecar", str(sidecar)))
    command.extend(("--compare-receipt", str(SOFTWARE / "records/receipts" / receipt_name)))
    command.extend(("--compare-system", result["control"]))
    scored = step(out, "score-human", command, {"title_sidecar": sidecar.is_file()})
    summary = json.loads(Path(scored["receipt"]).parent.joinpath("stdout.log").read_text().splitlines()[0])
    receipt = json.loads((human / "receipt.json").read_text())
    comparable = receipt["rows"] == 1283 and sidecar.is_file()
    return {
        "scores": str(out / "scores-human.json.gz"),
        "rows": summary["rows"],
        "maximum": summary["maximum"],
        "fixed_zero_bias": summary["fixed_zero_bias"],
        "paired_versus_o4": summary["paired_versus_o4"],
        "paper_o4": PAPER_O4_HUMAN,
        "comparable_to_paper": comparable,
        "note": "Redaction regions at 80% overlap (maximum, fixed); exact regions in the paired "
        "comparison; paper title and coverage policy. O4's paper numbers are served output."
        + ("" if comparable else " Not paper-comparable: subset rows or missing title sidecar."),
    }


def name_kind_bundle(work: Path) -> Path | None:
    """The name-kind model the names command built, if any."""
    bundle = work / "name-kind/bundle/name-kind.config.json"
    return bundle.resolve() if bundle.is_file() else None


def served_sweep(work: Path, out: Path, inputs: Path, sweep: Path, name: str) -> tuple[Path, list[str]]:
    """Apply the paper's serving stages to a prediction sweep, in serving order.

    Boundary refinement with the shipped refiner, then name-kind postprocessing
    when WORK/name-kind holds a bundle, then regex supplementation.
    """
    refined = out / f"sweep-{name}-refined.json"
    step(
        out,
        f"refine-{name}",
        [
            python(),
            "scripts/pii_character_boundary_refiner.py",
            "apply-sweep",
            "--exclude-reference-spans",
            "--refiner",
            str(SOFTWARE / "models/boundary-refiner"),
            "--gold",
            str(inputs),
            "--predictions",
            str(sweep),
            "--output",
            str(refined),
            "--receipt",
            str(out / f"sweep-{name}-refined.receipt.json"),
        ],
        {"stage": "boundary refinement (shipped refiner)"},
    )
    bundle = name_kind_bundle(work)
    served = out / f"sweep-{name}-served.json"
    chain = [python(), "scripts/pii_serving_chain.py", "--sweep", str(refined), "--inputs", str(inputs)]
    chain += ["--output", str(served)]
    chain += ["--name-kind-bundle", str(bundle)] if bundle else ["--no-name-kind"]
    step(
        out,
        f"serve-{name}",
        chain,
        {"stage": "name-kind and regex", "name_kind_bundle": str(bundle) if bundle else None},
    )
    return served, ["boundary_refinement", *(["name_kind"] if bundle else []), "regex"]


def redact_command(args) -> dict:
    command = [
        python(),
        "scripts/pii_redact.py",
        "--checkpoint",
        str(args.checkpoint.resolve()),
        "--input",
        str(args.input.resolve()),
        "--out",
        str(args.out.resolve()),
        "--lang",
        args.lang,
    ]
    types = args.types
    receipt = args.work / "mixture/receipt.json" if getattr(args, "work", None) else None
    if types is None and receipt is not None and receipt.is_file():
        # Emit only the types the training data supervised positively.
        types = ",".join(json.loads(receipt.read_text())["supported_types"])
    if types:
        command.extend(("--types", types))
    served = []
    if not getattr(args, "raw", False):
        # The paper's serving order: boundary refinement, name-kind, regex.
        command.extend(("--refiner", str(SOFTWARE / "models/boundary-refiner"), "--regex"))
        served = ["boundary_refinement", "regex"]
        bundle = getattr(args, "work", None) and name_kind_bundle(args.work)
        if bundle:
            command.extend(("--name-kind-bundle", str(bundle)))
            served.insert(1, "name_kind")
    work = args.out.resolve().parent
    receipt = step(
        work, f"redact-{args.out.stem}", command, {"checkpoint": str(args.checkpoint), "serving": served}
    )
    return {"ok": True, "out": str(args.out.resolve()), "serving": served, "receipt": receipt["receipt"]}


def verify_command(args) -> dict:
    """Recompute the paper's reported numbers from the shipped text-free receipts."""
    import subprocess

    command = [
        python(),
        "scripts/pii_software_receipts.py",
        "verify",
        *(["--details"] if args.details else []),
    ]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"verification failed: {result.stdout.strip() or result.stderr.strip()[-500:]}")
    return json.loads(result.stdout.splitlines()[0])


DEMO_TEXT = (
    "Contact Maria Keller at maria.keller@example.org or +49 30 1234567 before 12 March 2025.\n"
    "Ahmed Hassan joined Siemens in Munich last year.\n"
)


def demo_command(args) -> dict:
    """Minutes-scale end-to-end check that the code works; no quality claim."""
    import argparse

    work = args.work.resolve()
    work.mkdir(parents=True, exist_ok=False)
    base = {"work": work, "demo": True}
    results = {"data": data_command(argparse.Namespace(**base))}
    results["human_gold"] = human_gold_command(argparse.Namespace(**base))
    results["mixture"] = mixture_command(argparse.Namespace(**base, gold_share=0.5, o4_membership=False))
    train = driver().train_command(
        argparse.Namespace(
            data=work / "mixture",
            out=work / "model",
            model=args.model,
            recipe="o4-fresh",
            init_from_checkpoint=None,
            eval_steps=args.steps,
            steps=args.steps,
            step_scale=1.0,
            max_steps=None,
            batch=4,
            grad_accum=2,
            seed=20260927,
            extra=[],
        )
    )
    results["train"] = train
    checkpoint = work / "model/fit/model" / f"checkpoint-{args.steps}"
    results["evaluate"] = evaluate_command(
        argparse.Namespace(
            work=work, out=None, human_gold=None, checkpoint=checkpoint, name="demo", population="human"
        )
    )
    sample = work / "demo-input.txt"
    sample.write_text(DEMO_TEXT, encoding="utf-8")
    results["redact"] = redact_command(
        argparse.Namespace(
            checkpoint=checkpoint,
            input=sample,
            out=work / "demo-redacted.jsonl",
            lang="en",
            types=None,
            work=work,
            serve=True,
        )
    )
    results["note"] = (
        "Demo scale: three small corpora and a few hundred updates prove the pipeline runs; "
        "scores are not quality results and the evaluation is a declared subset."
    )
    (work / "demo-summary.json").write_text(json.dumps(results, indent=2, default=str) + "\n")
    return results


if __name__ == "__main__":
    sys.exit("run through pii-reproduce.py")
