"""Reproduction commands and the software supplement's Markdown manual."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import acli

ROOT = Path(__file__).resolve().parents[1]
SOFTWARE = ROOT / "research/pii/frontier/software"


class MarkdownHelpParser(acli.args.ArgumentParser):
    """Render the command reference as Markdown for terminal and README use."""

    def format_help(self) -> str:
        if self.prog != "pii-reproduce.py":
            return super().format_help()
        guide = (SOFTWARE / "guide.md").read_text(encoding="utf-8").rstrip()
        reference = super().format_help().rsplit("\nacli:", 1)[0].rstrip()
        return f"{guide}\n\n## Command reference\n\n```text\n{reference}\n```\n\nacli: 1 complete\n"


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def positive_scale(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be finite and greater than zero")
    return number


def build_parser() -> MarkdownHelpParser:
    parser = MarkdownHelpParser(prog="pii-reproduce.py", description="Portable PII reproduction workflow.")
    acli.add_standard_args(parser)
    commands = parser.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser("doctor", help="Report local prerequisites without loading models.")
    doctor.set_defaults(action=doctor_command)
    readme = commands.add_parser("readme", help="Write this help as the staged package README.md.")
    readme.add_argument("--out", type=Path, required=True)
    readme.set_defaults(action=readme_command)
    files = commands.add_parser("files", help="List the local Python dependency subset (no data or builds).")
    files.set_defaults(action=files_command)
    stage = commands.add_parser(
        "stage", help="Stage repository-relative sources and README; optionally anonymize for review."
    )
    stage.add_argument("--out", type=Path, required=True)
    stage.add_argument(
        "--anonymous",
        action="store_true",
        help="Redact identities and scan staged files; default preserves attribution.",
    )
    redaction_arguments(stage)
    stage.set_defaults(action=stage_command)
    package = commands.add_parser(
        "package", help="Stage into OUT/software, build OUT/software.tgz and verify a fresh extraction."
    )
    package.add_argument("--out", type=Path, required=True, help="New directory for the stage and archive")
    package.add_argument("--anonymous", action="store_true", help="Anonymous review export (see stage)")
    redaction_arguments(package)
    package.add_argument(
        "--from-repo", type=Path, help="Package a published repository's committed files instead of this tree"
    )
    package.set_defaults(action=package_command)
    acli.add_standard_args(package)
    publish = commands.add_parser(
        "publish",
        help="Stage the public release into a git repository for review and commit (no commit made).",
    )
    publish.add_argument("--repo", type=Path, required=True, help="The release repository's checkout")
    publish.add_argument(
        "--uncommitted",
        action="store_true",
        help="Test publish: allow uncommitted shipped draft files, to run verify in the repository before "
        "deciding to commit. A release still needs them committed and published again.",
    )
    redaction_arguments(publish)
    publish.set_defaults(action=publish_command)
    acli.add_standard_args(publish)
    install = commands.add_parser("install", help="Create the pinned Pixi environment (Linux, CUDA 13).")
    install.add_argument("--log-dir", type=Path, default=Path("work/install"))
    install.set_defaults(action=install_command)
    for name, native_command in (("fetch", "fetch"), ("prepare", "build")):
        command = commands.add_parser(
            name,
            help="Fetch a pinned source."
            if name == "fetch"
            else "Convert a source with its existing ontology map.",
        )
        command.add_argument("--source", required=True)
        command.add_argument("--source-root", type=Path, required=True)
        command.add_argument("--output-root", type=Path, default=Path("work/data"))
        command.add_argument("--log-dir", type=Path, required=True)
        if name == "prepare":
            command.add_argument("--max-records-per-source", type=positive_int)
        command.set_defaults(action=source_command, native_command=native_command)
        acli.add_standard_args(command)
    annotate = commands.add_parser(
        "annotate",
        help="Annotate with a Gemma-4 31B teacher (default; in-process or --server) or a --codex-model such "
        "as Luna, O4's teacher.",
    )
    annotate.add_argument("--input", type=Path, required=True)
    annotate.add_argument(
        "--out",
        type=Path,
        required=True,
        help="New directory for predictions, raw output, prompt contract and logs.",
    )
    annotate.add_argument(
        "--model",
        help=f"Hugging Face teacher run in-process or behind --server (default {DEFAULT_HF_TEACHER})",
    )
    codex = annotate.add_mutually_exclusive_group()
    codex.add_argument(
        "--codex-model",
        help="Annotate with this OpenAI model through Codex instead, e.g. gpt-6-luna; needs a "
        "`codex login`. Luna labeled O4's training rows: use it for annotation quality like the paper's",
    )
    codex.add_argument(
        "--luna",
        dest="codex_model",
        action="store_const",
        const="gpt-6-luna",
        help="--codex-model gpt-6-luna",
    )
    annotate.add_argument(
        "--codex-home",
        type=Path,
        help="Isolated authenticated Codex home for --codex-model, used as is. Default: the driver "
        "builds ~/.codex-pii-annotate from scripts/codex-annotation-home.config.toml and your "
        "login's auth.json ($CODEX_HOME or ~/.codex), refreshed every run",
    )
    annotate.add_argument(
        "--server",
        help="OpenAI-compatible server base URL serving --model, e.g. http://127.0.0.1:8000/v1 from "
        "`pixi run serve-teacher` in research/pii/frontier/software/serve (pinned vLLM); requests "
        "run concurrently instead of one at a time",
    )
    annotate.add_argument(
        "--concurrency",
        type=positive_int,
        help="Requests in flight: default 64 with --server, 6 with --codex-model (as the paper's Luna "
        "batches ran)",
    )
    annotate.add_argument(
        "--prompt-revision",
        choices=sorted(PROMPT_REVISIONS),
        default="paper-prompted",
        help="paper-prompted (default): the prompt behind the paper's prompted-LLM result "
        "(prompts/pii-label/paper-eval/); paper-teacher: the prompt O4's teacher annotated its "
        "training rows with",
    )
    annotate.add_argument("--limit", type=positive_int, help="Annotate only the first N admitted records.")
    annotate.add_argument("--lang", default="en")
    admission = annotate.add_mutually_exclusive_group(required=True)
    admission.add_argument("--dedup-receipt", type=Path)
    admission.add_argument("--reannotation-receipt", type=Path)
    admission.add_argument("--evaluation-replay-admission", type=Path)
    admission.add_argument(
        "--web-receipt",
        type=Path,
        help="fetch-web receipt: input rows are O4's own hash-verified web training text",
    )
    admission.add_argument(
        "--verification-replay",
        action="store_true",
        help="re-annotate rows a published result already annotated (for example the authors' own "
        "evaluation rows) to check that result: skips the screening/admission receipts, records the "
        "bypass, and writes no training rows",
    )
    annotate.set_defaults(action=annotate_command)
    acli.add_standard_args(annotate)
    codex_home = commands.add_parser(
        "codex-home",
        help="Build or refresh the isolated Codex home annotate --codex-model uses (it runs this itself).",
    )
    codex_home.add_argument("--home", type=Path, help="Home to build (default ~/.codex-pii-annotate)")
    codex_home.add_argument(
        "--login",
        type=Path,
        help="Codex home holding your login's auth.json (default $CODEX_HOME or ~/.codex)",
    )
    codex_home.set_defaults(action=codex_home_command)
    acli.add_standard_args(codex_home)
    train = commands.add_parser(
        "train", help="Fit the existing affine BIOES encoder on a prepared training directory."
    )
    train.add_argument(
        "--data", type=Path, required=True, help="Directory with train.jsonl, val.jsonl and labels.json."
    )
    train.add_argument("--out", type=Path, required=True)
    train.add_argument("--model", default="FacebookAI/xlm-roberta-large")
    train.add_argument(
        "--recipe",
        choices=("affine", "o4", "o4-fresh"),
        default="affine",
        help="affine: minimal BIOES head; o4: O4's recorded trainer options (continuation, needs a parent); "
        "o4-fresh: the same options at the lineage root's learning rates for a fit from the base encoder",
    )
    train.add_argument("--init-from-checkpoint", type=Path)
    train.add_argument(
        "--eval-steps", type=positive_int, default=2000, help="Validation interval (capped at steps)"
    )
    train.add_argument(
        "--steps",
        type=positive_int,
        help="Unscaled update budget; default 12000 for o4-fresh (measured to reach O4 on human gold), "
        "else 4000 (O4's continuation budget)",
    )
    train.add_argument("--step-scale", type=positive_scale, default=1.0)
    train.add_argument("--max-steps", type=positive_int)
    train.add_argument("--batch", type=positive_int, default=8)
    train.add_argument("--grad-accum", type=positive_int, default=8)
    train.add_argument("--seed", type=int, default=20260927)
    train.add_argument(
        "extra",
        nargs=argparse.REMAINDER,
        help="Additional native trainer options after --; cannot override driver-owned options.",
    )
    train.set_defaults(action=train_command)
    acli.add_standard_args(train)
    generic = commands.add_parser(
        "score-jsonl",
        help="Predict and score any gold JSONL with the generic evaluator (not the paper's view).",
    )
    generic.add_argument(
        "--input", type=Path, required=True, help="Gold JSONL with id, text, spans [{start,end,type}]."
    )
    generic.add_argument("--checkpoint", type=Path, required=True)
    generic.add_argument("--out", type=Path, required=True)
    generic.set_defaults(action=evaluate_command)
    acli.add_standard_args(generic)
    add_workflow_commands(commands)
    fetch_model = commands.add_parser(
        "fetch-model", help="Prefetch model weights into the Hugging Face cache (optional)."
    )
    fetch_model.add_argument("--model", required=True)
    fetch_model.add_argument("--revision", required=True, help="Pinned upstream commit or revision.")
    fetch_model.add_argument("--log-dir", type=Path, required=True)
    fetch_model.set_defaults(action=fetch_model_command)
    acli.add_standard_args(fetch_model)
    assemble = commands.add_parser(
        "assemble", help="Assemble selected prepared sources into train/validation JSONL and labels."
    )
    assemble.add_argument(
        "--source",
        action="append",
        required=True,
        help="Assembler source name, e.g. idner-news-natural; repeat for a mixture.",
    )
    assemble.add_argument("--prepared-root", type=Path, required=True)
    assemble.add_argument("--out", type=Path, required=True)
    assemble.add_argument("--seed", type=int, default=20260927)
    assemble.set_defaults(action=assemble_command)
    acli.add_standard_args(assemble)
    for command in (doctor, readme, files, stage, install):
        acli.add_standard_args(command)
    return parser


def add_workflow_commands(commands) -> None:
    """Task-level pipeline steps; each writes under one --work directory."""
    from scripts import pii_software_workflow as workflow

    def work_command(name: str, action, help_text: str, *, demo: bool = True):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--work", type=Path, default=Path("work"), help="Pipeline work directory")
        if demo:
            command.add_argument(
                "--demo", action="store_true", help="Use the three small demo corpora instead of the full set"
            )
        command.set_defaults(action=action)
        acli.add_standard_args(command)
        return command

    licenses = commands.add_parser(
        "licenses",
        help="List every data source's license terms and record your approvals; fetches require approval.",
    )
    licenses.add_argument(
        "--approve", action="append", help="Approve this source's recorded terms; repeatable"
    )
    licenses.add_argument(
        "--approve-all", action="store_true", help="Approve all listed sources except commercial ones"
    )
    licenses.set_defaults(action=workflow.licenses_command)
    acli.add_standard_args(licenses)
    work_command(
        "data",
        workflow.data_command,
        "Fetch and convert the public corpora into WORK/onboarded and assemble base rows in WORK/base.",
    )
    work_command(
        "human-gold",
        workflow.human_gold_command,
        "Rebuild the paper's 1,283-row public human-gold evaluation in WORK/human-gold.",
    )
    mixture = work_command(
        "mixture", workflow.mixture_command, "Compile an O4-style training directory in WORK/mixture."
    )
    mixture.add_argument("--gold-share", type=float, default=0.5, help="Human-gold sampling share (O4: 0.5)")
    mixture.add_argument(
        "--o4-membership",
        action="store_true",
        help="Keep exactly O4's overlap-screened human-gold rows (from the shipped membership record)",
    )
    mixture.add_argument("--out", type=Path, help="New training directory; default WORK/mixture")
    mixture.add_argument(
        "--screen",
        choices=("overlap", "exact"),
        default="overlap",
        help="overlap (default): drop training rows the paper's overlap detector matches to a human-gold "
        "or Ont3 evaluation row; exact: drop only exact text matches",
    )
    mixture.add_argument(
        "--language-cap",
        action="append",
        metavar="LANG=SHARE",
        help="Override one language's maximum share of training draws (defaults in "
        "research/pii/frontier/software/language-caps.yaml: 0.20 for any language, 0.08 for Hindi); repeatable",
    )
    mixture.add_argument(
        "--no-language-caps", action="store_true", help="Sample languages by row weight alone"
    )
    mixture.add_argument(
        "--annotated",
        type=Path,
        action="append",
        help="Your teacher-annotated JSONL in Ont3 labels (annotate output) to add to the base branch",
    )
    fetch_web = work_command(
        "fetch-web",
        workflow.fetch_web_command,
        "Recover the exact public web text of O4's teacher-annotated rows into WORK/web for your own annotation.",
        demo=False,
    )
    fetch_web.add_argument("--language", action="append", help="Only this language code; repeatable")
    fetch_web.add_argument(
        "--max-scan", type=positive_int, default=500000, help="Records to stream per config"
    )
    select = work_command(
        "select",
        workflow.select_command,
        "Draw new unlabeled candidate text from FineWeb / FineWeb-2 into WORK/select/NNN for screen.",
        demo=False,
    )
    select.add_argument(
        "--language", action="append", required=True, help="Language code(s); repeatable or comma-separated"
    )
    select.add_argument(
        "--rows",
        type=positive_int,
        default=200,
        help="Candidates per language (sentences, or paragraphs with --domain)",
    )
    select.add_argument(
        "--needles",
        action="append",
        nargs="?",
        const=workflow.NEEDLE_SET,
        metavar="FILE[:WEIGHT]",
        help="Instead select paragraphs matching rare-identifier surface patterns (default set: the paper's "
        "rare-type needles); repeatable or comma-separated, weights scale each file's needles. Untested for "
        "efficacy; worth trying when adapting to a domain",
    )
    select.add_argument(
        "--domain",
        action="append",
        nargs="?",
        const=workflow.HUMAN_GOLD_DOMAIN,
        metavar="FILE[:WEIGHT]",
        help="Instead select paragraphs most similar to example text (JSONL with text, or one example per "
        "line; default: the rebuilt human-gold evaluation text); repeatable or comma-separated, a weight "
        "counts a file's examples that many times",
    )
    select.add_argument(
        "--scan",
        type=positive_int,
        help="Documents read per language this run (default 20000; 2000 with --needles, 3000 with --domain); "
        "later --needles/--domain runs continue past documents already read",
    )
    select.add_argument(
        "--exclude", type=Path, action="append", help="JSONL rows never to select; repeatable"
    )
    select.add_argument("--seed", type=int, default=20260927, help="Draw seed (default selection)")
    select.add_argument("--out", type=Path, help="New directory; default WORK/select/NNN")
    select.add_argument(
        "--device", choices=("cuda", "cpu"), default="cuda", help="Embedding device for --domain"
    )
    screen = work_command(
        "screen",
        workflow.screen_command,
        "Screen your own text against evaluation and training data; writes the receipt annotate requires.",
        demo=False,
    )
    screen.add_argument("--input", type=Path, required=True, help="Candidate JSONL rows {id, text, lang}")
    screen.add_argument(
        "--compare",
        action="append",
        metavar="ROLE[,ROLE...]=NAME=PATH",
        help="Data to keep your rows apart from (repeatable); roles: training, development, validation, "
        "evaluation, prior_draws, annotation_attempts, reserved. WORK/human-gold is always the "
        "evaluation source unless you name one",
    )
    screen.add_argument("--out", type=Path, help="New directory; default WORK/screen/<input stem>")
    screen.add_argument("--device", choices=("cuda", "cpu"), default="cuda", help="Embedding device")
    names = work_command(
        "names",
        workflow.names_command,
        "Fetch approved public name lexicons and build the name-kind model into WORK/name-kind (CPU).",
        demo=False,
    )
    names.add_argument("--threads", type=positive_int, default=4, help="CPU training threads")
    refiner = work_command(
        "train-refiner",
        workflow.train_refiner_command,
        "Fit a character-boundary refiner for your tagger from complete Ont3 rows (e.g. annotate output).",
        demo=False,
    )
    refiner.add_argument(
        "--checkpoint", type=Path, required=True, help="Your tagger; proposes spans to refine"
    )
    refiner.add_argument("--annotated", type=Path, action="append", required=True)
    refiner.add_argument("--out", type=Path, help="New refiner directory; default WORK/refiner")
    refiner.add_argument("--epochs", type=positive_int, default=8)
    refiner.add_argument("--validation-share", type=float, default=0.1)
    evaluate = work_command(
        "evaluate",
        workflow.evaluate_command,
        "Score a checkpoint on the paper's human-gold and Ont3 views and compare with O4.",
        demo=False,
    )
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--human-gold", type=Path, help="Default: WORK/human-gold")
    evaluate.add_argument("--out", type=Path, help="New directory; default WORK/evaluation")
    evaluate.add_argument("--name", default="model", help="System name in the score file")
    evaluate.add_argument(
        "--raw",
        action="store_true",
        help="Score the tagger's own output, paired against O4 without boundary refinement; by default "
        "the paper's serving stages (boundary refiner, name-kind when built, regex) are applied first "
        "and the pairing is against served O4, as for O4's reported numbers",
    )
    evaluate.add_argument(
        "--population",
        choices=("all", "human", "ont3"),
        default="all",
        help="human: the rebuilt public human gold; ont3: the paper's shipped 31-type evaluation "
        "(659 selection + 542 held-out rows); all: both",
    )
    redact = commands.add_parser("redact", help="Tag and redact raw text (one document per line, or JSONL).")
    redact.add_argument("--checkpoint", type=Path, required=True)
    redact.add_argument("--input", type=Path, required=True)
    redact.add_argument("--out", type=Path, required=True, help="New JSONL with spans and redacted text")
    redact.add_argument("--lang", default="en")
    redact.add_argument(
        "--types",
        help="Comma-separated Ont3 types to emit; default: the types WORK/mixture supervised, else all",
    )
    redact.add_argument("--work", type=Path, help="Pipeline work directory whose mixture trained the model")
    serving = redact.add_mutually_exclusive_group()
    serving.add_argument(
        "--raw",
        action="store_true",
        help="Emit the tagger's own spans without the paper's serving stages (shipped boundary refiner, "
        "name-kind postprocessing when WORK/name-kind holds a bundle built by names, regex supplementation), "
        "which are applied by default",
    )
    serving.add_argument("--serve", action="store_true", help="Apply the serving stages (the default)")
    redact.set_defaults(action=workflow.redact_command)
    acli.add_standard_args(redact)
    verify = commands.add_parser(
        "verify", help="Recompute the paper's reported scores and intervals from the text-free receipts."
    )
    verify.add_argument("--details", action="store_true", help="List every recomputed number")
    verify.set_defaults(action=workflow.verify_command)
    acli.add_standard_args(verify)
    demo = commands.add_parser(
        "demo", help="Run the whole pipeline at minutes scale on three small corpora; proves the code works."
    )
    demo.add_argument("--work", type=Path, default=Path("work-demo"))
    demo.add_argument("--model", default="FacebookAI/xlm-roberta-large")
    demo.add_argument("--steps", type=positive_int, default=200)
    demo.set_defaults(action=workflow.demo_command)
    acli.add_standard_args(demo)


def assemble_command(args: argparse.Namespace) -> dict:
    output = args.out.resolve()
    command = [
        workflow_python(),
        "scripts/pii_assemble_corpus.py",
        "build",
        "--out",
        str(output / "corpus"),
        "--min-node-count",
        "1",
        "--rule-completion-mode",
        "audit",
        "--seed",
        str(args.seed),
    ]
    for source in args.source:
        command.extend(("--source", source))
    return run_logged(
        command,
        output,
        settings={"sources": args.source, "prepared_root": str(args.prepared_root.resolve())},
        environment_overrides={
            "PII_ONBOARDED_HOME": str(args.prepared_root.resolve()),
            "PII_EVAL_HOME": str(output / "unused-evaluation-home"),
        },
    )


def fetch_model_command(args: argparse.Namespace) -> dict:
    hf = Path(workflow_python()).with_name("hf")
    return run_logged(
        [str(hf), "download", args.model, "--revision", args.revision],
        args.log_dir,
        settings={"model": args.model, "revision": args.revision},
    )


def evaluate_command(args: argparse.Namespace) -> dict:
    import shutil

    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=False)
    gold = output / "gold"
    gold.mkdir()
    shutil.copyfile(args.input, gold / "evaluation.jsonl")
    environment = {"PII_EVAL_HOME": str(output), "PII_EVAL_LOCAL_MODEL": str(args.checkpoint.resolve())}
    predict = run_logged(
        [
            "nice",
            "-n",
            "10",
            workflow_python(),
            "scripts/pii_eval.py",
            "predict",
            "--model",
            "local",
            "--datasets",
            "evaluation",
            "--hf-windowing",
            "token-capacity",
        ],
        output / "predict-log",
        settings={"phase": "predict"},
        environment_overrides=environment,
    )
    score = run_logged(
        [
            workflow_python(),
            "scripts/pii_eval.py",
            "score",
            "--models",
            "local",
            "--datasets",
            "evaluation",
            "--output",
            str(output / "scores.json"),
        ],
        output / "score-log",
        settings={"phase": "score"},
        environment_overrides=environment,
    )
    return {
        "ok": True,
        "predict_receipt": predict["receipt"],
        "score_receipt": score["receipt"],
        "scores": str(output / "scores.json"),
        "view": "native evaluator default; paper-specific projections require the recorded comparison recipe",
    }


# Options the driver binds per invocation; a recipe's recorded values are dropped.
DRIVER_BOUND = {
    "--data": 1,
    "--out": 1,
    "--model": 1,
    "--init-from-checkpoint": 1,
    "--dual-head-map": 1,
    "--max-steps": 1,
    "--eval-steps": 1,
    "--batch": 1,
    "--grad-accum": 1,
    "--seed": 1,
    "--keep-final-checkpoint": 0,
}
O4_RUN = "runs/aim/pii-gs30-titles-v6-g50-seed2-4000/runs/20260928T142402Z.json"
# The O4 lineage root (pii-ont3-context-large-pretrained-masked-*-v1) fit
# pretrained XLM-R large at these rates; O4 itself continued at 2e-5 / 1e-5.
ROOT_LEARNING_RATES = {"--lr": "5e-5", "--encoder-lr": "3e-5"}


def recorded_trainer_options(record_path: str) -> list[str]:
    """Trainer options from a shipped run record, without driver-bound values."""
    records = json.loads((SOFTWARE / "records/paper-run-records.json").read_text())["records"]
    record = next((item for item in records if item["path"] == record_path), None)
    if record is None:
        raise ValueError(f"paper-run-records.json lacks {record_path}")
    argv = record["record"]["params"]["command"]["argv"]
    position = next(i for i, value in enumerate(argv) if value.endswith("scripts/pii_encoder_train.py"))
    options, remaining = [], argv[position + 1 :]
    while remaining:
        value = remaining.pop(0)
        name = value.split("=", 1)[0]
        if name in DRIVER_BOUND:
            if "=" not in value:
                del remaining[: DRIVER_BOUND[name]]
            continue
        options.append(value)
    return options


def replace_option(options: list[str], name: str, value: str) -> list[str]:
    position = options.index(name)
    return [*options[: position + 1], value, *options[position + 2 :]]


def recipe_options(recipe: str, data: Path) -> list[str]:
    if recipe == "affine":
        return [
            "--decoder",
            "linear",
            "--head-kind",
            "affine",
            "--training-windowing",
            "token-capacity",
            "--sampling-length-window-steps",
            "1000",
            "--selection-metric",
            "span-f1",
            "--victory-lap-lr-scale",
            "0",
        ]
    options = recorded_trainer_options(O4_RUN)
    declared = data / "language-round.yaml"
    if declared.is_file():
        # O4's round names its private pools' 35 languages; compiled public
        # data declares the languages it actually samples.
        options = [
            f"--language-round={declared.resolve()}" if value.startswith("--language-round=") else value
            for value in options
        ]
    if recipe in ("o4-fresh", "head-init"):
        for name, value in ROOT_LEARNING_RATES.items():
            options = replace_option(options, name, value)
    if recipe == "head-init":
        # Same options in the native Ont3 label space: no old-space map, no mapped head.
        position = options.index("--dual-head-old-weight-schedule")
        options = options[:position] + options[position + 2 :]
        options.remove("--dual-head-mapped-single-head")
        return [*options, "--native-new-label-space"]
    mapping = data / "mapping.json"
    if not mapping.is_file():
        raise ValueError(f"the {recipe} recipe needs {mapping}; compile data with pii_public_mixture.py")
    options = [*options, "--dual-head-map", str(mapping.resolve())]
    if recipe == "o4-fresh":
        # The head-initialized parent is native and unmapped; bind its first map.
        options.append("--dual-head-bind-native-map")
    return options


def train_command(args: argparse.Namespace) -> dict:
    if args.steps is None:
        args.steps = 12000 if args.recipe == "o4-fresh" else 4000
    steps = max(1, math.ceil(args.steps * args.step_scale))
    if args.max_steps is not None:
        steps = min(steps, args.max_steps)
    for name in ("train.jsonl", "val.jsonl", "labels.json"):
        if not (args.data / name).is_file():
            raise ValueError(f"training data is missing {name}")
    if args.recipe == "o4" and args.init_from_checkpoint is None:
        raise ValueError("--recipe o4 continues a parent; give --init-from-checkpoint or use o4-fresh")
    if args.recipe == "o4-fresh" and args.init_from_checkpoint is not None:
        raise ValueError(
            "--recipe o4-fresh starts from the base encoder; use --recipe o4 to continue a parent"
        )
    extra = args.extra[1:] if args.extra[:1] == ["--"] else args.extra
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=False)
    settings = {
        "recipe": args.recipe,
        "recipe_source": O4_RUN if args.recipe != "affine" else None,
        "parent": str(args.init_from_checkpoint) if args.init_from_checkpoint else None,
        "requested_steps": args.steps,
        "step_scale": args.step_scale,
        "max_steps": args.max_steps,
        "effective_steps": steps,
        "smoke": steps < args.steps,
    }
    parent = args.init_from_checkpoint.resolve() if args.init_from_checkpoint else None
    receipts = {}
    if args.recipe == "o4-fresh":
        # The mapped O4 head only continues an existing Ont3 head, so create one
        # with a single native-label update on the shipped root rows.
        root = args.data / "root"
        if not (root / "labels.json").is_file():
            raise ValueError(f"{root} is missing; compile the mixture with --base to get root rows")
        init = run_logged(
            trainer_command(args, "head-init", root, output / "head-init" / "model", 1, 1, None, []),
            output / "head-init",
            settings={**settings, "phase": "Ont3 affine head initialization, one update"},
        )
        receipts["head_init"] = init["receipt"]
        parent = output / "head-init/model/checkpoint-1"
        if not (parent / "config.json").is_file():
            raise RuntimeError(f"head initialization left no checkpoint at {parent}")
    command = trainer_command(
        args, args.recipe, args.data, output / "fit" / "model", steps, args.eval_steps, parent, extra
    )
    fit = run_logged(command, output / "fit", settings=settings)
    receipts["fit"] = fit["receipt"]
    return {"ok": True, "model": str(output / "fit/model"), "receipts": receipts, "settings": settings}


def trainer_command(args, recipe, data, model_out, steps, eval_steps, parent, extra):
    command = [
        "nice",
        "-n",
        "10",
        workflow_python(),
        "scripts/pii_encoder_train.py",
        "--data",
        str(data.resolve()),
        "--out",
        str(model_out),
        "--model",
        args.model,
        *recipe_options(recipe, data),
        "--max-steps",
        str(steps),
        "--eval-steps",
        str(min(eval_steps, steps)),
        "--batch",
        str(args.batch),
        "--grad-accum",
        str(args.grad_accum),
        "--seed",
        str(args.seed),
        "--keep-final-checkpoint",
    ]
    if parent is not None:
        command.extend(("--init-from-checkpoint", str(parent)))
    owned = {value.split("=", 1)[0] for value in command if value.startswith("--")}
    if any(value.split("=", 1)[0] in owned for value in extra):
        raise ValueError("native options cannot override driver-owned or recipe training settings")
    return [*command, *extra]


DEFAULT_HF_TEACHER = "google/gemma-4-31B-it"
CODEX_HOME_CONFIG = ROOT / "scripts/codex-annotation-home.config.toml"
DEFAULT_CODEX_HOME = Path.home() / ".codex-pii-annotate"
CODEX_HOME_MARKER = ".pii-reproduce-codex-home"
# Context Codex may add to a home; the annotation runner refuses a home holding any.
CODEX_HOME_CONTEXT = ("memories", "plugins", "skills")


def prepare_codex_home(home: Path, login_home: Path) -> Path:
    """Build or refresh the driver's isolated annotation home from the user's Codex login.

    The home gets only the shipped tool-free configuration and a copy of the
    login's `auth.json`. Each run recopies both and clears context Codex
    recreates between runs, so a refreshed login or a stale plugin never
    reaches the annotator. A directory the driver did not create is refused.
    """
    auth = login_home / "auth.json"
    if not auth.is_file():
        raise ValueError(f"no Codex login at {auth}; run `codex login` first, or pass --codex-home")
    marker = home / CODEX_HOME_MARKER
    if home.exists() and not marker.is_file():
        raise ValueError(
            f"{home} exists and was not created by pii-reproduce.py; pass it as --codex-home or remove it"
        )
    home.mkdir(mode=0o700, parents=True, exist_ok=True)
    marker.touch()
    for name in CODEX_HOME_CONTEXT:
        if (home / name).is_dir():
            shutil.rmtree(home / name)
    for path in home.glob("AGENTS*.md"):
        path.unlink()
    (home / "config.toml").write_bytes(CODEX_HOME_CONFIG.read_bytes())
    copy = home / "auth.json"
    copy.write_bytes(auth.read_bytes())
    copy.chmod(0o600)
    return home


def codex_login_home() -> Path:
    return Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex")


def codex_home_command(args: argparse.Namespace) -> dict:
    home = prepare_codex_home(args.home or DEFAULT_CODEX_HOME, args.login or codex_login_home())
    return {"codex_home": str(home), "config": str(CODEX_HOME_CONFIG.relative_to(ROOT))}


PROMPT_REVISIONS = {
    # Rebuilds the paper's prompted Gemma-31B run byte for byte
    # (prompts/pii-label/paper-eval/README.md).
    "paper-prompted": {
        "task": "paper-eval/task.txt",
        "catalog": "paper-eval/catalog.md",
        "examples": "paper-eval/examples.json",
        "source_paragraph_guidance": True,
        "max_new": 3072,
        "format_retries": 1,
    },
    # The prompt O4's teacher used for its training rows.
    "paper-teacher": {
        "task": "task-ont3-primary-annotate-context-v9.txt",
        "catalog": "catalog-ontology-v3-primary-v3.md",
        "examples": "examples-reviewed-final35-recall-ont3-v2.json",
        "source_paragraph_guidance": False,
        "max_new": None,
        "format_retries": None,
    },
}


def annotate_command(args: argparse.Namespace) -> dict:
    codex = args.codex_model is not None
    if codex and (args.server or args.model):
        raise ValueError(
            "--codex-model selects the Codex route; --server and --model apply to Hugging Face teachers"
        )
    if not codex and args.codex_home is not None:
        raise ValueError("--codex-home applies only to --codex-model")
    if codex and args.codex_home is None:
        codex_home = prepare_codex_home(DEFAULT_CODEX_HOME, codex_login_home())
    else:
        codex_home = args.codex_home
    model = args.codex_model if codex else args.model or DEFAULT_HF_TEACHER
    output = args.out.resolve()
    prompt = ROOT / "prompts/pii-label"
    revision = PROMPT_REVISIONS[args.prompt_revision]
    catalog = prompt / revision["catalog"]
    tags = re.findall(r"^\| `([^`]+)` \|", catalog.read_text(), re.MULTILINE)
    if not tags:
        raise ValueError("annotation catalog has no tag definitions")
    if revision["source_paragraph_guidance"]:
        # The paper's prompted run listed its inventory alphabetically.
        tags = sorted(tags)
    command = [
        workflow_python(),
        "scripts/pii_api_label.py" if codex or args.server else "scripts/pii_llm_label.py",
        "--model",
        model,
        "--gold",
        str(args.input.resolve()),
        "--out",
        str(output / "predictions.jsonl"),
        "--lang",
        args.lang,
        "--fmt",
        "json-seq",
        "--tags",
        ",".join(tags),
        "--tag-catalog",
        str(catalog),
        "--task-template",
        str(prompt / revision["task"]),
        "--examples",
        str(prompt / revision["examples"]),
        "--prompt-contract-out",
        str(output / "prompt-contract.json"),
    ]
    if revision["source_paragraph_guidance"]:
        command.append("--source-paragraph-guidance")
    if revision["max_new"] is not None:
        command.extend(("--max-new", str(revision["max_new"])))
    if revision["format_retries"] is not None and (args.server or codex):
        command.extend(("--format-retries", str(revision["format_retries"])))
    if args.limit is not None:
        command.extend(("--limit", str(args.limit)))
    if codex and args.verification_replay:
        raise ValueError("--verification-replay runs on a local or --server teacher")
    if args.concurrency is not None and not (args.server or codex):
        raise ValueError(
            "--concurrency applies to --server or --codex-model; the in-process teacher runs one row at a time"
        )
    if args.server or codex:
        # Same prompt and parser; the API runner sends requests concurrently and
        # checks the same admission receipts itself.
        if args.server:
            concurrency = args.concurrency or 64
            command.extend(("--backend", "openai", "--url", args.server.rstrip("/") + "/chat/completions"))
            command.extend(("--effort", "none"))
        else:
            # As the paper's Luna batches ran: low effort, six sessions at a time.
            concurrency = args.concurrency or 6
            command.extend(("--backend", "codex", "--effort", "low", "--retries", "0"))
            command.extend(("--codex-home", str(codex_home.expanduser().resolve())))
            command.extend(("--codex-workdir", str(output)))
        # Rows are written in order per wave; a wide wave keeps every request slot busy.
        command.extend(("--concurrency", str(concurrency), "--wave", str(8 * concurrency)))
        command.extend(("--raw-out", str(output / "raw.jsonl")))
        for field in ("dedup_receipt", "reannotation_receipt", "evaluation_replay_admission", "web_receipt"):
            if path := getattr(args, field):
                command.extend(("--" + field.replace("_", "-"), str(path.resolve())))
        if args.verification_replay:
            command.append("--verification-replay")
    else:
        # The local runner uses the same input admission checks as the API runner.
        from scripts.pii_dedup_gate import (
            require_annotation_dedup,
            require_evaluation_replay,
            require_o4_web_intake,
            require_training_reannotation,
        )

        docs = [json.loads(line) for line in args.input.read_text().splitlines()]
        if args.limit is not None:
            docs = docs[: args.limit]
        if args.verification_replay:
            pass  # recorded bypass: the receipt says so and no training rows are written
        elif args.dedup_receipt:
            require_annotation_dedup(args.dedup_receipt, args.input, docs)
        elif args.reannotation_receipt:
            require_training_reannotation(args.reannotation_receipt, args.input, docs)
        elif args.web_receipt:
            require_o4_web_intake(args.web_receipt, docs)
        else:
            require_evaluation_replay(args.evaluation_replay_admission, args.input, docs)
        command.extend(("--batch", "1"))
    result = run_logged(
        command,
        output,
        settings={
            "annotator": model,
            "backend": "codex-exec" if codex else "server" if args.server else "transformers",
            "server": args.server,
            "prompt_revision": args.prompt_revision,
            "verification_replay": args.verification_replay,
            "limit": args.limit,
            "smoke": args.limit is not None,
        },
    )
    if args.verification_replay:
        # Replayed rows verify a published result; they never become training data.
        return {**result, **prompts_by_language(output), "training_rows": None}
    return {**result, **prompts_by_language(output), **training_rows(args.input, output, model)}


def prompts_by_language(output: Path) -> dict:
    """Keep each language's complete rendered prompt beside the run as metadata.

    The labeler's prompt contract renders one real prompt per input language;
    this copy lets a reader see and hash the exact instructions, catalog and
    examples every language received without rebuilding them.
    """
    contract = json.loads((output / "prompt-contract.json").read_text(encoding="utf-8"))
    prompts = {
        sample["language"]: {
            "example_language": sample["example_language"],
            "prompt_sha256": sample["prompt_sha256"],
            "prompt": sample["prompt"],
        }
        for sample in contract["samples"]
    }
    path = output / "prompts-by-language.json"
    path.write_text(json.dumps(prompts, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"prompts_by_language": str(path), "prompt_languages": sorted(prompts)}


def training_rows(input_path: Path, output: Path, model: str) -> dict:
    """Join teacher predictions with their input text as Ont3 training rows for `mixture --annotated`.

    The prompt asks the teacher for every Ont3 type, so its rows are complete
    supervision, as O4's teacher rows were. Rows the parser could not read
    cleanly are left out.
    """
    inputs = {row["id"]: row for row in map(json.loads, input_path.read_text(encoding="utf-8").splitlines())}
    kept = dropped = 0
    with (output / "training-rows.jsonl").open("x", encoding="utf-8") as sink:
        for prediction in map(
            json.loads, (output / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
        ):
            stats = prediction.get("parse_stats", {})
            if any(stats.get(key) for key in ("bad_json", "unknown_tag", "unmatched")):
                dropped += 1
                continue
            row = inputs[prediction["id"]]
            spans = sorted([span["start"], span["end"], span["label"]] for span in prediction["preds"])
            sink.write(
                json.dumps(
                    {
                        "id": row["id"],
                        "text": row["text"],
                        "lang": row["lang"],
                        "spans": spans,
                        "label_space": "v2",
                        "supervision": "complete",
                        "unknown_primary_types": [],
                        "src": f"teacher:{model}",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            kept += 1
    return {
        "training_rows": str(output / "training-rows.jsonl"),
        "rows_kept": kept,
        "rows_dropped_parse": dropped,
    }


def workflow_python() -> str:
    executable = SOFTWARE / ".pixi/envs/default/bin/python"
    if not executable.is_file():
        raise ValueError("reproduction environment is missing; run pii-reproduce.py install")
    return str(executable)


def source_command(args: argparse.Namespace) -> dict:
    command = [
        workflow_python(),
        "scripts/pii_onboard_sources.py",
        args.native_command,
        args.source,
        "--source-root",
        str(args.source_root.resolve()),
        "--output-root",
        str(args.output_root.resolve()),
    ]
    cap = getattr(args, "max_records_per_source", None)
    if cap is not None:
        command.extend(("--max-records", str(cap)))
    return run_logged(
        command,
        args.log_dir,
        settings={"source": args.source, "max_records_per_source": cap, "smoke": cap is not None},
    )


def install_command(args: argparse.Namespace) -> dict:
    return run_logged(
        ["pixi", "install", "--manifest-path", str(SOFTWARE / "pixi.toml")],
        args.log_dir,
        settings={"purpose": "environment installation"},
    )


def redaction_arguments(command) -> None:
    command.add_argument(
        "--public-redactions",
        type=Path,
        default=SOFTWARE / "public-redactions.json",
        help="Internal-infrastructure replacements every export applies; never exported itself.",
    )
    command.add_argument(
        "--redactions",
        type=Path,
        default=SOFTWARE / "review-redactions.json",
        help="Identity replacements --anonymous adds; never exported itself.",
    )


def redaction_profiles(args: argparse.Namespace, anonymous: bool) -> list[Path]:
    profiles = [args.public_redactions, *([args.redactions] if anonymous else [])]
    if missing := [str(profile) for profile in profiles if not profile.is_file()]:
        raise ValueError(f"missing redaction profile: {', '.join(missing)}")
    return profiles


def stage_command(args: argparse.Namespace) -> dict:
    from scripts.pii_reproduction_export import stage_sources

    return stage_sources(
        ROOT,
        args.out,
        readme=build_parser().format_help(),
        redactions=redaction_profiles(args, args.anonymous),
        anonymous=args.anonymous,
    )


def package_command(args: argparse.Namespace) -> dict:
    from scripts.pii_reproduction_export import build_archive, stage_files, tracked_files

    profiles = redaction_profiles(args, args.anonymous)
    args.out.mkdir(parents=True, exist_ok=False)
    source = None
    if args.from_repo:
        # The archive is a function of a published commit (plus redaction), not of the draft tree.
        files, readme = tracked_files(args.from_repo)
        staged = stage_files(
            files, args.out / "software", readme=readme, redactions=profiles, anonymous=args.anonymous
        )
        head = git_output(args.from_repo, "rev-parse", "HEAD").strip()
        source = {"repository": str(args.from_repo.resolve()), "commit": head}
        # Kept beside the archive, never inside it: a public commit hash identifies the authors.
        (args.out / "source.json").write_text(json.dumps(source, indent=2) + "\n")
    else:
        staged = stage_command(
            argparse.Namespace(
                out=args.out / "software",
                anonymous=args.anonymous,
                redactions=args.redactions,
                public_redactions=args.public_redactions,
            )
        )
    result = {**build_archive(args.out / "software", args.out / "software.tgz"), "stage": staged}
    if source:
        result["source"] = source
    (args.out / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    return {"ok": True, **result}


def git_output(repository: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repository), *args], check=True, capture_output=True, text=True
    ).stdout


PUBLISH_TAG_PREFIX = "pii-span/"


def publish_command(args: argparse.Namespace) -> dict:
    """Stage the public release into a git repository's worktree and index, without committing.

    The repository holds only published releases, so its release files are
    replaced wholesale. Ignored files (the installed environment, run outputs,
    verify's recorded results) stay. The repository may hold an earlier
    uncommitted publish, which is fully staged; any unstaged or untracked
    change is someone's edit and stops the publish.

    The draft files it ships must be committed, so the release maps to one
    draft commit, which the publisher tags PUBLISH_TAG_PREFIX + version after
    committing in the repository. --uncommitted lifts that for a test publish
    whose files are verified before anything is committed; publishing the same
    files again once they are committed reproduces the verified tree.
    """
    import tempfile

    from scripts.pii_reproduction_export import export_files, stage_files

    repo = args.repo.resolve()
    if not (repo / ".git").exists():
        raise ValueError(f"{repo} is not a git repository; create it with an empty initial commit first")
    status = git_output(repo, "status", "--porcelain").splitlines()
    if edits := [line for line in status if line[1] != " "]:
        raise ValueError(f"{repo} has unstaged or untracked changes:\n" + "\n".join(edits))
    files = export_files(ROOT)
    shipped = [str(source.relative_to(ROOT)) for source, _ in files]
    dirty = git_output(ROOT, "status", "--porcelain", "--", *shipped)
    if dirty and not args.uncommitted:
        raise ValueError(f"commit the shipped draft files first, or test with --uncommitted:\n{dirty}")
    with tempfile.TemporaryDirectory(prefix=".pii-publish-", dir=repo.parent) as temporary:
        staged = stage_files(
            files,
            Path(temporary) / "software",
            readme=build_parser().format_help(),
            redactions=redaction_profiles(args, anonymous=False),
            anonymous=False,
        )
        release_files = git_output(repo, "ls-files", "-z", "--cached", "--others", "--exclude-standard")
        for name in release_files.split("\0"):
            if name and (repo / name).is_file():
                (repo / name).unlink()
        for directory in sorted((d for d in repo.rglob("*") if d.is_dir()), reverse=True):
            if ".git" not in directory.relative_to(repo).parts and not any(directory.iterdir()):
                directory.rmdir()
        for source in sorted(Path(staged["stage"]).rglob("*")):
            if source.is_file():
                target = repo / source.relative_to(staged["stage"])
                target.parent.mkdir(parents=True, exist_ok=True)
                source.rename(target)
    git_output(repo, "add", "--all")
    draft_head = git_output(ROOT, "rev-parse", "HEAD").strip()
    tags = git_output(ROOT, "tag", "--list", f"{PUBLISH_TAG_PREFIX}*", "--sort=-creatordate").split()
    since = tags[0] if tags else None
    log_range = [f"{since}..HEAD"] if since else ["HEAD"]
    draft_log = git_output(ROOT, "log", "--no-merges", "--format=%h %s", *log_range, "--", *shipped)
    release = (
        f"review `git -C {repo} diff --cached`, commit it with a message describing what changed "
        f"for users, tag the repository vX.Y, then tag this draft commit {PUBLISH_TAG_PREFIX}vX.Y"
    )
    return {
        "ok": True,
        "repository": str(repo),
        "staged": git_output(repo, "diff", "--cached", "--shortstat").strip() or "no changes",
        "files": staged["files"],
        "draft_commit": draft_head,
        "draft_uncommitted": dirty.splitlines(),
        "previous_release_tag": since,
        "draft_commits": draft_log.splitlines(),
        "next": (
            f"test publish of uncommitted draft files: run `python3 scripts/verify` in {repo}; once it "
            "passes, commit the draft files and publish again (identical files keep verify's result, "
            f"`scripts/verify --passed`), then {release}"
            if dirty
            else release
        ),
    }


def files_command(args: argparse.Namespace) -> dict:
    from scripts.pii_reproduction_export import python_dependencies

    paths = python_dependencies(ROOT)
    return {
        "ok": True,
        "files": [str(path.relative_to(ROOT)) for path in paths],
        "count": len(paths),
        "bytes": sum(path.stat().st_size for path in paths),
        "status": "Python dependency inventory; non-Python assets and release review pending",
    }


def doctor_command(args: argparse.Namespace) -> dict:
    import shutil

    return {
        "ok": True,
        "root": str(ROOT),
        "python": sys.executable,
        "python_version": sys.version.split()[0],
        "tools": {name: shutil.which(name) for name in ("pixi", "git", "codex")},
        "acli": str(Path(acli.__file__).resolve()),
        "ya_required": False,
    }


def readme_command(args: argparse.Namespace) -> dict:
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as output:
        output.write(build_parser().format_help())
    return {"ok": True, "readme": str(args.out.resolve())}


def run_logged(
    command: list[str],
    directory: Path,
    *,
    settings: dict,
    environment_overrides: dict[str, str] | None = None,
) -> dict:
    """Run a workflow step with separate logs and an execution receipt."""
    directory.mkdir(parents=True, exist_ok=False)
    receipt = {
        "command": command,
        "cwd": str(ROOT),
        "settings": settings,
        "started_unix": time.time(),
        "status": "running",
    }
    receipt_path = directory / "receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    # Tools from the pinned environment (git, git-lfs) take precedence over the host's.
    environment_bin = SOFTWARE / ".pixi/envs/default/bin"
    if environment_bin.is_dir():
        environment["PATH"] = f"{environment_bin}{os.pathsep}{environment.get('PATH', '')}"
    environment["AGENTS_RUN_METRICS_ENABLED"] = "0"
    environment["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,garbage_collection_threshold:0.5"
    environment.update(environment_overrides or {})
    with (directory / "stdout.log").open("w") as stdout, (directory / "stderr.log").open("w") as stderr:
        result = subprocess.run(command, cwd=ROOT, env=environment, stdout=stdout, stderr=stderr)
    receipt.update(
        status="completed" if result.returncode == 0 else "failed",
        returncode=result.returncode,
        elapsed_seconds=time.time() - receipt["started_unix"],
    )
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    if result.returncode:
        raise RuntimeError(f"workflow step failed ({result.returncode}); see {directory}")
    return {"ok": True, "receipt": str(receipt_path.resolve()), **receipt}


def main() -> None:
    parser = build_parser()
    acli.maybe_complete(parser)
    args = parser.parse_args()
    try:
        result = args.action(args)
    except (OSError, ValueError, RuntimeError) as error:
        acli.die(str(error), acli.ExitCode.SOFTWARE)
    acli.emit(result, fmt=acli.resolve_format(args))
