"""Interactive REPL behind the reserved `--repl` flag.

Contract: topics/acli-spec.md § REPL package. Each input line is one invocation
of the tool (argv without the program name); Tab completion reuses the
same candidate engine `--acli-complete` serves, so per-candidate help
and data-peeking completers behave identically. Rich menus (candidate
descriptions, a bottom-toolbar slot hint) need the optional
prompt_toolkit; a terminal without it gets stdlib readline completion
plus a banner naming the install command, and a non-TTY stdin runs a
plain prompt-less line loop (pipe a command script to batch invocations
in one process).
"""

from __future__ import annotations

import os
import shlex
import sys
from collections.abc import Callable, Iterable

from .args import candidates


def install_advice() -> str:
    """The command that most likely installs prompt_toolkit for this interpreter."""
    python = sys.executable or "python3"
    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    user_flag = "" if in_venv else " --user"
    return f"{python} -m pip install{user_flag} prompt_toolkit"


def _history_path(prog: str) -> str:
    state = os.environ.get("XDG_STATE_HOME") or os.path.expanduser("~/.local/state")
    directory = os.path.join(state, "acli")
    os.makedirs(directory, exist_ok=True)
    return os.path.join(directory, f"{prog}.history")


def _completion_tokens(text: str) -> list[str] | None:
    """Line prefix -> partial argv; a trailing "" means a fresh token."""
    try:
        tokens = shlex.split(text)
    except ValueError:  # unclosed quote: nothing sensible to offer
        return None
    if not tokens or text != text.rstrip():
        tokens.append("")
    return tokens


# A rewrite hook maps one repl line's tokens into engine argv, giving a
# bound "personality" (almanac's launcher grammar: bare filters = query on
# the bound dataset, verbs pass through). Obligation: preserve the final
# token, which is what completion is completing.
Rewrite = Callable[[list[str]], list[str]]


def _completion_rows(parser, tokens: list[str], rewrite: Rewrite | None):
    """Candidates for a line, honoring a rewrite hook.

    A bound repl accepts two spellings of a line — the personality's
    shorthand and raw engine argv — so completion offers the union:
    rewritten-grammar candidates first, raw-grammar next, deduped. Hint
    rows survive only when nothing insertable matched (a shorthand miss
    should not shout while the other grammar has answers).
    """
    if rewrite is None:
        return candidates(parser, tokens)
    rewritten = rewrite(list(tokens))
    row_sets = [candidates(parser, rewritten)]
    if rewritten != tokens:
        row_sets.append(candidates(parser, tokens))
    insertable: list[dict] = []
    hints: list[dict] = []
    seen: set[str] = set()
    for rows in row_sets:
        for row in rows:
            if row.get("kind") == "hint":
                hints.append(row)
            elif row["completion"] not in seen:
                seen.add(row["completion"])
                insertable.append(row)
    return insertable if insertable else hints[:1]


def execute(parser, line: str, rewrite: Rewrite | None = None) -> int | None:
    """Run one repl line as an invocation; return its exit status (None: blank)."""
    try:
        tokens = shlex.split(line)
    except ValueError as exc:
        print(f"parse error: {exc}", file=sys.stderr)
        return 2
    if not tokens:
        return None
    if rewrite is not None:
        tokens = rewrite(tokens)
    try:
        args = parser.parse_args(tokens)
        if getattr(args, "format", None) is None:
            # Interactive default; explicit per-line format flags still win.
            args.format = "pretty"
        func = getattr(args, "func", None)
        if func is None:
            print("nothing to run for that input", file=sys.stderr)
            return 2
        return int(func(args) or 0)
    except SystemExit as exc:  # argparse errors and die() both raise it
        code = exc.code
        if code is None:
            return 0
        return code if isinstance(code, int) else 1
    except KeyboardInterrupt:
        print("^C", file=sys.stderr)
        return 130


def _plain_reader(prompt: str) -> Callable[[], str]:
    return lambda: input(prompt)


def _readline_reader(
    parser, prompt: str, rewrite: Rewrite | None = None
) -> Callable[[], str]:
    try:
        import readline
    except ImportError:
        return _plain_reader(prompt)

    def complete_word(text: str, state: int) -> str | None:
        buffer = readline.get_line_buffer()[: readline.get_endidx()]
        tokens = _completion_tokens(buffer)
        if tokens is None:
            return None
        matches = [
            row["completion"]
            for row in _completion_rows(parser, tokens, rewrite)
            if row.get("kind") != "hint" and row["completion"].startswith(text)
        ]
        return matches[state] if state < len(matches) else None

    readline.set_completer_delims(" \t")
    readline.set_completer(complete_word)
    readline.parse_and_bind("tab: complete")
    return _plain_reader(prompt)


def _prompt_toolkit_reader(
    parser, prompt: str, history_name: str, rewrite: Rewrite | None = None
) -> Callable[[], str] | None:
    try:
        from prompt_toolkit import PromptSession
        from prompt_toolkit.completion import Completer, Completion
        from prompt_toolkit.history import FileHistory
    except ImportError:
        return None

    state = {"hint": ""}

    class AcliCompleter(Completer):
        def get_completions(self, document, complete_event):
            del complete_event
            state["hint"] = ""
            tokens = _completion_tokens(document.text_before_cursor)
            if tokens is None:
                return
            current = tokens[-1]
            for row in _completion_rows(parser, tokens, rewrite):
                if row.get("kind") == "hint":
                    state["hint"] = row.get("help", "")
                    continue
                yield Completion(
                    row["completion"],
                    start_position=-len(current),
                    display_meta=row.get("help", ""),
                )

    session = PromptSession(
        history=FileHistory(_history_path(history_name)),
        completer=AcliCompleter(),
        complete_while_typing=True,
        bottom_toolbar=lambda: state["hint"] or None,
    )
    return lambda: session.prompt(prompt)


def run(
    parser,
    *,
    input_lines: Iterable[str] | None = None,
    rewrite: Rewrite | None = None,
    prog: str | None = None,
    intro: str | None = None,
) -> int:
    """Serve the repl until EOF or exit/quit; returns the process exit status.

    `rewrite` gives the repl a bound personality (see `Rewrite`); `prog`
    overrides the banner/prompt/history name; `intro` adds one banner
    line describing the personality.
    """
    prog_name = prog or parser.prog or "acli"
    interactive = input_lines is None and sys.stdin.isatty() and sys.stdout.isatty()
    reader: Callable[[], str]
    if input_lines is not None:
        iterator = iter(input_lines)

        def _next_line() -> str:
            try:
                return next(iterator)
            except StopIteration:
                raise EOFError from None

        reader = _next_line
    elif not interactive:
        reader = _plain_reader("")
    else:
        prompt = f"{prog_name}> "
        rich = _prompt_toolkit_reader(parser, prompt, prog_name, rewrite)
        if rich is None:
            print(
                "repl: prompt_toolkit not installed — plain tab completion only.\n"
                f"repl: for menus with inline help, run: {install_advice()}",
                file=sys.stderr,
            )
            reader = _readline_reader(parser, prompt, rewrite)
        else:
            reader = rich
        print(
            f"{prog_name} repl: one command per line (Tab completes); "
            "exit or Ctrl-D ends."
        )
        if intro:
            print(intro)
    while True:
        try:
            line = reader()
        except EOFError:
            return 0
        except KeyboardInterrupt:
            print(file=sys.stderr)
            continue
        if line.strip() in {"exit", "quit"}:
            return 0
        status = execute(parser, line, rewrite)
        if status:
            print(f"# exit {status}", file=sys.stderr)
