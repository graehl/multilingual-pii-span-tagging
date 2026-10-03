import logging
import os
from pathlib import Path

default_logger = logging.getLogger("log_format")


def formatlog(*args, **kwargs):
    r = " ".join(str(x) for x in args)
    if len(kwargs):
        r += " | " + " ".join(f"{k}={v}" for k, v in kwargs.items())
    return r


def info(*args, **kwargs):
    default_logger.info(formatlog(*args, **kwargs))


def debug(*args, **kwargs):
    default_logger.debug(formatlog(*args, **kwargs))


def warn(*args, **kwargs):
    default_logger.warning(formatlog(*args, **kwargs))


def _headline_text(text, max_chars=240):
    flat = " | ".join(part.strip() for part in str(text).splitlines() if part.strip())
    if len(flat) > max_chars:
        flat = flat[: max_chars - 3].rstrip() + "..."
    return flat


def headline(*args, also_log=False, **kwargs):
    msg = _headline_text(formatlog(*args, **kwargs))
    if not msg:
        return ""
    path_text = os.environ.get("AGENTCTL_HEADLINE_FILE", "").strip()
    if path_text:
        path = Path(path_text)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(msg + "\n", encoding="utf-8")
    if also_log:
        default_logger.info(msg)
    return msg


def short(name=None, force=True, time=False):
    global default_logger
    logging.basicConfig(level=logging.INFO, force=force, format="[%(levelname).1s]: %(message)s")
    if not name:
        return default_logger
    l = logging.getLogger(name)
    default_logger = l
    return l


def head(x, hd=1, hd2=5):
    """intended for x list [list of] of tuple or string - elide middle of long things showing structure. hd2 used as hd on recurse"""
    if not x:
        return str(x)
    omit = f"[{len(x) - hd * 2}]" if hasattr(x, "__len__") else ""
    return (
        f"{x:.{hd}}"
        if isinstance(x, float)
        else f"{head(x[0], hd=hd2, hd2=hd2)} {' '.join(head(x, 5, hd2) for x in x[1:])}"
        if isinstance(x, tuple)
        else (x if len(x) < hd * 2 + 5 else f"{x[:hd]}...[{omit}]...{x[-hd:]}")
        if isinstance(x, str)
        else f"[{len(x)}]:[{', '.join(head(x, hd=hd2, hd2=hd2) for x in x[:hd])} ,[{omit}]..., {', '.join(head(x, hd=hd2, hd2=hd2) for x in x[-hd:])}]"
        if isinstance(x, list)
        else str(x)
    )


def escape(s, nl=True, backslash=True):
    s = s if isinstance(s, str) else str(s)
    if backslash:
        s = s.replace("\\", "\\\\")
    if nl:
        s = s.replace("\n", "\\n")
    return s


def show(s, chars=40):
    s = s if isinstance(s, str) else str(s)
    return f"{escape(s[:chars])}...{escape(s[-chars:])}" if len(s) > chars * 2 else escape(s)


def preview_items(xs, n=2, chars=120):
    items = list(xs)
    if not items:
        return "[]"
    shown = [show(x, chars=chars) for x in items]
    if len(shown) <= n * 2:
        return f"[{len(shown)}] " + " | ".join(shown)
    first = " | ".join(shown[:n])
    last = " | ".join(shown[-n:])
    return f"[{len(shown)}] first: {first} || last: {last}"
