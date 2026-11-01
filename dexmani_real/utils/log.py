from __future__ import annotations

import logging
import sys

CONSOLE_FORMATTER = logging.Formatter(
    "[%(asctime)s] %(levelname)s: %(message)s", datefmt="%H:%M:%S"
)


def configure_logging() -> None:
    logger = logging.getLogger("dexmani_real")
    if logger.handlers:
        return
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(CONSOLE_FORMATTER)
    console.setLevel(logging.INFO)
    logger.addHandler(console)
    logger.setLevel(logging.INFO)
    logger.propagate = False


def attach_cleanup_errors(error, errors):
    """按对象及 cause 关系附加后续错误，保留主错，避免重复和反向引用。"""
    def related_ids(root):
        pending, seen = [root], set()
        while pending:
            current = pending.pop()
            if id(current) in seen:
                continue
            seen.add(id(current))
            if current.__cause__ is not None:
                pending.append(current.__cause__)
            pending.extend(detail for _, detail in getattr(current, "cleanup_errors", ()))
        return seen

    known = related_ids(error)
    additions = []
    for stage, detail in errors:
        if id(detail) in known or id(error) in related_ids(detail):
            continue
        additions.append((stage, detail))
        known.update(related_ids(detail))
    if additions:
        error.cleanup_errors = [*getattr(error, "cleanup_errors", ()), *additions]
    return error


def error_detail(stage, error):
    message = str(error)
    if error.__cause__ is not None:
        cause = error.__cause__
        message += f"; caused by {type(cause).__name__}: {error_detail('cause', cause)['message']}"
    for cleanup_stage, detail in getattr(error, "cleanup_errors", ()):
        detail_message = error_detail(cleanup_stage, detail)["message"]
        message += f"; {cleanup_stage}: {type(detail).__name__}: {detail_message}"
    return dict(stage=stage, exception_type=type(error).__name__, message=message)
