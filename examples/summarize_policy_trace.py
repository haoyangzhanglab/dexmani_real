"""Summarize one saved attempt trace; no model or device access."""

import argparse
import json
from pathlib import Path

from dexmani_real.deployment.timing import summarize_trace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("attempt", type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize_trace(json.loads(args.attempt.read_text())["trace"]), indent=2))


if __name__ == "__main__":
    main()
