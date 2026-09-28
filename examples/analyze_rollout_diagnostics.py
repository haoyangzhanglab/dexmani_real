#!/usr/bin/env python3
"""Offline: python examples/analyze_rollout_diagnostics.py SESSION_DIRECTORY."""

import argparse

from dexmani_real.deployment.diagnostic_analysis import write_analysis


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session")
    args = parser.parse_args()
    print(write_analysis(args.session))


if __name__ == "__main__":
    main()
