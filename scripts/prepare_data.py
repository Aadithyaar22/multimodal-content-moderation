#!/usr/bin/env python
"""Build normalized manifests for one or more datasets.

    python scripts/prepare_data.py --dataset hatexplain
    python scripts/prepare_data.py --dataset hateful_memes fakeddit
    python scripts/prepare_data.py --all
    python scripts/prepare_data.py --dataset fakeddit --tier scale --sample-size 40000

Idempotent: already-downloaded files and already-fetched images are reused, so
re-running after an interruption picks up where it left off.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace

from mcm.config import load_data_config
from mcm.data.prepare import PREPARERS
from mcm.utils.logging import get_logger

log = get_logger("prepare_data")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", nargs="+", choices=sorted(PREPARERS), help="datasets to prepare")
    ap.add_argument("--all", action="store_true", help="prepare every enabled dataset")
    ap.add_argument("--tier", choices=["offline", "scale"], help="fakeddit only: override tier")
    ap.add_argument("--sample-size", type=int, help="fakeddit scale tier only: rows to sample")
    args = ap.parse_args()

    specs = load_data_config()

    if args.all:
        names = [n for n, s in specs.items() if s.enabled]
    elif args.dataset:
        names = args.dataset
    else:
        ap.error("pass --dataset NAME [NAME ...] or --all")

    failed: list[str] = []
    for name in names:
        spec = specs.get(name)
        if spec is None:
            log.error("no config entry for %r in configs/data.yaml", name)
            failed.append(name)
            continue

        overrides = {}
        if args.tier and name == "fakeddit":
            overrides["tier"] = args.tier
        if args.sample_size and name == "fakeddit":
            overrides["sample_size"] = args.sample_size
        if overrides:
            spec = replace(spec, options={**spec.options, **overrides})

        log.info("=" * 70)
        log.info("preparing %s (%s)", name, spec.hf_repo)
        log.info("=" * 70)
        t0 = time.time()
        try:
            PREPARERS[name](spec)
            log.info("%s done in %.1fs", name, time.time() - t0)
        except Exception:
            log.exception("%s FAILED", name)
            failed.append(name)

    if failed:
        log.error("failed: %s", ", ".join(failed))
        return 1
    log.info("all requested datasets prepared")
    return 0


if __name__ == "__main__":
    sys.exit(main())
