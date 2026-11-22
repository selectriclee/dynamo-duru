# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Duru admission hosted by the shared ThunderAgent router service."""

import argparse
import math
import os
import sys

import uvloop
from dynamo.duru_router.args import parse_args
from dynamo.duru_router.router import DuruScheduler
from dynamo.runtime import DistributedRuntime, dynamo_worker
from dynamo.thunderagent_router.__main__ import (
    ThunderAgentRouterHandler as BaseRouterHandler,
)
from dynamo.thunderagent_router.__main__ import serve_router


class DuruRouterHandler(BaseRouterHandler):
    scheduler_cls = DuruScheduler


@dynamo_worker()
async def worker(runtime: DistributedRuntime) -> None:
    config = parse_args()
    await serve_router(runtime, config, DuruRouterHandler(runtime, config))


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--alpha", type=float, required=True)
    args, remaining = parser.parse_known_args()
    if not math.isfinite(args.alpha) or args.alpha <= 0:
        parser.error("--alpha must be finite and positive")
    for name in ("DURU_SCORES", "DURU_AUDIT_DIR"):
        if not os.environ.get(name):
            parser.error(f"Set {name} before starting the router")
    os.environ["DURU_ADMISSION_ALPHA"] = str(args.alpha)
    sys.argv = [sys.argv[0], *remaining]
    uvloop.run(worker())


if __name__ == "__main__":
    main()
