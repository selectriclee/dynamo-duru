# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Duru admission hosted by the shared ThunderAgent router service."""

import argparse
import asyncio
import json
import math
import os
import sys
from pathlib import Path

import uvloop

from dynamo.duru_router.args import parse_args
from dynamo.duru_router.handler import AccountingUnavailable, DuruRouterHandler
from dynamo.duru_router.scoring import Ingress, RouterAccounting, score_channel
from dynamo.runtime import DistributedRuntime, dynamo_worker
from dynamo.thunderagent_router.__main__ import serve_router


async def serve(runtime, config, handler):
    serving = asyncio.create_task(serve_router(runtime, config, handler))
    monitoring = asyncio.create_task(handler.monitor())
    try:
        done, _ = await asyncio.wait(
            (serving, monitoring), return_when=asyncio.FIRST_COMPLETED
        )
        if monitoring in done:
            try:
                monitoring.result()
            except AccountingUnavailable:
                # Keep endpoints available to return explicit request errors
                # and status, while already-dispatched requests finish.
                await serving
                return
        for task in done:
            task.result()
    finally:
        serving.cancel()
        monitoring.cancel()
        await asyncio.gather(serving, monitoring, return_exceptions=True)
        await handler.shutdown()


@dynamo_worker()
async def worker(runtime: DistributedRuntime) -> None:
    config = parse_args()
    accounting_config = os.environ["DURU_ACCOUNTING_CONFIG"]
    settings = json.loads(Path(accounting_config).read_text())
    ingress_path = Path(settings["ingress"])
    if ingress_path.name != "duru-ingress.jsonl":
        raise ValueError("Ingress channel must be named duru-ingress.jsonl")
    os.environ["DURU_SCORES"] = settings["scores"]
    score_channel()
    ingress = Ingress(ingress_path.parent, in_process=True)
    accounting = RouterAccounting(accounting_config, ingress=ingress)
    try:
        accounting.start()
        handler = DuruRouterHandler(
            runtime, config, accounting=accounting, ingress=ingress
        )
        await serve(runtime, config, handler)
    finally:
        try:
            await asyncio.to_thread(ingress.close)
        finally:
            await asyncio.to_thread(accounting.close)


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--alpha", type=float, required=True)
    args, remaining = parser.parse_known_args()
    if not math.isfinite(args.alpha) or args.alpha <= 0:
        parser.error("--alpha must be finite and positive")
    for name in ("DURU_ACCOUNTING_CONFIG",):
        if not os.environ.get(name):
            parser.error(f"Set {name} before starting the router")
    os.environ["DURU_ADMISSION_ALPHA"] = str(args.alpha)
    sys.argv = [sys.argv[0], *remaining]
    uvloop.run(worker())


if __name__ == "__main__":
    main()
