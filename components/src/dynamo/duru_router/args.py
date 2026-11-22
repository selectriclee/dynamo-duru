# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Duru router CLI parsing and config assembly."""

from __future__ import annotations

import argparse
from typing import Optional

from dynamo.duru_router.router import ThunderAgentConfig
from dynamo.router.args import (
    DynamoRouterConfig,
    build_ais_perf_config,
    build_kv_router_config,
)
from dynamo.thunderagent_router.args import ThunderAgentArgGroup as BaseArgGroup


class DuruRouterConfig(DynamoRouterConfig):
    """Extends the standalone-router config with ThunderAgent scheduler params."""

    pause_threshold: float
    pause_target: float
    soft_demote_threshold: float
    soft_demote_priority_jump: float
    resume_priority_boost: float
    resume_hysteresis: float
    acting_token_weight: float
    acting_decay_tau_seconds: float
    scheduler_interval_seconds: float
    model_name: Optional[str] = None
    model_path: Optional[str] = None
    tool_call_parser: Optional[str] = None
    reasoning_parser: Optional[str] = None
    publish_sglang_generate: bool = False

    def to_thunderagent_config(self) -> ThunderAgentConfig:
        return ThunderAgentConfig(
            pause_threshold=self.pause_threshold,
            pause_target=self.pause_target,
            soft_demote_threshold=self.soft_demote_threshold,
            soft_demote_priority_jump=self.soft_demote_priority_jump,
            resume_priority_boost=self.resume_priority_boost,
            resume_hysteresis=self.resume_hysteresis,
            acting_token_weight=self.acting_token_weight,
            acting_decay_tau_seconds=self.acting_decay_tau_seconds,
            scheduler_interval_seconds=self.scheduler_interval_seconds,
        )

    def validate(self) -> None:  # type: ignore[override]
        super().validate()
        if not 0.0 <= self.pause_threshold <= 1.0:
            raise ValueError("--pause-threshold must be in [0, 1]")
        if not 0.0 <= self.pause_target <= self.pause_threshold:
            raise ValueError("--pause-target must be in [0, --pause-threshold]")
        if not 0.0 <= self.soft_demote_threshold <= self.pause_threshold:
            raise ValueError(
                "--soft-demote-threshold must be in [0, --pause-threshold]"
            )
        if not 0.0 <= self.resume_hysteresis <= self.pause_threshold:
            raise ValueError("--resume-hysteresis must be in [0, --pause-threshold]")
        if self.acting_token_weight <= 0:
            raise ValueError("--acting-token-weight must be > 0")
        if self.scheduler_interval_seconds <= 0:
            raise ValueError("--scheduler-interval-seconds must be > 0")
        if self.publish_sglang_generate and not self.model_name:
            raise ValueError("--publish-sglang-generate requires --model-name")


class DuruArgGroup(BaseArgGroup):
    # Duru resumes by score/capacity, never by the upstream forced-resume timer.
    include_resume_timeout = False


def parse_args(argv: Optional[list[str]] = None) -> DuruRouterConfig:
    parser = argparse.ArgumentParser(
        description="Dynamo Duru Router: program-level scheduler with "
        "tool-boundary pause/resume on top of native KV-aware routing",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    DuruArgGroup().add_arguments(parser)
    args = parser.parse_args(argv)
    config = DuruRouterConfig.from_cli_args(args)
    config.validate()
    return config


__all__ = [
    "DuruArgGroup",
    "DuruRouterConfig",
    "build_ais_perf_config",
    "build_kv_router_config",
    "parse_args",
]
