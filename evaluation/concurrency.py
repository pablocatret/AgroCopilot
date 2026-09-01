"""Adaptive concurrency control for evaluation batches."""
from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass


@dataclass
class ConcurrencyStats:
    initial: int
    current: int
    maximum: int
    reductions: int = 0
    increases: int = 0
    saturation_events: int = 0


class AdaptiveConcurrency:
    """A bounded AIMD controller with a provider-independent active limit."""

    def __init__(self, initial: int, *, maximum: int | None = None) -> None:
        override = os.environ.get("EVALUATION_INITIAL_CONCURRENT", "").strip()
        if override:
            try:
                initial = int(override)
            except ValueError as exc:
                raise ValueError("EVALUATION_INITIAL_CONCURRENT must be an integer") from exc
        if initial < 1:
            raise ValueError("initial concurrency must be >= 1")
        self._limit = initial
        self._maximum = max(initial, maximum or initial * 4)
        self._active = 0
        self._condition = asyncio.Condition()
        self._stable = 0
        self.stats = ConcurrencyStats(initial=initial, current=initial, maximum=initial)

    async def acquire(self) -> None:
        async with self._condition:
            await self._condition.wait_for(lambda: self._active < self._limit)
            self._active += 1

    async def release(self, *, saturated: bool = False) -> None:
        async with self._condition:
            self._active -= 1
            if saturated:
                new_limit = max(1, self._limit // 2)
                if new_limit < self._limit:
                    self._limit = new_limit
                    self.stats.reductions += 1
                self.stats.saturation_events += 1
                self._stable = 0
            else:
                self._stable += 1
                # Add one slot after a full stable window, up to the configured ceiling.
                if self._stable >= max(4, self._limit) and self._limit < self._maximum:
                    self._limit += 1
                    self._stable = 0
                    self.stats.increases += 1
                    self.stats.maximum = max(self.stats.maximum, self._limit)
            self.stats.current = self._limit
            self._condition.notify_all()

    @property
    def current(self) -> int:
        return self._limit


class AdaptiveConcurrencyPool:
    """Independent adaptive controllers for each configured provider."""

    def __init__(self, initial: int) -> None:
        self.initial = initial
        self._controllers: dict[str, AdaptiveConcurrency] = {}

    def _controller(self, provider: str, role: str = "system") -> AdaptiveConcurrency:
        key = f"{role}:{provider or 'unknown'}"
        controller = self._controllers.get(key)
        if controller is None:
            controller = AdaptiveConcurrency(self.initial)
            self._controllers[key] = controller
        return controller

    async def acquire(self, provider: str, role: str = "system") -> None:
        await self._controller(provider, role).acquire()

    async def release(self, provider: str, *, role: str = "system", saturated: bool = False) -> None:
        await self._controller(provider, role).release(saturated=saturated)

    def snapshot(self) -> dict[str, dict[str, int]]:
        return {
            provider: {
                "initial": controller.stats.initial,
                "current": controller.stats.current,
                "maximum": controller.stats.maximum,
                "reductions": controller.stats.reductions,
                "increases": controller.stats.increases,
                "saturation_events": controller.stats.saturation_events,
            }
            for provider, controller in self._controllers.items()
        }
