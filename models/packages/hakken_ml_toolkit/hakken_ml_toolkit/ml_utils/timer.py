from __future__ import annotations

import time


class TimersCollection:
    def __init__(
        self,
        timer_dict: dict[str, float] | None = None,
        pause_dict: dict[str, float] | None = None,
    ) -> None:
        self.timer_dict = timer_dict if timer_dict is not None else {}
        self.pause_dict = pause_dict if pause_dict is not None else {}

    def tic(self, name: str) -> None:
        self.timer_dict[name] = time.time()

    def toc(self, name: str) -> float:
        assert name in self.timer_dict
        elapsed = time.time() - self.timer_dict[name]
        del self.timer_dict[name]
        return elapsed

    def pause(self, name: str) -> None:
        self.pause_dict[name] = time.time()

    def resume(self, name: str) -> None:
        if name not in self.timer_dict:
            del self.pause_dict[name]
            return
        elapsed = time.time() - self.pause_dict[name]
        self.timer_dict[name] = self.timer_dict[name] + elapsed
        del self.pause_dict[name]
