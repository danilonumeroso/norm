import time
import datetime
from contextlib import ContextDecorator
from typing import Callable


class Timer(ContextDecorator):

    def __init__(self,
                 name: str = "Default",
                 logger: Callable[[str], None] = print):

        self.name = name
        self.logger = logger

    def start(self) -> None:
        self._start = time.perf_counter()

    def stop(self) -> float:
        elapsed = time.perf_counter() - self._start

        if self.logger:
            self.logger(f"{self.name}: {datetime.timedelta(seconds=elapsed)}")
        return elapsed

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, t, v, traceback):
        self.stop()
