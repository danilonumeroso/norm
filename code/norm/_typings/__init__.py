from typing import Optional


class BaseLoader:
    def next(self, batch_size: Optional[int] = None):
        pass
