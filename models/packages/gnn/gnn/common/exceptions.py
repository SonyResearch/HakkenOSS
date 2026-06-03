from __future__ import annotations


class WrongDimensionsError(Exception):
    def __init__(self, message: str = "Wrong dimensions provided"):
        self.message = message
        super().__init__(self.message)
