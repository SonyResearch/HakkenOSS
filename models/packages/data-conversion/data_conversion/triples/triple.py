from __future__ import annotations


class Triple:
    def __init__(self, head: str, relation: str, tail: str):
        self.head = head
        self.relation = relation
        self.tail = tail

    @classmethod
    def from_tuple(cls, triple: tuple[str, str, str]) -> Triple:
        return cls(head=triple[0], relation=triple[1], tail=triple[2])

    @staticmethod
    def to_tuple(triple: Triple) -> tuple[str, str, str]:
        return triple.head, triple.relation, triple.tail
