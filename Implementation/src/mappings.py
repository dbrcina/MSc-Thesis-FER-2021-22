import string
from typing import List

_CHARS = list(string.digits + string.ascii_uppercase)
# i+1 because i=0 represents blank character used for CTC classification
_CHAR2LABEL = {c: i + 1 for i, c in enumerate(_CHARS)}
_LABEL2CHAR = {label: c for c, label in _CHAR2LABEL.items()}


def text2labels(text: str) -> List[int]:
    return [_CHAR2LABEL[c] for c in text]


def labels2text(labels: List[int]) -> str:
    return "".join([_LABEL2CHAR[label] for label in labels])
