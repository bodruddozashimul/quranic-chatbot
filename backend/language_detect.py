"""Script-based language detection. Bengali and Latin scripts do not overlap,
so counting code points is both sufficient and far cheaper than a classifier."""

import re

_BENGALI = re.compile(r"[\u0980-\u09FF]")
_LETTER = re.compile(r"[^\W\d_]", re.UNICODE)


def bengali_ratio(text: str) -> float:
    """Share of letters written in Bengali script, ignoring digits and marks."""
    letters = _LETTER.findall(text)
    if not letters:
        return 0.0
    return len(_BENGALI.findall(text)) / len(letters)


def detect_language(text: str) -> str:
    """Returns 'bengali' or 'english'.

    The threshold is on letters rather than raw characters: the original
    version divided by len(text), so punctuation and spaces diluted the ratio
    and short Bengali questions containing digits could be misread as English.
    """
    return "bengali" if bengali_ratio(text) > 0.20 else "english"
