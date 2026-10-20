"""Reference tier 2 scorer: content-word overlap with the evidence.

This is a baseline, not a grounding model. It shows the `SentenceScorer` contract and gives
benchmarks something to beat; a real deployment plugs in a small trained detector.
"""

from __future__ import annotations

import re

from switchboard.grounding.evidence import Evidence

_STOPWORDS_TEXT = """
    the and for you your yours our ours are was were will would can could should have has had
    with this that these those from into onto about there their them they then than what when
    where which who whom how why just also very really let lets i'm i'll we're we'll you're
    it's that's here okay sure yes yeah thanks thank please right now well all any some more
    being been not but its out get got going see check one sec moment hold minute
"""
DEFAULT_STOPWORDS = frozenset(_STOPWORDS_TEXT.split())
_WORD_RE = re.compile(r"[a-z0-9']+")


class LexicalScorer:
    def __init__(self, stopwords: frozenset[str] = DEFAULT_STOPWORDS, min_length: int = 3) -> None:
        self.stopwords = stopwords
        self.min_length = min_length

    def score(self, sentence: str, evidence: Evidence) -> float:
        words = [
            w.strip("'")
            for w in _WORD_RE.findall(sentence.lower())
            if len(w) >= self.min_length and w not in self.stopwords
        ]
        if not words:
            return 1.0
        known = evidence.words
        hits = sum(1 for w in words if w in known or (w.endswith("s") and w[:-1] in known))
        return hits / len(words)
