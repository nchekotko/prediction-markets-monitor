"""Text normalisation and entity extraction for cross-venue matching."""

from __future__ import annotations

import re
import tomllib
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from pathlib import Path

STOPWORDS = frozenset(
    [
        "a",
        "an",
        "the",
        "will",
        "be",
        "is",
        "are",
        "was",
        "of",
        "on",
        "in",
        "at",
        "to",
        "by",
        "for",
        "and",
        "or",
        "vs",
        "v",
        "versus",
        "who",
        "which",
        "what",
        "win",
        "wins",
        "winner",
        "won",
        "beat",
        "game",
        "match",
        "market",
        "price",
        "resolve",
        "resolves",
        "yes",
        "no",
        "than",
        "before",
        "after",
        "during",
        "this",
        "that",
        "it",
        "its",
        "pro",
        "football",
        "decision",
        "meeting",
        "interest",
        "rates",
        "rate",
        "there",
    ]
)

# Phrase-level synonyms applied before tokenisation (longest first).
PHRASE_SYNONYMS: dict[str, str] = {
    "no change": "hold",
    "maintains rate": "hold",
    "maintain rates": "hold",
    "keeps rates": "hold",
    "basis points": "bps",
    "more than": "over",
    "greater than": "over",
    "less than": "under",
    "fewer than": "under",
}

# Token-level synonyms -> canonical token.
TOKEN_SYNONYMS: dict[str, str] = {
    "hike": "increase",
    "hikes": "increase",
    "raise": "increase",
    "raises": "increase",
    "increases": "increase",
    "cut": "decrease",
    "cuts": "decrease",
    "lower": "decrease",
    "lowers": "decrease",
    "decreases": "decrease",
    "maintains": "hold",
    "unchanged": "hold",
    "bp": "bps",
    "above": "over",
    "below": "under",
    "jan": "january",
    "feb": "february",
    "mar": "march",
    "apr": "april",
    "jun": "june",
    "jul": "july",
    "aug": "august",
    "sep": "september",
    "sept": "september",
    "oct": "october",
    "nov": "november",
    "dec": "december",
}

# Mutually exclusive direction words: two contracts with different directions never match.
DIRECTION_GROUPS: tuple[frozenset[str], ...] = (
    frozenset({"increase", "decrease", "hold"}),
    frozenset({"over", "under"}),
)

_NUM = re.compile(
    r"(?<![\w.])[-+>]?\$?\d[\d,]*(?:\.\d+)?\+?\s*(?:k|m|bn|%|bps|bp)?\+?(?![\w])", re.IGNORECASE
)
_NON_ALNUM = re.compile(r"[^a-z0-9.+\-$%> ]+")
_DIGIT_ALPHA = re.compile(r"(\d)([a-z])")
_WS = re.compile(r"\s+")


def ascii_fold(text: str) -> str:
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()


def basic_normalise(text: str) -> str:
    t = ascii_fold(text or "").lower()
    t = t.replace("&", " and ").replace("’", "'").replace("'s", "")
    t = _NON_ALNUM.sub(" ", t)
    return _WS.sub(" ", t).strip()


@dataclass(frozen=True)
class AliasTable:
    phrases: tuple[tuple[str, str], ...]  # (alias, canonical), longest alias first
    safe_short: frozenset[str] = frozenset()  # short aliases that are not English words

    @classmethod
    def load(cls, path: str | Path | None = None) -> AliasTable:
        if path is None:
            raw = resources.files("pmmon.mapping.data").joinpath("aliases.toml").read_bytes()
        else:
            raw = Path(path).read_bytes()
        data = tomllib.loads(raw.decode())
        pairs = [
            (basic_normalise(a), canon)
            for canon, aliases in data.get("entities", {}).items()
            for a in aliases
        ]
        pairs.sort(key=lambda x: -len(x[0]))
        safe = frozenset(basic_normalise(a) for a in data.get("safe_short_aliases", []))
        return cls(tuple(pairs), safe)

    def extract(self, text: str) -> tuple[set[str], str]:
        """Return (canonical entities, text with aliases replaced by canonical ids).

        Aliases of <= 3 characters (``kc``, ``no``, ``was``) collide with English words,
        so they only count when written in upper case in the original text or when the
        whole text is just that alias (an outcome label like "KC").
        """
        upper_tokens = {basic_normalise(t) for t in re.findall(r"\b[A-Z0-9]{2,4}\b", text or "")}
        norm = f" {basic_normalise(text)} "
        whole = norm.strip()
        found: set[str] = set()
        for alias, canon in self.phrases:
            if (
                len(alias) <= 3
                and alias not in self.safe_short
                and alias not in upper_tokens
                and alias != whole
            ):
                continue
            needle = f" {alias} "
            if needle in norm:
                found.add(canon)
                norm = norm.replace(needle, f" {canon} ")
        return found, _WS.sub(" ", norm).strip()


@lru_cache(maxsize=1)
def default_aliases() -> AliasTable:
    return AliasTable.load()


def numbers(text: str) -> set[str]:
    """Numeric tokens (thresholds, lines, strikes) normalised: '$100,000' -> '100000',
    '25 bps' -> '25bps', '>25bps' -> '>25bps', '50+ bps' -> '50+bps'."""
    out: set[str] = set()
    for m in _NUM.finditer(ascii_fold(text or "")):
        tok = m.group(0).lower().replace("$", "").replace(",", "").replace(" ", "")
        prefix = ">" if tok.startswith(">") else ""
        tok = tok.lstrip("+>")
        plus = "+" if tok.endswith("+") else ""
        tok = tok.rstrip("+")
        unit = ""
        for u in ("bps", "bp", "%"):
            if tok.endswith(u):
                unit, tok = ("bps" if u.startswith("bp") else u), tok[: -len(u)]
                break
        if tok.endswith("+"):  # "50+ bps"
            plus, tok = "+", tok[:-1]
        mult = 1.0
        if tok.endswith("k"):
            mult, tok = 1e3, tok[:-1]
        elif tok.endswith("bn"):
            mult, tok = 1e9, tok[:-2]
        elif tok.endswith("m"):
            mult, tok = 1e6, tok[:-1]
        try:
            v = float(tok) * mult
        except ValueError:
            continue
        if 1900 <= v <= 2100 and v.is_integer() and not unit:
            continue  # years are compared via dates, not as numeric lines
        out.add(f"{prefix}{v:g}{plus}{unit}")
    return out


def content_tokens(text: str, aliases: AliasTable | None = None) -> list[str]:
    al = aliases or default_aliases()
    _, replaced = al.extract(text)
    replaced = f" {_DIGIT_ALPHA.sub(r'\1 \2', replaced)} "
    for phrase, canon in sorted(PHRASE_SYNONYMS.items(), key=lambda x: -len(x[0])):
        replaced = replaced.replace(f" {phrase} ", f" {canon} ")
    toks = [TOKEN_SYNONYMS.get(t, t) for t in replaced.split()]
    return [t.strip("?.") for t in toks if t not in STOPWORDS and t.strip("?.")]


def directions(tokens: list[str] | set[str]) -> list[frozenset[str]]:
    """For each direction group, the subset of its words present in ``tokens``."""
    ts = set(tokens)
    return [g & ts for g in DIRECTION_GROUPS]
