"""
Tiny wikitext → plain-text normalizer for the balatrowiki.org extract
(data/wiki_dump.json). Handles the templates the wiki actually uses on card
pages; anything unknown collapses to its last positional argument.

    >>> plain("{{hl|green|1 in 4}} chance for each played {{hl|orange|8}}")
    '1 in 4 chance for each played 8'
"""
from __future__ import annotations

import html
import re

_TPL = re.compile(r"\{\{([^{}]*)\}\}")          # innermost template
_LINK = re.compile(r"\[\[([^\[\]|]*)(?:\|((?:[^\[\]]|\[[^\[\]]*\])*))?\]\]")
_FILE = re.compile(r"\[\[(?:File|Image):[^\[\]]*(?:\[\[[^\]]*\]\][^\[\]]*)*\]\]", re.I)
_COMMENT = re.compile(r"<!--.*?-->", re.S)
_TAG = re.compile(r"<[^>]+>")

_SUFFIX = {"d": " Deck", "stake": " Stake", "seal": " Seal", "tag": " Tag",
           "booster": " Pack", "blind": "", "suit": "s", "enhancement": "",
           "edition": "", "sticker": "", "v": "", "j": "", "ph": "", "f": "",
           "achievement": ""}
_BARE = {"tarot": "Tarot", "planet": "Planet", "spectral": "Spectral",
         "room": "(Must have room)", "fullpagename": "", "clear": "", "main": ""}


def _args(body: str) -> tuple[str, list[str], dict[str, str]]:
    parts = re.split(r"\|(?![^\[]*\]\])", body)   # pipes inside [[a|b]] are not args
    name = parts[0].strip().lower()
    pos, kw = [], {}
    for p in parts[1:]:
        k, eq, v = p.partition("=")
        if eq and re.fullmatch(r"\s*[A-Za-z_]+\s*", k):
            kw[k.strip().lower()] = v.strip()
        else:
            pos.append(p.strip())
    return name, pos, kw


_TITLE = [""]                              # current page title for {{FULLPAGENAME}}


def _expand(m: re.Match) -> str:
    name, a, kw = _args(m.group(1))
    last = a[-1] if a else ""
    if name == "fullpagename":
        return _TITLE[0]
    if name in ("joker table", "navbox", "reflist", "clear"):
        return ""
    if name == "currently":
        return f"(Currently {last})"
    if name == "sticker" and kw.get("name"):
        return kw["name"]
    if name == "hl":                      # {{hl|color|text}}
        return a[-1] if len(a) >= 2 else last
    if name in ("mult",):
        return f"{last} Mult"
    if name == "xmult":
        v = last.lstrip("xX×")
        return f"X{v} Mult"
    if name == "chips":
        return f"{last} Chips"
    if name in ("money", "$"):
        v = last.lstrip("$")
        return f"${v}"
    if name in _BARE:
        return last or _BARE[name]
    if name.startswith("#lst") or name.startswith("#lsth"):
        return ""
    if name == "trim":
        return last
    if name in _SUFFIX:
        if name == "suit":                # {{suit|Spade}} → Spades
            return last if last.endswith("s") else last + "s"
        return f"{last}{_SUFFIX[name]}" if last else ""
    return last


def _templates(s: str) -> str:
    for _ in range(8):                    # peel nested templates inside-out
        s2 = _TPL.sub(_expand, s)
        if s2 == s:
            break
        s = s2
    return s


def _br(m: re.Match) -> str:
    """<br> becomes ', ' unless the break sits next to punctuation / a parenthetical."""
    before, after = m.group(1), m.group(2)
    if not before or before[-1] in ",.;:" or not after or after in "([":
        return f"{before} {after}"
    return f"{before}, {after}"


def plain(s: str | None, title: str = "") -> str:
    """Wikitext → one-line plain text."""
    if not s:
        return ""
    _TITLE[0] = title
    s = _COMMENT.sub("", s)
    s = _FILE.sub("", s)
    s = _templates(s)
    s = _LINK.sub(lambda m: (m.group(2) or m.group(1)).strip(), s)
    s = re.sub(r"</?(small|big|span|b|i|u|sup|sub|div|p|nowiki)\b[^>]*>", "", s, flags=re.I)
    s = re.sub(r"\s*(\S?)\s*<br\s*/?>\s*(\S?)", _br, s, flags=re.I)
    s = _TAG.sub("", s)
    s = html.unescape(s)
    s = s.replace("'" * 3, "").replace("'" * 2, "")
    s = re.sub(r"\s*\[\d+\]", "", s)              # "[23]" remaining-counters
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"\s+([,.;:)])", r"\1", s)
    s = re.sub(r"\(\s+", "(", s)
    return s


def paragraphs(s: str | None, title: str = "") -> list[str]:
    """Wiki section body → list of plain paragraphs / bullets (each one line)."""
    if not s:
        return []
    out = []
    _TITLE[0] = title
    s = _templates(_COMMENT.sub("", s))    # multi-line templates first
    for raw in re.split(r"\n+", s):
        raw = raw.strip()
        if not raw or raw.startswith(("{|", "|", "!", "==")):
            continue
        raw = re.sub(r"^[*#:;]+\s*", "", raw)
        p = plain(raw, title)
        if "{{" in p or "<!--" in p or "[[" in p:
            continue                       # truncated tail of the extract
        if p and len(p) > 2:
            out.append(p)
    return out
