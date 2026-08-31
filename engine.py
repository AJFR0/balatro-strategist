"""
Balatro scoring engine — deterministic core for the Balatro Strategist app.

Implements the Balatro scoring pipeline:
  1. Hand detection (with Four Fingers / Shortcut / Smeared Joker flags)
  2. Level-adjusted base chips & mult
  3. Scored cards left-to-right: rank chips, enhancements, editions, seals,
     per-card joker triggers, retriggers
  4. Held-in-hand effects (Steel, Baron, etc.)
  5. Independent joker effects left-to-right (incl. Blueprint/Brainstorm copying)
  6. Final score = chips x mult

Covers every scoring-relevant joker that can be evaluated deterministically;
scaling jokers take their *current* value as input (you read it off your screen).
Economy / shop / utility jokers don't change the score of a played hand, so they
are intentionally out of scoring scope (the app still knows them for synergy).

This is a faithful-but-not-perfect model: probabilistic effects (Lucky cards,
Bloodstone, Misprint) are scored at expected value by default, with an
"optimist mode" that assumes they always hit.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from itertools import combinations
from collections import Counter
from typing import Optional

# ---------------------------------------------------------------------------
# Cards
# ---------------------------------------------------------------------------

RANK_CHIPS = {2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10,
              11: 10, 12: 10, 13: 10, 14: 11}  # J/Q/K = 10, A = 11
RANK_NAMES = {2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8", 9: "9",
              10: "10", 11: "J", 12: "Q", 13: "K", 14: "A"}
NAME_RANKS = {v: k for k, v in RANK_NAMES.items()}
SUIT_NAMES = {"H": "Hearts", "D": "Diamonds", "S": "Spades", "C": "Clubs"}

ENHANCEMENTS = ("none", "bonus", "mult", "wild", "glass", "steel", "stone", "gold", "lucky")
EDITIONS = ("none", "foil", "holo", "polychrome")
SEALS = ("none", "red", "gold", "blue", "purple")


@dataclass
class Card:
    rank: int                      # 2..14 (A=14)
    suit: str                      # H D S C
    enhancement: str = "none"
    edition: str = "none"
    seal: str = "none"

    @property
    def is_face(self) -> bool:
        return self.rank in (11, 12, 13)

    def label(self) -> str:
        s = RANK_NAMES[self.rank] + self.suit
        extras = [x for x in (self.enhancement, self.edition, self.seal) if x != "none"]
        return s + ("(" + ",".join(extras) + ")" if extras else "")


def parse_card(tok: str) -> Card:
    """Parse tokens like 'AH', '10s', 'KD(glass,red)', 'Qc(steel)'."""
    tok = tok.strip()
    mods = []
    if "(" in tok:
        tok, rest = tok.split("(", 1)
        mods = [m.strip().lower() for m in rest.rstrip(")").split(",")]
    tok = tok.upper()
    suit = tok[-1]
    rank_s = tok[:-1]
    if suit not in SUIT_NAMES:
        raise ValueError(f"Bad suit in {tok!r} (use H/D/S/C)")
    if rank_s not in NAME_RANKS:
        raise ValueError(f"Bad rank in {tok!r} (use 2-10, J, Q, K, A)")
    c = Card(NAME_RANKS[rank_s], suit)
    for m in mods:
        if m in ENHANCEMENTS: c.enhancement = m
        elif m in EDITIONS: c.edition = m
        elif m in SEALS: c.seal = m
        else: raise ValueError(f"Unknown modifier {m!r} on {tok!r}")
    return c


def parse_cards(text: str) -> list[Card]:
    text = text.replace(",", " ")
    return [parse_card(t) for t in text.split() if t.strip()]


# ---------------------------------------------------------------------------
# Hands
# ---------------------------------------------------------------------------

HAND_ORDER = ["Flush Five", "Flush House", "Five of a Kind", "Straight Flush",
              "Four of a Kind", "Full House", "Flush", "Straight",
              "Three of a Kind", "Two Pair", "Pair", "High Card"]

# wiki-verified: base_chips, base_mult, chips_per_level, mult_per_level
HAND_BASE = {
    "High Card":       (5, 1, 10, 1),
    "Pair":            (10, 2, 15, 1),
    "Two Pair":        (20, 2, 20, 1),
    "Three of a Kind": (30, 3, 20, 2),
    "Straight":        (30, 4, 30, 3),
    "Flush":           (35, 4, 15, 2),
    "Full House":      (40, 4, 25, 2),
    "Four of a Kind":  (60, 7, 30, 3),
    "Straight Flush":  (100, 8, 40, 4),
    "Five of a Kind":  (120, 12, 35, 3),
    "Flush House":     (140, 14, 40, 4),
    "Flush Five":      (160, 16, 50, 3),
}


@dataclass
class Rules:
    four_fingers: bool = False     # flushes & straights need only 4 cards
    shortcut: bool = False         # straights may gap by 1 rank
    smeared: bool = False          # Hearts=Diamonds, Spades=Clubs
    splash: bool = False           # every played card scores
    pareidolia: bool = False       # every card counts as a face card
    optimist: bool = False         # probabilistic effects always hit (else EV)


def _suit_group(c: Card, rules: Rules) -> str:
    if c.enhancement == "wild":
        return "*"
    if rules.smeared:
        return "RED" if c.suit in "HD" else "BLK"
    return c.suit


def _is_flush(cards: list[Card], rules: Rules) -> bool:
    need = 4 if rules.four_fingers else 5
    real = [c for c in cards if c.enhancement != "stone"]
    if len(real) < need:
        return False
    groups = [_suit_group(c, rules) for c in real]
    wilds = groups.count("*")
    counts = Counter(g for g in groups if g != "*")
    top = max(counts.values()) if counts else 0
    return top + wilds >= need and len(real) >= need


def _is_straight(cards: list[Card], rules: Rules) -> bool:
    need = 4 if rules.four_fingers else 5
    ranks = sorted({c.rank for c in cards if c.enhancement != "stone"})
    if len(ranks) < need:
        return False
    rank_sets = [ranks]
    if 14 in ranks:  # ace-low
        rank_sets.append(sorted({1 if r == 14 else r for r in ranks}))
    max_gap = 2 if rules.shortcut else 1
    for rs in rank_sets:
        run = 1
        for a, b in zip(rs, rs[1:]):
            if 1 <= b - a <= max_gap:
                run += 1
                if run >= need:
                    return True
            else:
                run = 1
    return False


def detect_hand(cards: list[Card], rules: Rules) -> tuple[str, list[int]]:
    """Return (hand_name, indices_of_scoring_cards). Stone cards always score."""
    real = [(i, c) for i, c in enumerate(cards) if c.enhancement != "stone"]
    stones = [i for i, c in enumerate(cards) if c.enhancement == "stone"]
    ranks = Counter(c.rank for _, c in real)
    top = ranks.most_common()
    flush = _is_flush(cards, rules)
    straight = _is_straight(cards, rules)
    n_top = top[0][1] if top else 0
    pair_ranks = [r for r, n in top if n >= 2]

    if n_top >= 5:
        name = "Flush Five" if flush else "Five of a Kind"
    elif flush and n_top >= 3 and len(pair_ranks) >= 2:
        name = "Flush House"
    elif straight and flush:
        name = "Straight Flush"
    elif n_top >= 4:
        name = "Four of a Kind"
    elif n_top >= 3 and len(pair_ranks) >= 2:
        name = "Full House"
    elif flush:
        name = "Flush"
    elif straight:
        name = "Straight"
    elif n_top >= 3:
        name = "Three of a Kind"
    elif len(pair_ranks) >= 2:
        name = "Two Pair"
    elif n_top >= 2:
        name = "Pair"
    else:
        name = "High Card"

    # scoring cards
    if rules.splash:
        scoring = list(range(len(cards)))
    elif name in ("Flush", "Straight", "Straight Flush", "Flush Five", "Flush House",
                  "Five of a Kind", "Full House"):
        scoring = [i for i, _ in real] + stones
    elif name == "Four of a Kind":
        r = top[0][0]
        scoring = [i for i, c in real if c.rank == r] + stones
    elif name in ("Three of a Kind", "Pair"):
        r = top[0][0]
        scoring = [i for i, c in real if c.rank == r] + stones
    elif name == "Two Pair":
        rs = set(pair_ranks[:2])
        scoring = [i for i, c in real if c.rank in rs] + stones
    else:  # High Card
        if real:
            hi = max(real, key=lambda ic: ic[1].rank)[0]
            scoring = [hi] + stones
        else:
            scoring = stones
    return name, sorted(set(scoring))


def contains_hand(cards: list[Card], target: str, rules: Rules) -> bool:
    """Balatro 'contains a X' checks (The Duo etc.): does the played hand contain X?"""
    real = [c for c in cards if c.enhancement != "stone"]
    ranks = Counter(c.rank for c in real)
    top = ranks.most_common()
    n_top = top[0][1] if top else 0
    pairs = sum(1 for _, n in top if n >= 2)
    if target == "Pair": return n_top >= 2
    if target == "Two Pair": return pairs >= 2 or n_top >= 4
    if target == "Three of a Kind": return n_top >= 3
    if target == "Four of a Kind": return n_top >= 4
    if target == "Five of a Kind": return n_top >= 5
    if target == "Straight": return _is_straight(cards, rules)
    if target == "Flush": return _is_flush(cards, rules)
    if target == "Full House": return n_top >= 3 and pairs >= 2
    return detect_hand(cards, rules)[0] == target


# ---------------------------------------------------------------------------
# Jokers
# ---------------------------------------------------------------------------

@dataclass
class JokerState:
    """A joker in your lineup. `value` holds the current value for scaling
    jokers (read it off your screen: e.g. Ride the Bus current +Mult,
    Hologram current xMult)."""
    name: str
    value: Optional[float] = None
    edition: str = "none"          # foil / holo / polychrome add to score


def _suit_match(c: Card, suit: str, rules: Rules) -> bool:
    if c.enhancement == "wild":
        return True
    if rules.smeared:
        return (c.suit in "HD") if suit in ("H", "D") else (c.suit in "SC")
    return c.suit == suit


def _face(c: Card, rules: Rules) -> bool:
    return True if rules.pareidolia else c.is_face

# Effect registry. Each entry: dict describing when/how the joker scores.
# kinds:
#   flat            — unconditional chips/mult/xmult when scored phase hits it
#   per_scored      — chips/mult/xmult for each scored card matching filter
#   per_held        — for each card held in hand matching filter
#   hand_contains   — bonus if played hand contains a hand type
#   retrigger       — retrigger scored cards matching filter
#   state           — scaling joker: uses JokerState.value (chips/mult/xmult)
#   dynamic         — custom lambda(ctx) -> (chips, mult, xmult, note)
J = {}

def _reg(name, **kw):
    J[name] = kw

# --- flat & simple conditionals -------------------------------------------
_reg("Joker", kind="flat", mult=4)
_reg("Jolly Joker", kind="hand_contains", hand="Pair", mult=8)
_reg("Zany Joker", kind="hand_contains", hand="Three of a Kind", mult=12)
_reg("Mad Joker", kind="hand_contains", hand="Two Pair", mult=10)
_reg("Crazy Joker", kind="hand_contains", hand="Straight", mult=12)
_reg("Droll Joker", kind="hand_contains", hand="Flush", mult=10)
_reg("Sly Joker", kind="hand_contains", hand="Pair", chips=50)
_reg("Wily Joker", kind="hand_contains", hand="Three of a Kind", chips=100)
_reg("Clever Joker", kind="hand_contains", hand="Two Pair", chips=80)
_reg("Devious Joker", kind="hand_contains", hand="Straight", chips=100)
_reg("Crafty Joker", kind="hand_contains", hand="Flush", chips=80)
_reg("Half Joker", kind="dynamic",
     fn=lambda ctx: (0, 20, 1, "≤3 cards played") if len(ctx["played"]) <= 3 else None)
_reg("The Duo", kind="hand_contains", hand="Pair", xmult=2)
_reg("The Trio", kind="hand_contains", hand="Three of a Kind", xmult=3)
_reg("The Family", kind="hand_contains", hand="Four of a Kind", xmult=4)
_reg("The Order", kind="hand_contains", hand="Straight", xmult=3)
_reg("The Tribe", kind="hand_contains", hand="Flush", xmult=2)

# --- per scored card ---------------------------------------------------------
_reg("Greedy Joker", kind="per_scored", suit="D", mult=3)
_reg("Lusty Joker", kind="per_scored", suit="H", mult=3)
_reg("Wrathful Joker", kind="per_scored", suit="S", mult=3)
_reg("Gluttonous Joker", kind="per_scored", suit="C", mult=3)
_reg("Fibonacci", kind="per_scored", ranks=(14, 2, 3, 5, 8), mult=8)
_reg("Scary Face", kind="per_scored", face=True, chips=30)
_reg("Even Steven", kind="per_scored", ranks=(2, 4, 6, 8, 10), mult=4)
_reg("Odd Todd", kind="per_scored", ranks=(14, 3, 5, 7, 9), chips=31)
_reg("Smiley Face", kind="per_scored", face=True, mult=5)
_reg("Scholar", kind="per_scored", ranks=(14,), chips=20, mult=4)
_reg("Walkie Talkie", kind="per_scored", ranks=(10, 4), chips=10, mult=4)
_reg("Triboulet", kind="per_scored", ranks=(13, 12), xmult=2)
_reg("Bloodstone", kind="per_scored", suit="H", xmult=1.5, prob=0.5)
_reg("Arrowhead", kind="per_scored", suit="S", chips=50)
_reg("Onyx Agate", kind="per_scored", suit="C", mult=7)
_reg("The Idol", kind="dynamic", needs=("idol_rank", "idol_suit"),
     fn=None)  # handled specially below
_reg("Photograph", kind="dynamic",
     fn=None)  # first scored face card x2 — special-cased
_reg("Ancient Joker", kind="dynamic", needs=("ancient_suit",), fn=None)

# --- held in hand ------------------------------------------------------------
_reg("Baron", kind="per_held", ranks=(13,), xmult=1.5)
_reg("Shoot the Moon", kind="per_held", ranks=(12,), mult=13)
_reg("Raised Fist", kind="dynamic", fn=None)  # 2x rank chips of lowest held card as mult

# --- retriggers ---------------------------------------------------------------
_reg("Sock and Buskin", kind="retrigger", face=True)
_reg("Hack", kind="retrigger", ranks=(2, 3, 4, 5))
_reg("Seltzer", kind="retrigger", all=True)
_reg("Dusk", kind="retrigger", all=True, note="final hand of round only — toggle in app")
_reg("Hanging Chad", kind="retrigger", first=True, times=2)

# --- scaling (current value supplied by you) ----------------------------------
for _n, _k in [("Ride the Bus", "mult"), ("Green Joker", "mult"), ("Red Card", "mult"),
               ("Spare Trousers", "mult"), ("Abstract Joker", "mult"),
               ("Supernova", "mult"), ("Fortune Teller", "mult"), ("Flash Card", "mult"),
               ("Popcorn", "mult"), ("Swashbuckler", "mult"), ("Gros Michel", "mult"),
               ("Cavendish", "xmult"), ("Misprint", "mult"),
               ("Bull", "chips"), ("Blue Joker", "chips"), ("Runner", "chips"),
               ("Ice Cream", "chips"), ("Stone Joker", "chips"), ("Square Joker", "chips"),
               ("Wee Joker", "chips"), ("Castle", "chips"), ("Stuntman", "chips"),
               ("Hologram", "xmult"), ("Vampire", "xmult"), ("Constellation", "xmult"),
               ("Madness", "xmult"), ("Hit the Road", "xmult"), ("Glass Joker", "xmult"),
               ("Obelisk", "xmult"), ("Lucky Cat", "xmult"), ("Canio", "xmult"),
               ("Yorick", "xmult"), ("Campfire", "xmult"), ("Throwback", "xmult"),
               ("Bootstraps", "mult"), ("Baseball Card", "xmult"), ("Ceremonial Dagger", "mult"),
               ("Loyalty Card", "xmult"), ("Steel Joker", "xmult"), ("Driver's License", "xmult"),
               ("Cloud 9", "chips"), ("Acrobat", "xmult"), ("The Idol", "xmult")]:
    if _n not in J:
        _reg(_n, kind="state", stat=_k)
# Some flat notes for well-known defaults
J["Stuntman"] = dict(kind="flat", chips=250, note="-2 hand size")
J["Misprint"] = dict(kind="dynamic",
                     fn=lambda ctx: (0, 23 if ctx["rules"].optimist else 11.5, 1, "0–23 Mult (EV 11.5)"))
J["Abstract Joker"] = dict(kind="dynamic",
                           fn=lambda ctx: (0, 3 * len(ctx["jokers"]), 1, f"+3 per joker"))
J["Acrobat"] = dict(kind="dynamic",
                    fn=lambda ctx: (0, 0, 3, "final hand of round") if ctx.get("final_hand") else None)
J["Blackboard"] = dict(kind="dynamic",
                       fn=lambda ctx: (0, 0, 3, "all held cards ♠/♣")
                       if all(c.suit in "SC" or c.enhancement == "wild" for c in ctx["held"]) else None)
J["Flower Pot"] = dict(kind="dynamic",
                       fn=lambda ctx: (0, 0, 3, "all four suits scored")
                       if _flower_pot(ctx) else None)
J["Seeing Double"] = dict(kind="dynamic",
                          fn=lambda ctx: (0, 0, 2, "club + other suit scored")
                          if _seeing_double(ctx) else None)
J["Raised Fist"] = dict(kind="dynamic", fn=None)
J["Photograph"] = dict(kind="dynamic", fn=None)
J["The Idol"] = dict(kind="dynamic", needs=("idol_rank", "idol_suit"), fn=None)
J["Ancient Joker"] = dict(kind="dynamic", needs=("ancient_suit",), fn=None)
J["Blueprint"] = dict(kind="copy", direction="right")
J["Brainstorm"] = dict(kind="copy", direction="leftmost")
J["Splash"] = dict(kind="rule", rule="splash")
J["Pareidolia"] = dict(kind="rule", rule="pareidolia")
J["Four Fingers"] = dict(kind="rule", rule="four_fingers")
J["Shortcut"] = dict(kind="rule", rule="shortcut")
J["Smeared Joker"] = dict(kind="rule", rule="smeared")


def _flower_pot(ctx) -> bool:
    suits = set()
    wilds = 0
    for i in ctx["scoring_idx"]:
        c = ctx["played"][i]
        if c.enhancement == "wild":
            wilds += 1
        elif c.enhancement != "stone":
            suits.add(c.suit)
    return len(suits) + wilds >= 4


def _seeing_double(ctx) -> bool:
    suits = set()
    wilds = 0
    for i in ctx["scoring_idx"]:
        c = ctx["played"][i]
        if c.enhancement == "wild":
            wilds += 1
        elif c.enhancement != "stone":
            suits.add(c.suit)
    if wilds >= 2: return True
    if "C" in suits and (len(suits) >= 2 or wilds >= 1): return True
    return wilds >= 1 and len(suits) >= 1 and "C" not in suits  # wild covers the club


SUPPORTED = set(J)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

@dataclass
class ScoreResult:
    hand: str
    level: int
    chips: float
    mult: float
    total: int
    steps: list[str] = field(default_factory=list)
    scoring_idx: list[int] = field(default_factory=list)
    unknown_jokers: list[str] = field(default_factory=list)


def _resolve_copy_idx(jokers: list[JokerState], idx: int) -> Optional[int]:
    spec = J.get(jokers[idx].name)
    seen = set()
    while spec and spec.get("kind") == "copy":
        if idx in seen:
            return None
        seen.add(idx)
        if spec["direction"] == "right":
            idx += 1
            if idx >= len(jokers): return None
        else:  # leftmost
            if idx == 0: return None
            idx = 0
        spec = J.get(jokers[idx].name)
    return idx if spec else None


def _resolve_copy(jokers: list[JokerState], idx: int) -> Optional[str]:
    ridx = _resolve_copy_idx(jokers, idx)
    return jokers[ridx].name if ridx is not None else None


def score_hand(played: list[Card], held: list[Card], jokers: list[JokerState],
               levels: dict[str, int] | None = None, rules: Rules | None = None,
               extra: dict | None = None) -> ScoreResult:
    """Score one play. `held` = cards remaining in hand (not played).
    `levels` maps hand name -> level (default 1). `extra` carries context like
    idol_rank/idol_suit/ancient_suit/final_hand."""
    rules = rules or Rules()
    levels = levels or {}
    extra = extra or {}

    # passive rule jokers modify detection
    active_names = [j.name for j in jokers]
    r = Rules(**vars(rules))
    for n in active_names:
        spec = J.get(n)
        if spec and spec.get("kind") == "rule":
            setattr(r, spec["rule"], True)

    hand, scoring_idx = detect_hand(played, r)
    lvl = max(1, int(levels.get(hand, 1)))
    b_chips, b_mult, cpl, mpl = HAND_BASE[hand]
    chips = b_chips + cpl * (lvl - 1)
    mult = b_mult + mpl * (lvl - 1)
    steps = [f"{hand} lvl {lvl}: base {chips} chips × {mult} mult"]
    unknown = sorted(set(n for n in active_names
                         if n not in J) )

    ctx = dict(played=played, held=held, jokers=jokers, scoring_idx=scoring_idx,
               rules=r, final_hand=False)
    ctx.update(extra)

    # ---- retrigger counts per scored card
    retrig = [1] * len(played)
    for jidx, js in enumerate(jokers):
        name = _resolve_copy(jokers, jidx)
        spec = J.get(name) if name else None
        if spec and spec["kind"] == "retrigger":
            for pos, i in enumerate(scoring_idx):
                c = played[i]
                match = (spec.get("all")
                         or (spec.get("face") and _face(c, r))
                         or (spec.get("ranks") and c.rank in spec["ranks"])
                         or (spec.get("first") and pos == 0))
                if match:
                    retrig[i] += spec.get("times", 1)
    for i in scoring_idx:
        if played[i].seal == "red":
            retrig[i] += 1

    # ---- scored cards
    first_face_done = False
    for i in scoring_idx:
        c = played[i]
        for t in range(retrig[i]):
            add_c = 0.0
            add_m = 0.0
            x_m = 1.0
            if c.enhancement == "stone":
                add_c += 50
            else:
                add_c += RANK_CHIPS[c.rank]
            if c.enhancement == "bonus": add_c += 30
            if c.enhancement == "mult": add_m += 4
            if c.enhancement == "glass": x_m *= 2
            if c.enhancement == "lucky":
                add_m += 20 if r.optimist else 20 * 0.2
            if c.edition == "foil": add_c += 50
            if c.edition == "holo": add_m += 10
            if c.edition == "polychrome": x_m *= 1.5
            # per-card joker triggers (in joker order)
            for jidx, js in enumerate(jokers):
                name = _resolve_copy(jokers, jidx)
                spec = J.get(name) if name else None
                if not spec: continue
                if spec["kind"] == "per_scored":
                    ok = True
                    if spec.get("suit") and not _suit_match(c, spec["suit"], r): ok = False
                    if spec.get("ranks") and c.rank not in spec["ranks"]: ok = False
                    if spec.get("face") and not _face(c, r): ok = False
                    if ok:
                        p = spec.get("prob")
                        scale = 1.0 if (p is None or r.optimist) else p
                        add_c += spec.get("chips", 0) * scale
                        add_m += spec.get("mult", 0) * scale
                        if spec.get("xmult"):
                            xm = spec["xmult"]
                            x_m *= xm if (p is None or r.optimist) else (1 + (xm - 1) * p)
                elif name == "Photograph" and not first_face_done and _face(c, r):
                    x_m *= 2
                elif name == "The Idol":
                    ir, isuit = extra.get("idol_rank"), extra.get("idol_suit")
                    if ir and isuit and c.rank == ir and _suit_match(c, isuit, r):
                        x_m *= 2
                elif name == "Ancient Joker":
                    asuit = extra.get("ancient_suit")
                    if asuit and _suit_match(c, asuit, r):
                        x_m *= 1.5
            if _face(c, r):
                first_face_done = True
            chips += add_c
            mult += add_m
            mult *= x_m
            tag = f" (retrigger {t})" if t else ""
            steps.append(f"  {c.label()}{tag}: +{add_c:g} chips"
                         + (f", +{add_m:g} mult" if add_m else "")
                         + (f", ×{x_m:g} mult" if x_m != 1 else ""))

    # ---- held-in-hand effects
    for c in held:
        gold_seal_retrig = 2 if c.seal == "red" else 1
        for t in range(gold_seal_retrig):
            if c.enhancement == "steel":
                mult *= 1.5
                steps.append(f"  {c.label()} held (steel): ×1.5 mult")
            for jidx, js in enumerate(jokers):
                name = _resolve_copy(jokers, jidx)
                spec = J.get(name) if name else None
                if not spec: continue
                if spec["kind"] == "per_held":
                    ok = True
                    if spec.get("ranks") and c.rank not in spec["ranks"]: ok = False
                    if spec.get("suit") and not _suit_match(c, spec["suit"], r): ok = False
                    if ok:
                        mult += spec.get("mult", 0)
                        if spec.get("xmult"):
                            mult *= spec["xmult"]
                        steps.append(f"  {c.label()} held → {name}: "
                                     + (f"+{spec.get('mult')} mult" if spec.get("mult")
                                        else f"×{spec.get('xmult')} mult"))

    # Raised Fist: 2x rank chips of lowest held card added as mult
    if any(_resolve_copy(jokers, i) == "Raised Fist" for i in range(len(jokers))) and held:
        low = min(held, key=lambda c: c.rank if c.enhancement != "stone" else 99)
        if low.enhancement != "stone":
            mult += 2 * RANK_CHIPS[low.rank]
            steps.append(f"  Raised Fist: +{2 * RANK_CHIPS[low.rank]} mult (lowest held {low.label()})")

    # ---- independent jokers, left to right
    for jidx, js in enumerate(jokers):
        ridx = _resolve_copy_idx(jokers, jidx)
        name = jokers[ridx].name if ridx is not None else None
        spec = J.get(name) if name else None
        label = js.name if js.name == name else f"{js.name}→{name}"
        if spec:
            add_c = add_m = 0.0
            x_m = 1.0
            note = ""
            if spec["kind"] == "flat":
                add_c, add_m = spec.get("chips", 0), spec.get("mult", 0)
                x_m = spec.get("xmult", 1)
            elif spec["kind"] == "hand_contains":
                if contains_hand(played, spec["hand"], r):
                    add_c, add_m = spec.get("chips", 0), spec.get("mult", 0)
                    x_m = spec.get("xmult", 1)
                    note = f"contains {spec['hand']}"
            elif spec["kind"] == "state":
                v = jokers[ridx].value
                if v is None:
                    v = {"chips": 0, "mult": 0, "xmult": 1}[spec["stat"]]
                    note = "no current value set"
                if spec["stat"] == "chips": add_c = v
                elif spec["stat"] == "mult": add_m = v
                else: x_m = v
            elif spec["kind"] == "dynamic" and spec.get("fn"):
                out = spec["fn"](ctx)
                if out:
                    add_c, add_m, x_m, note = out
            if add_c or add_m or x_m != 1:
                chips += add_c
                mult += add_m
                mult *= x_m
                bits = []
                if add_c: bits.append(f"+{add_c:g} chips")
                if add_m: bits.append(f"+{add_m:g} mult")
                if x_m != 1: bits.append(f"×{x_m:g} mult")
                steps.append(f"  {label}: " + ", ".join(bits) + (f" ({note})" if note else ""))
        # joker edition bonuses apply regardless of effect support
        if js.edition == "foil":
            chips += 50; steps.append(f"  {js.name} (foil): +50 chips")
        elif js.edition == "holo":
            mult += 10; steps.append(f"  {js.name} (holo): +10 mult")
        elif js.edition == "polychrome":
            mult *= 1.5; steps.append(f"  {js.name} (polychrome): ×1.5 mult")

    total = int(chips * mult)
    steps.append(f"TOTAL: {chips:g} × {mult:g} = {total:,}")
    return ScoreResult(hand, lvl, chips, mult, total, steps, scoring_idx, unknown)


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------

def best_plays(hand_cards: list[Card], jokers: list[JokerState],
               levels: dict[str, int] | None = None, rules: Rules | None = None,
               extra: dict | None = None, top_n: int = 5) -> list[dict]:
    """Enumerate every legal play (1–5 cards) from your current hand and rank
    by score. Returns top_n dicts with play, held, and ScoreResult."""
    n = len(hand_cards)
    results = []
    for k in range(1, min(5, n) + 1):
        for combo in combinations(range(n), k):
            played = [hand_cards[i] for i in combo]
            held = [hand_cards[i] for i in range(n) if i not in combo]
            res = score_hand(played, held, jokers, levels, rules, extra)
            results.append({"played": played, "held": held, "result": res})
    results.sort(key=lambda x: x["result"].total, reverse=True)
    return results[:top_n]
