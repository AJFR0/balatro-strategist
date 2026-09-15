"""
Data-quality guard: every card table must agree with the balatrowiki.org
extract it was generated from, carry no wikitext leftovers, and reference
only images we know about. Also pins the engine to the wiki-verified
numbers for jokers whose text was wrong in v1.6 and earlier.

    python tests/test_data.py
"""
import json
import os
import re
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
from wikitext import plain  # noqa: E402
from wiki_enrich import effect_text, RENAME  # noqa: E402
from engine import J  # noqa: E402

DATA = os.path.join(ROOT, "data")
ART = re.compile(r"\{\{|\}\}|\[\[|\]\]|<br|<!--")
FAILS = []


def check(cond, msg):
    if not cond:
        FAILS.append(msg)


def main():
    d = json.load(open(os.path.join(DATA, "wiki_dump.json"), encoding="utf-8"))
    imgs = set(json.load(open(os.path.join(DATA, "wiki_images.json")))["files"])
    tables = ["jokers", "tarots", "planets", "spectrals", "vouchers", "tags", "decks",
              "blinds", "enhancements", "editions", "seals", "stakes", "joker_notes"]
    for t in tables:
        df = pd.read_csv(os.path.join(DATA, f"{t}.csv")).fillna("")
        check(len(df) > 0, f"{t}: empty")
        for c in df.columns:
            if df[c].dtype == object:
                bad = df[df[c].astype(str).str.contains(ART)]
                check(bad.empty, f"{t}.{c}: wikitext leftovers in {bad['name'].tolist()[:3] if 'name' in bad else len(bad)}")
        if "image" in df:
            missing = [i for i in df["image"] if i and i not in imgs]
            check(not missing, f"{t}: images not in wiki_images.json: {missing[:3]}")
            check(all(df["image"]), f"{t}: rows without image")

    # jokers ↔ wiki
    jk = pd.read_csv(os.path.join(DATA, "jokers.csv")).fillna("")
    W = {RENAME.get(x["t"], x["t"]): x for x in d["Jokers"] if x.get("rarity") not in (None, "Unknown")}
    check(len(jk) == 150 and set(jk["name"]) == set(W), f"jokers: {len(jk)} rows, name diff {set(jk['name']) ^ set(W)}")
    for _, r in jk.iterrows():
        w = W[r["name"]]
        check(r["effect"] == effect_text(w), f"jokers: effect drift for {r['name']}")
        check(str(r["cost"]) == str(w.get("buyprice", "")).split(" ")[0], f"jokers: cost drift {r['name']}")
        check(r["rarity"] == w["rarity"], f"jokers: rarity drift {r['name']}")
        check(r["archetype"] and r["strategy"], f"jokers: lost hand-written columns for {r['name']}")

    # the engine numbers that the old CSV text contradicted
    check(J["Mad Joker"].get("mult") == 10 and J["Mad Joker"].get("hand") == "Two Pair", "engine: Mad Joker")
    check(J["Clever Joker"].get("chips") == 80 and J["Clever Joker"].get("hand") == "Two Pair", "engine: Clever Joker")
    check(J["Odd Todd"].get("chips") == 31 and 14 in J["Odd Todd"].get("ranks", ()), "engine: Odd Todd")
    check(J["Stuntman"].get("chips") == 250, "engine: Stuntman")
    check(J["Hanging Chad"].get("times") == 2, "engine: Hanging Chad")

    # engine registry numbers vs the wiki-verified card text (+N Mult / +N Chips / XN Mult)
    checked = 0
    for n, spec in J.items():
        if n not in jk.set_index("name").index or spec.get("kind") not in ("per_scored", "flat", "hand_contains", "held"):
            continue
        eff = jk.set_index("name").loc[n]["effect"]
        for key, pat in (("mult", r"\+(\d+) Mult"), ("chips", r"\+(\d+) Chips"), ("xmult", r"X(\d+(?:\.\d+)?) Mult")):
            m = re.search(pat, eff)
            if spec.get(key) is not None and m:
                checked += 1
                check(float(spec[key]) == float(m.group(1)), f"engine vs wiki: {n} {key} {spec[key]} != {m.group(1)}")
    check(checked >= 30, f"engine vs wiki: only {checked} numbers cross-checked")

    # the wiki's hand table agrees with hands.csv (planet chips/mult per level)
    hands = pd.read_csv(os.path.join(DATA, "hands.csv"))
    planets = pd.read_csv(os.path.join(DATA, "planets.csv"))
    for _, p in planets.iterrows():
        m = re.search(r"by \+(\d+) Mult and \+(\d+) Chips", p["effect"])
        h = hands[hands["hand"] == p["hand"]].iloc[0]
        check(m and int(m.group(1)) == h["mult_per_level"] and int(m.group(2)) == h["chips_per_level"],
              f"hands/planets drift: {p['name']}")

    # blinds: the base multipliers the run bar relies on
    bl = pd.read_csv(os.path.join(DATA, "blinds.csv")).set_index("name")
    check(float(bl.loc["The Wall", "score_mult"]) == 4.0, "blinds: The Wall x4")
    check(float(bl.loc["Violet Vessel", "score_mult"]) == 6.0, "blinds: Violet Vessel x6")
    check(float(bl.loc["Big Blind", "score_mult"]) == 1.5, "blinds: Big Blind x1.5")

    for f in FAILS:
        print("  FAIL", f)
    print("ALL PASS" if not FAILS else f"{len(FAILS)} FAILURES")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
