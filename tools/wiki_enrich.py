"""
Regenerate the card tables in data/ from the balatrowiki.org extract
(data/wiki_dump.json), keeping every hand-written column that already
exists (category, scaling, tags, archetype, strategy, test_idea ...).

    python tools/wiki_enrich.py            # rewrites data/*.csv
    python tools/wiki_enrich.py --check    # only prints the diff summary

Source text: balatrowiki.org, CC BY-NC-SA 3.0. Card art © LocalThunk; only
the wiki file name is stored (served through the app's /img/ proxy).
"""
from __future__ import annotations

import json
import os
import re
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")
sys.path.insert(0, HERE)
from wikitext import plain, paragraphs  # noqa: E402

# wiki title → in-game spelling used across the app
RENAME = {"8 Ball": "8 Ball", "Riff-Raff": "Riff-Raff", "Deja Vu": "Déjà Vu",
          "The Soul": "The Soul", "Director's Cut": "Director's Cut"}
# old CSV spellings → in-game spelling (so hand-written columns carry over)
LEGACY = {"8-Ball": "8 Ball", "Riff-raff": "Riff-Raff", "Soul": "The Soul",
          "Director’s Cut": "Director's Cut", "Skip": "Speed", "Top-Up": "Top-up"}

_CURRENTLY = re.compile(r"\s*\(Currently[^)]*\)")
_REMAIN = re.compile(r"\s*\d+ remaining\b")


def effect_text(rec: dict, field: str = "effect") -> str:
    s = plain(rec.get(field) or "", rec["t"])
    s = _CURRENTLY.sub("", s)
    s = _REMAIN.sub("", s)
    s = re.sub(r"\s+([,.;:)])", r"\1", s)
    return s.strip(" ,")


def unlock_text(rec: dict) -> str:
    u = plain(rec.get("unlock") or "", rec["t"])
    return "" if re.match(r"(available|unlocked) from (the )?start", u.lower()) else u


def img(rec: dict) -> str:
    return (rec.get("image") or "").replace(" ", "_")


def _int(v) -> str:
    m = re.match(r"\s*\$?(\d+)", str(v or ""))
    return m.group(1) if m else ""


def load_dump() -> dict:
    with open(os.path.join(DATA, "wiki_dump.json"), encoding="utf-8") as f:
        return json.load(f)


def build(d: dict) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}

    # ---------------- jokers (merge with hand-written columns) -------------
    old = pd.read_csv(os.path.join(DATA, "jokers.csv")).fillna("")
    old["name"] = old["name"].map(lambda n: LEGACY.get(n, n))
    keep = old.set_index("name")[["category", "scaling", "tags", "archetype",
                                  "strategy", "test_idea"]].to_dict("index")
    rows, notes = [], []
    for j in d["Jokers"]:
        if j.get("rarity") in (None, "Unknown"):
            continue                       # stub page (e.g. "Chaos Theory")
        name = RENAME.get(j["t"], j["t"])
        h = keep.get(name, {})
        rows.append({
            "name": name, "rarity": j["rarity"], "cost": _int(j.get("buyprice")),
            "effect": effect_text(j),
            "category": h.get("category", ""), "scaling": h.get("scaling", ""),
            "tags": h.get("tags", ""), "image": img(j),
            "archetype": h.get("archetype", ""), "strategy": h.get("strategy", ""),
            "test_idea": h.get("test_idea", ""),
            "type": j.get("type", ""), "activation": j.get("activation", "") or "",
            "unlock": unlock_text(j), "probability": j.get("probability", "") or "",
            "copyable": int(j.get("compat-copyable") == "1"),
            "perishable": int(j.get("compat-perishable") == "1"),
            "eternal": int(j.get("compat-eternal") == "1"),
            "wiki_number": j.get("number", ""),
        })
        for key, section in (("syn", "Synergies"), ("anti", "Anti-Synergies"),
                             ("strats", "Strategy"), ("ov", "Overview"), ("st", "Notes")):
            for i, p in enumerate(paragraphs(j.get(key), name)):
                notes.append({"name": name, "section": section, "seq": i, "text": p})
    out["jokers"] = pd.DataFrame(rows).sort_values("name").reset_index(drop=True)
    out["joker_notes"] = pd.DataFrame(notes)

    # ---------------- consumables ------------------------------------------
    def consum(key, kind, default_cost):
        rs = []
        for r in d[key]:
            rs.append({"name": RENAME.get(r["t"], r["t"]), "kind": kind,
                       "cost": _int(r.get("buyprice")) or default_cost,
                       "effect": effect_text(r), "unlock": unlock_text(r),
                       "shop": int("cannot be found in shop" not in str(r.get("buyprice", ""))),
                       "image": img(r)})
        return pd.DataFrame(rs).sort_values("name").reset_index(drop=True)

    out["tarots"] = consum("Tarot Cards", "Tarot", "3")
    out["spectrals"] = consum("Spectral Cards", "Spectral", "4")

    planets_old = pd.read_csv(os.path.join(DATA, "planets.csv"))
    pw = {r["t"]: r for r in d["Planet Cards"]}
    planets_old["effect"] = planets_old["name"].map(lambda n: effect_text(pw[n]))
    planets_old["cost"] = "3"
    planets_old["image"] = planets_old["name"].map(lambda n: img(pw[n]))
    planets_old["unlock"] = planets_old["name"].map(lambda n: unlock_text(pw[n]))
    out["planets"] = planets_old

    out["vouchers"] = pd.DataFrame([{
        "name": RENAME.get(v["t"], v["t"]), "cost": "10", "tier": v.get("type", ""),
        "effect": effect_text(v), "requires": unlock_text(v), "image": img(v)}
        for v in d["Vouchers"]]).sort_values(["tier", "name"]).reset_index(drop=True)

    out["tags"] = pd.DataFrame([{
        "name": t["t"].replace(" Tag", ""), "effect": effect_text(t, "description"),
        "min_ante": t.get("ante", ""), "unlock": unlock_text(t), "image": img(t)}
        for t in d["Tags"]]).sort_values("name").reset_index(drop=True)

    out["decks"] = pd.DataFrame([{
        "name": k["t"].replace(" Deck", ""), "effect": effect_text(k, "limit"),
        "unlock": unlock_text(k), "image": img(k), "order": int(k.get("number") or 0)}
        for k in d["Decks"]]).sort_values("order").reset_index(drop=True)

    out["blinds"] = pd.DataFrame([{
        "name": b["t"], "kind": b.get("type", ""), "min_ante": b.get("ante", ""),
        "score_mult": b.get("score", ""), "reward": b.get("reward", ""),
        "effect": effect_text(b, "description"), "image": img(b),
        "order": int(b.get("number") or 0)}
        for b in d["Blinds"]]).sort_values("order").reset_index(drop=True)

    for key, fname, field in (("Enhancements", "enhancements", "effect"),
                              ("Editions", "editions", "effect"),
                              ("Seals", "seals", "effect"),
                              ("Stakes", "stakes", "description")):
        out[fname] = pd.DataFrame([{
            "name": r["t"], "effect": effect_text(r, field), "image": img(r),
            "order": int(r.get("number") or 0)} for r in d[key]]).sort_values(
            ["order", "name"]).reset_index(drop=True)

    # synergy edges / benchmarks: only the spelling changes
    for fname in ("synergy_edges", "joker_benchmarks"):
        df = pd.read_csv(os.path.join(DATA, f"{fname}.csv"))
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].map(lambda v: LEGACY.get(v, v) if isinstance(v, str) else v)
        out[fname] = df
    return out


def main() -> None:
    d = load_dump()
    tables = build(d)
    print(f"wiki fetched {d['meta']['fetched']} from {d['meta']['source']}")
    for k, df in tables.items():
        print(f"  {k:16s} {len(df):5d} rows  cols={list(df.columns)}")
    if "--check" in sys.argv:
        return
    for k, df in tables.items():
        df.to_csv(os.path.join(DATA, f"{k}.csv"), index=False)
    images = sorted({v for df in tables.values() if "image" in df for v in df["image"] if v})
    with open(os.path.join(DATA, "wiki_images.json"), "w") as f:
        json.dump({"base": "https://balatrowiki.org/images/", "files": images,
                   "credit": "Card art © LocalThunk · data via balatrowiki.org (CC BY-NC-SA 3.0)"},
                  f, indent=0)
    print(f"  wiki_images.json {len(images)} files")


if __name__ == "__main__":
    main()
