"""
Balatro Strategist — FastAPI backend.

The deterministic engine (engine.py) and the Lakebase/Model-Serving layer
(db.py) are unchanged; this file exposes them as JSON APIs and serves the
hand-built SPA in static/index.html. Runs on Databricks Apps.
"""
from __future__ import annotations

import json
import os
import threading
from typing import Any, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

import db
from engine import (J as JSPEC, HAND_BASE, NAME_RANKS, SUPPORTED,
                    JokerState, Rules, best_plays, parse_cards)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")

app = FastAPI(title="Balatro Strategist", docs_url=None, redoc_url=None)

# ---------------------------------------------------------------------------
# Data + boot
# ---------------------------------------------------------------------------
TABLES: dict[str, pd.DataFrame] = {}
DIAG: dict[str, str] = {}


def _load_tables() -> None:
    for name in ["jokers", "hands", "planets", "tarots", "spectrals",
                 "vouchers", "decks", "tags"]:
        TABLES[name] = pd.read_csv(os.path.join(DATA_DIR, f"{name}.csv"))


def _joker_records() -> list[dict]:
    df = TABLES["jokers"].fillna("")
    recs = df.to_dict(orient="records")
    for r in recs:
        spec = JSPEC.get(r["name"], {})
        r["kind"] = spec.get("kind", "")
        r["stat"] = spec.get("stat", "")
        r["supported"] = r["name"] in SUPPORTED
    return recs


@app.on_event("startup")
def boot() -> None:
    _load_tables()
    DIAG.update(db.diagnostics(run_chat_test=True))
    db.build_tfidf(TABLES["jokers"])
    print("=== BALATRO STRATEGIST STARTUP DIAGNOSTICS ===", flush=True)
    for k, v in DIAG.items():
        print(f"  {k}: {v}", flush=True)
    if str(DIAG.get("lakebase", "")).startswith("OK") \
            and str(DIAG.get("embeddings", "")).startswith("OK"):
        def backfill() -> None:
            r = db.ensure_embeddings(TABLES["jokers"])
            print(f"  joker_embeddings backfill: {r['stored']}/{r['total']} stored"
                  + (f" ({r['error']})" if r.get("error") else " — complete"), flush=True)
        threading.Thread(target=backfill, daemon=True).start()
        print("  joker_embeddings: backfill started in background", flush=True)
    print("=== END DIAGNOSTICS ===", flush=True)


def _lakebase_ok() -> bool:
    return str(DIAG.get("lakebase", "")).startswith("OK")


# ---------------------------------------------------------------------------
# Static
# ---------------------------------------------------------------------------
@app.get("/")
def index() -> FileResponse:
    return FileResponse(os.path.join(HERE, "static", "index.html"))


# ---------------------------------------------------------------------------
# Bootstrap payload
# ---------------------------------------------------------------------------
@app.get("/api/bootstrap")
def bootstrap() -> dict:
    stats: dict[str, Any] = {}
    if _lakebase_ok():
        try:
            stats = db.run_stats()
        except Exception:
            stats = {}
    return {
        "jokers": _joker_records(),
        "hands": TABLES["hands"].to_dict(orient="records"),
        "decks": [d for d in TABLES["decks"]["name"].dropna().tolist() if str(d).strip()],
        "diag": DIAG,
        "demo": db.DEMO,
        "lakebase_ok": _lakebase_ok(),
        "semantic_ok": _lakebase_ok() and str(DIAG.get("embeddings", "")).startswith("OK"),
        "instance": db.INSTANCE,
        "endpoints": {"chat": db.CHAT_ENDPOINT, "embed": db.EMBED_ENDPOINT},
        "run_stats": stats,
    }


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------
class LineupItem(BaseModel):
    name: str
    value: Optional[float] = None
    edition: str = "none"


class OptimizeReq(BaseModel):
    hand: str
    lineup: list[LineupItem] = []
    levels: dict[str, int] = {}
    optimist: bool = False
    final_hand: bool = False
    blind_req: int = 0
    idol: Optional[str] = None       # e.g. "KH"
    ancient: Optional[str] = None    # e.g. "H"


_CANON = {n.lower(): n for n in SUPPORTED}
_EDS = {"none": "none", "foil": "foil", "holo": "holo",
        "holographic": "holo", "polychrome": "polychrome", "poly": "polychrome"}


def _canon_joker(name: str) -> str:
    """Forgive case and a missing 'The ' prefix for API callers."""
    n = name.strip()
    low = n.lower()
    return _CANON.get(low) or _CANON.get("the " + low) or n


def _canon_edition(ed: str) -> str:
    return _EDS.get((ed or "none").strip().lower(), "none")


@app.post("/api/optimize")
def optimize(req: OptimizeReq) -> dict:
    try:
        cards = parse_cards(req.hand)
    except ValueError as e:
        raise HTTPException(400, str(e))
    if not 1 <= len(cards) <= 12:
        raise HTTPException(400, "Give me 1-12 cards.")
    extra: dict[str, Any] = {"final_hand": req.final_hand}
    if req.idol:
        extra["idol_rank"] = NAME_RANKS[req.idol[:-1]]
        extra["idol_suit"] = req.idol[-1]
    if req.ancient:
        extra["ancient_suit"] = req.ancient
    jokers = [JokerState(_canon_joker(i.name), i.value,
                         _canon_edition(i.edition)) for i in req.lineup]
    plays = best_plays(cards, jokers, req.levels, Rules(optimist=req.optimist),
                       extra, top_n=5)
    out = []
    for p in plays:
        r = p["result"]
        out.append({
            "played": [c.label() for c in p["played"]],
            "held": [c.label() for c in p["held"]],
            "hand": r.hand, "level": r.level, "chips": r.chips, "mult": r.mult,
            "total": r.total, "steps": r.steps, "unknown": r.unknown_jokers,
        })
    best = out[0]
    verdict = ""
    if req.blind_req:
        verdict = "ok" if best["total"] >= req.blind_req \
            else f"short by {req.blind_req - best['total']:,}"
    return {"plays": out, "verdict": verdict}


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------
class SearchReq(BaseModel):
    q: str
    semantic: bool = True


@app.post("/api/search")
def search(req: SearchReq) -> dict:
    q = req.q.strip()
    if not q:
        return {"hits": [], "note": ""}
    if req.semantic and _lakebase_ok():
        try:
            if db.embedding_count() > 0:
                hits = db.semantic_search(q, top_n=24)
                note = "semantic · Lakebase " + \
                    ("pgvector" if db._state.get("pgvector") else "JSONB")
                n = db.embedding_count()
                if n < 150:
                    note += f" · index warming {n}/150"
                return {"hits": [{"name": h, "sim": s} for h, s in hits], "note": note}
        except Exception as e:
            note = f"keyword fallback ({str(e)[:60]})"
            hits = db.tfidf_search(q, top_n=24)
            return {"hits": [{"name": h, "sim": s} for h, s in hits], "note": note}
    hits = db.tfidf_search(q, top_n=24)
    return {"hits": [{"name": h, "sim": s} for h, s in hits], "note": "keyword (TF-IDF)"}


# ---------------------------------------------------------------------------
# Synergy
# ---------------------------------------------------------------------------
@app.get("/api/synergy")
def synergy(center: str, n: int = 10) -> dict:
    df = TABLES["jokers"]
    row = df[df["name"] == center]
    if row.empty:
        raise HTTPException(404, "unknown joker")
    ctr = row.iloc[0]
    ctags = set(str(ctr["tags"]).split("|")) - {"", "nan"}
    scores = []
    for _, r in df.iterrows():
        if r["name"] == center:
            continue
        shared = ctags & (set(str(r["tags"]).split("|")) - {"", "nan"})
        if shared:
            scores.append({"name": r["name"], "rarity": r["rarity"],
                           "shared": sorted(shared), "w": len(shared)})
    scores.sort(key=lambda s: (-s["w"], s["name"]))
    return {"center": {"name": ctr["name"], "rarity": ctr["rarity"],
                       "effect": ctr["effect"], "tags": sorted(ctags)},
            "neighbors": scores[:max(1, min(n, 20))]}


# ---------------------------------------------------------------------------
# Run log
# ---------------------------------------------------------------------------
class RunReq(BaseModel):
    ante: int
    deck: str = "Red"
    stake: str = "White"
    lineup: list[str] = []
    best_hand: Optional[str] = None
    best_score: Optional[int] = None
    outcome: str = "lost"
    notes: str = ""


@app.get("/api/runs")
def runs() -> dict:
    if not _lakebase_ok():
        return {"ok": False, "runs": [], "reason": DIAG.get("lakebase", "")}
    rows = db.list_runs(200)
    for r in rows:
        r["ts"] = r["ts"].isoformat() if hasattr(r["ts"], "isoformat") else str(r["ts"])
        if isinstance(r.get("lineup"), str):
            try:
                r["lineup"] = json.loads(r["lineup"])
            except Exception:
                pass
    return {"ok": True, "runs": rows}


@app.post("/api/runs")
def log_run(req: RunReq) -> dict:
    if not _lakebase_ok():
        raise HTTPException(503, "Lakebase unavailable — run not persisted")
    db.save_run(req.ante, req.deck, req.stake, req.lineup,
                req.best_hand, req.best_score, req.outcome, req.notes)
    return {"ok": True}


@app.delete("/api/runs/{run_id}")
def del_run(run_id: int) -> dict:
    if not _lakebase_ok():
        raise HTTPException(503, "Lakebase unavailable")
    db.delete_run(run_id)
    return {"ok": True}


# ---------------------------------------------------------------------------
# AI strategist
# ---------------------------------------------------------------------------
class ChatReq(BaseModel):
    question: str
    ante: int = 4
    money: int = 12
    hands_left: int = 3
    discards_left: int = 2
    shop: str = ""
    lineup: list[LineupItem] = []
    last_plays: list[dict] = []
    hand_text: str = ""
    blind_req: int = 0


@app.post("/api/chat")
def chat(req: ChatReq) -> dict:
    df = TABLES["jokers"]
    lines = ["You are a Balatro strategy coach and theorycrafting partner. Be concrete "
             "and terse; think in builds, synergies, and expected value. NEVER recompute "
             "or invent scores — a deterministic engine already did the math below; treat "
             "it as ground truth. When the player proposes a theory, take it seriously: "
             "state what would confirm or refute it, use the engine output and run "
             "history as evidence, and if evidence is missing, design the concrete "
             "experiment (exact cards, jokers, order) they should run. Avoid spoiling "
             "unlock conditions or secret content — coach the strategy, not the "
             "checklist. If information is missing, say what you'd need."]
    lines.append(f"\n## Run context\nAnte {req.ante}, ${req.money}, "
                 f"{req.hands_left} hands left, {req.discards_left} discards left.")
    if req.shop:
        lines.append(f"Shop: {req.shop}")
    if req.lineup:
        lines.append("\n## Joker lineup (left to right)")
        for item in req.lineup:
            row = df[df["name"] == item.name]
            if row.empty:
                continue
            r = row.iloc[0]
            v = f" [current value: {item.value}]" if item.value else ""
            strat = r.get("strategy", "")
            lines.append(f"- {item.name} ({r['rarity']}, {r.get('archetype','')}){v}: "
                         f"{r['effect']}"
                         + (f" | playbook: {strat}" if isinstance(strat, str) and strat else ""))
    if req.last_plays:
        lines.append(f"\n## Deterministic engine output (hand: {req.hand_text}; "
                     f"blind requires {req.blind_req or 'unknown'} chips)")
        for i, p in enumerate(req.last_plays[:3], 1):
            lines.append(f"\n### Option {i}: {p.get('hand')} — play "
                         f"{' '.join(p.get('played', []))} → {p.get('total', 0):,} chips")
            lines.append("\n".join(p.get("steps", [])))
    if _lakebase_ok():
        try:
            hist = db.list_runs(10)
            if hist:
                lines.append("\n## My recent run history (from Lakebase)")
                for r in hist:
                    lu = r.get("lineup") or []
                    if isinstance(lu, str):
                        try:
                            lu = json.loads(lu)
                        except Exception:
                            lu = []
                    lines.append(f"- {r['outcome']} at ante {r['ante']} ({r['deck']} deck, "
                                 f"{r['stake']} stake) — jokers: {', '.join(lu) or 'n/a'}"
                                 + (f" — notes: {r['notes']}" if r.get("notes") else ""))
                lines.append("Use this history: call out patterns in what keeps "
                             "killing me or carrying me.")
        except Exception:
            pass
    lines.append(f"\n## Question\n{req.question}")
    prompt = "\n".join(lines)
    try:
        answer = db.chat(prompt)
        return {"ok": True, "answer": answer, "prompt": prompt}
    except Exception as e:
        fallback = ""
        if req.last_plays:
            p = req.last_plays[0]
            fallback = (f"Play {' '.join(p.get('played', []))} for {p.get('total', 0):,} "
                        f"({p.get('hand')}). For shopping: favor jokers sharing tags with "
                        "your lineup — see the Synergy web. ×Mult stacks multiplicatively, "
                        "so a second ×Mult usually beats a third +Mult.")
        return {"ok": False, "answer": fallback,
                "error": str(e)[:200], "prompt": prompt}


@app.get("/api/diag")
def diag() -> JSONResponse:
    return JSONResponse(DIAG)
