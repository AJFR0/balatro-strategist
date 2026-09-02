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
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

import db
from engine import (J as JSPEC, HAND_BASE, NAME_RANKS, SUPPORTED,
                    JokerState, Rules, best_plays, best_discards, parse_cards,
                    score_hand)

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
                 "vouchers", "decks", "tags", "joker_benchmarks"]:
        path = os.path.join(DATA_DIR, f"{name}.csv")
        if os.path.exists(path):
            TABLES[name] = pd.read_csv(path)


def _joker_records() -> list[dict]:
    df = TABLES["jokers"].fillna("")
    recs = df.to_dict(orient="records")
    bench = {}
    if "joker_benchmarks" in TABLES:
        bench = {b["name"]: b for b in
                 TABLES["joker_benchmarks"].fillna("").to_dict(orient="records")}
    for r in recs:
        spec = JSPEC.get(r["name"], {})
        r["kind"] = spec.get("kind", "")
        r["stat"] = spec.get("stat", "")
        r["supported"] = r["name"] in SUPPORTED
        b = bench.get(r["name"])
        if b:
            r["flush_lift"] = b["flush_lift"]
            r["pair_lift"] = b["pair_lift"]
            r["best_context"] = b["best_context"]
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


# --- PWA assets -----------------------------------------------------------
@app.get("/manifest.json")
def manifest() -> FileResponse:
    return FileResponse(os.path.join(HERE, "static", "manifest.json"),
                        media_type="application/manifest+json")


@app.get("/sw.js")
def service_worker() -> FileResponse:
    return FileResponse(os.path.join(HERE, "static", "sw.js"),
                        media_type="application/javascript",
                        headers={"Cache-Control": "no-cache"})


@app.get("/icon-192.png")
@app.get("/apple-touch-icon.png")
def icon_192() -> FileResponse:
    return FileResponse(os.path.join(HERE, "static", "icon-192.png"),
                        media_type="image/png")


@app.get("/icon-512.png")
def icon_512() -> FileResponse:
    return FileResponse(os.path.join(HERE, "static", "icon-512.png"),
                        media_type="image/png")


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
        "genie_ok": db.genie_ok(),
        "ai_ok": db.ai_ok(),
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
# Discard advisor  (seeded Monte Carlo over the unseen deck)
# ---------------------------------------------------------------------------
class DiscardReq(BaseModel):
    hand: str
    lineup: list[LineupItem] = []
    levels: dict[str, int] = {}
    optimist: bool = False
    max_discard: int = 5


@app.post("/api/discard")
def discard(req: DiscardReq) -> dict:
    try:
        cards = parse_cards(req.hand)
    except ValueError as e:
        raise HTTPException(400, str(e))
    if not 2 <= len(cards) <= 12:
        raise HTTPException(400, "Give me 2-12 cards to advise on discards.")
    jokers = [JokerState(_canon_joker(i.name), i.value,
                         _canon_edition(i.edition)) for i in req.lineup]
    r = best_discards(cards, jokers, req.levels, Rules(optimist=req.optimist),
                      max_discard=max(1, min(5, req.max_discard)), top_n=5)
    return r


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
def del_run(run_id: str) -> dict:
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
    levels: dict[str, int] = {}
    last_plays: list[dict] = []
    hand_text: str = ""
    blind_req: int = 0


# --- agentic coach: the model can call the deterministic engine -----------
_COACH_TOOLS = [
    {"type": "function", "function": {
        "name": "best_plays",
        "description": "Run the deterministic scoring engine over every legal "
                       "play from a hand (with the player's current joker "
                       "lineup and hand levels) and return the top plays.",
        "parameters": {"type": "object", "properties": {
            "hand": {"type": "string",
                     "description": "cards like 'AH KH 9H 5H 2C AS 3C 7D'"}},
            "required": ["hand"]}}},
    {"type": "function", "function": {
        "name": "score_play",
        "description": "Score exactly these played cards (1-5) with the "
                       "player's current lineup and levels. Use to test a "
                       "specific line or compare two plays.",
        "parameters": {"type": "object", "properties": {
            "cards": {"type": "string",
                      "description": "the exact cards to play, e.g. 'AH AS'"}},
            "required": ["cards"]}}},
    {"type": "function", "function": {
        "name": "discard_advisor",
        "description": "Seeded Monte-Carlo discard analysis: expected "
                       "best-play score after redrawing, for the best discard "
                       "choices from this hand.",
        "parameters": {"type": "object", "properties": {
            "hand": {"type": "string", "description": "the full held hand"},
            "max_discard": {"type": "integer", "minimum": 1, "maximum": 5}},
            "required": ["hand"]}}},
]


def _coach_tool_exec(name: str, args: dict, req: "ChatReq") -> dict:
    jokers = [JokerState(_canon_joker(i.name), i.value,
                         _canon_edition(i.edition)) for i in req.lineup]
    levels = req.levels or {}
    if name == "best_plays":
        cards = parse_cards(str(args.get("hand", "")))
        plays = best_plays(cards, jokers, levels, top_n=3)
        out = [{"hand": p["result"].hand,
                "played": [c.label() for c in p["played"]],
                "total": p["result"].total,
                "chips": p["result"].chips, "mult": p["result"].mult}
               for p in plays]
        if plays:
            out[0]["steps"] = plays[0]["result"].steps[:14]
        return {"plays": out}
    if name == "score_play":
        cards = parse_cards(str(args.get("cards", "")))
        r = score_hand(cards, [], jokers, levels)
        return {"hand": r.hand, "total": r.total, "chips": r.chips,
                "mult": r.mult, "steps": r.steps[:14]}
    if name == "discard_advisor":
        cards = parse_cards(str(args.get("hand", "")))
        md = max(1, min(5, int(args.get("max_discard", 5) or 5)))
        r = best_discards(cards, jokers, levels, max_discard=md, top_n=3,
                          stage1_samples=4, stage2_samples=40)
        return {"stand_pat": r["stand_pat"],
                "options": [{"discard": o["discard"], "ev": round(o["ev"], 1),
                             "ci95": round(o["ci95"], 1),
                             "delta": round(o["delta"], 1)}
                            for o in r["options"]],
                "note": r["assumption"]}
    raise ValueError(f"unknown tool {name}")


_TOOL_NOTE = ("\n\n## Tools\nYou can call the deterministic engine directly: "
              "best_plays(hand), score_play(cards), discard_advisor(hand). "
              "Use them to verify any line you recommend instead of guessing "
              "numbers; then answer with the results.")


def _agentic_chat(prompt: str, req: "ChatReq"):
    msgs = [{"role": "user", "content": prompt + _TOOL_NOTE}]
    trace: list[dict] = []
    for _ in range(3):
        m = db.chat_with_tools(msgs, _COACH_TOOLS)
        calls = m.get("tool_calls") or []
        if not calls:
            return (m.get("content") or "").strip(), trace
        msgs.append({"role": "assistant", "content": m.get("content") or "",
                     "tool_calls": calls})
        for tc in calls[:4]:
            fn = tc.get("function", {}) or {}
            name = fn.get("name", "")
            try:
                targs = json.loads(fn.get("arguments") or "{}")
            except Exception:
                targs = {}
            try:
                result = _coach_tool_exec(name, targs, req)
            except Exception as e:
                result = {"error": str(e)[:200]}
            trace.append({"tool": name, "args": targs, "result": result})
            msgs.append({"role": "tool", "tool_call_id": tc.get("id", ""),
                         "content": json.dumps(result)[:4000]})
    msgs.append({"role": "user",
                 "content": "No more tool calls — answer now with what you have."})
    m = db.chat_with_tools(msgs, [])
    return (m.get("content") or "").strip(), trace


@app.post("/api/chat")
def chat(req: ChatReq, request: Request) -> dict:
    if db.ai_ok() and db.DEMO:
        _ai_throttle(request)
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
    if db.ai_ok():
        try:
            answer, trace = _agentic_chat(prompt, req)
            if answer:
                return {"ok": True, "answer": answer, "prompt": prompt,
                        "tool_trace": trace}
        except Exception:
            pass                    # endpoint may not support tools — fall back
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


# ---------------------------------------------------------------------------
# Genie — ask-the-data (async start/poll so front-door timeouts never bite)
# ---------------------------------------------------------------------------
import time as _time

_AI_BUDGET: dict[str, list] = {}          # ip -> [window_start, count]
_AI_MAX_PER_HOUR = 30


def _ai_throttle(request: Request) -> None:
    """Cheap per-IP budget so an open review site can't drain the
    owner's Databricks Free Edition quota. Per-container, best-effort."""
    ip = (request.headers.get("x-forwarded-for", "") or "?").split(",")[0].strip()
    now = _time.time()
    win = _AI_BUDGET.get(ip)
    if not win or now - win[0] > 3600:
        _AI_BUDGET[ip] = [now, 1]
        return
    win[1] += 1
    if win[1] > _AI_MAX_PER_HOUR:
        raise HTTPException(429, "AI budget for this hour is spent — try later.")


class GenieReq(BaseModel):
    question: str


@app.post("/api/genie/start")
def genie_start(req: GenieReq, request: Request) -> dict:
    if not db.genie_ok():
        return {"ok": False, "error": "Genie is not wired up on this deployment"}
    q = req.question.strip()
    if not q:
        return {"ok": False, "error": "ask something"}
    _ai_throttle(request)
    try:
        ids = db.genie_start(q)
        if not ids.get("conversation_id"):
            return {"ok": False, "error": "Genie did not accept the question"}
        return {"ok": True, **ids}
    except Exception as e:
        return {"ok": False, "error": str(e)[:200]}


@app.get("/api/genie/poll")
def genie_poll(cid: str, mid: str) -> dict:
    if not db.genie_ok():
        return {"status": "FAILED", "error": "Genie is not wired up"}
    try:
        return db.genie_poll(cid, mid)
    except Exception as e:
        return {"status": "FAILED", "error": str(e)[:200]}


@app.get("/api/diag")
def diag() -> JSONResponse:
    return JSONResponse(DIAG)
