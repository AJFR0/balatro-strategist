"""
Balatro Strategist — FastAPI backend.

The deterministic engine (engine.py) and the Lakebase/Model-Serving layer
(db.py) are unchanged; this file exposes them as JSON APIs and serves the
hand-built SPA in static/index.html. Runs on Databricks Apps.
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
import urllib.parse
import uuid
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
CARD_SETS = ["tarots", "planets", "spectrals", "vouchers", "tags", "decks",
             "blinds", "enhancements", "editions", "seals", "stakes"]
IMAGES: dict[str, Any] = {"base": "https://balatrowiki.org/images/", "files": set(),
                          "credit": "Card art © LocalThunk · data via balatrowiki.org (CC BY-NC-SA 3.0)"}
NOTES: dict[str, list[dict]] = {}


def _load_tables() -> None:
    for name in ["jokers", "hands", "joker_benchmarks", "joker_notes"] + CARD_SETS:
        path = os.path.join(DATA_DIR, f"{name}.csv")
        if os.path.exists(path):
            TABLES[name] = pd.read_csv(path)
    imgs = os.path.join(DATA_DIR, "wiki_images.json")
    if os.path.exists(imgs):
        with open(imgs) as f:
            j = json.load(f)
        IMAGES.update({"base": j["base"], "files": set(j["files"]), "credit": j["credit"]})
    if "joker_notes" in TABLES:
        NOTES.clear()
        order = {"Overview": 0, "Synergies": 1, "Strategy": 2, "Anti-Synergies": 3, "Notes": 4}
        df = TABLES["joker_notes"].fillna("")
        df = df.assign(_o=df["section"].map(lambda s: order.get(s, 9)))
        for r in df.sort_values(["name", "_o", "seq"]).to_dict("records"):
            NOTES.setdefault(r["name"], []).append({"section": r["section"], "text": r["text"]})


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
    DIAG["tracing"] = db.init_tracing()["why"]
    db.build_tfidf(TABLES["jokers"], TABLES.get("joker_notes"))
    print("=== BALATRO STRATEGIST STARTUP DIAGNOSTICS ===", flush=True)
    for k, v in DIAG.items():
        print(f"  {k}: {v}", flush=True)
    _start_backfill()
    print("=== END DIAGNOSTICS ===", flush=True)


def _start_backfill() -> None:
    if str(DIAG.get("lakebase", "")).startswith("OK") \
            and str(DIAG.get("embeddings", "")).startswith("OK"):
        def backfill() -> None:
            r = db.ensure_embeddings(TABLES["jokers"])
            print(f"  joker_embeddings backfill: {r['stored']}/{r['total']} stored"
                  + (f" ({r['error']})" if r.get("error") else " — complete"), flush=True)
            try:
                sr = db.ensure_search(TABLES["jokers"], TABLES.get("joker_notes"))
                DIAG["search"] = ("Lakebase Search (BM25 + ANN, RRF)" if sr["lbsearch"]
                                  else f"pgvector ({sr.get('error')})")
                print(f"  lakebase search: {DIAG['search']}", flush=True)
            except Exception as e:
                DIAG["search"] = f"pgvector (search index error: {str(e)[:100]})"
        threading.Thread(target=backfill, daemon=True).start()
        print("  joker_embeddings: backfill started in background", flush=True)


def _pg_ok() -> bool:
    """A real Lakebase Postgres connection is up (pgvector / semantic search)."""
    return str(DIAG.get("lakebase", "")).startswith("OK")


def _lakebase_ok() -> bool:
    """The run log is usable. In demo/hybrid mode runs live locally
    (SQLite or DynamoDB), so they never depend on Lakebase being awake."""
    return db.DEMO or _pg_ok()


# --- Lakebase wake: a Free Edition branch archives when idle and only a real
# connection un-archives it. This is the "psql to wake it" button. ------------
_WAKE: dict = {"running": False, "result": None}


def _wake_worker() -> None:
    try:
        r = db.wake()
        _WAKE["result"] = r
        if r.get("ok"):
            DIAG.update(db.diagnostics(run_chat_test=False))
            _start_backfill()
    except Exception as e:
        _WAKE["result"] = {"ok": False, "error": str(e)[:200]}
    finally:
        _WAKE["running"] = False


@app.post("/api/lakebase/reconnect")
def lakebase_reconnect() -> dict:
    if db.DEMO and not db.CONNECTED:
        return {"ok": False, "reason": "demo mode — no Lakebase"}
    if not _WAKE["running"]:
        _WAKE["running"] = True
        _WAKE["result"] = None
        threading.Thread(target=_wake_worker, daemon=True).start()
    return {"started": True, "running": _WAKE["running"]}


@app.get("/api/lakebase/reconnect")
def lakebase_reconnect_status() -> dict:
    return {"running": _WAKE["running"], "result": _WAKE["result"],
            "lakebase": DIAG.get("lakebase"), "pgvector": DIAG.get("pgvector")}


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


# --- card art: same-origin proxy for the wiki thumbnails ---------------------
# Only file names that appear in data/wiki_images.json are served (no open
# proxy). Bytes are cached on local disk (Lambda: /tmp) and marked immutable,
# so CloudFront / the service worker keep them after the first fetch.
_IMG_DIR = os.path.join(tempfile.gettempdir(), "bs-img")
_IMG_LOCK = threading.Lock()


def _fetch_card_art(fname: str) -> Optional[bytes]:
    import urllib.request
    url = IMAGES["base"] + urllib.parse.quote(fname)
    req = urllib.request.Request(url, headers={
        "User-Agent": "BalatroStrategist/1.7 (+https://github.com/AJFR0/balatro-strategist)"})
    try:
        with urllib.request.urlopen(req, timeout=8) as r:
            data = r.read()
            return data if r.status == 200 and len(data) > 0 else None
    except Exception:
        return None


@app.get("/img/{fname}")
def card_art(fname: str):
    if fname not in IMAGES["files"] or "/" in fname or fname.startswith("."):
        raise HTTPException(404, "unknown image")
    path = os.path.join(_IMG_DIR, fname)
    if not os.path.exists(path):
        data = _fetch_card_art(fname)
        if not data:
            raise HTTPException(502, "card art unavailable")
        with _IMG_LOCK:
            os.makedirs(_IMG_DIR, exist_ok=True)
            tmp = path + ".part"
            with open(tmp, "wb") as f:
                f.write(data)
            os.replace(tmp, path)
    ext = fname.rsplit(".", 1)[-1].lower()
    mt = {"webp": "image/webp", "gif": "image/gif", "jpg": "image/jpeg",
          "jpeg": "image/jpeg"}.get(ext, "image/png")
    return FileResponse(path, media_type=mt, headers={
        "Cache-Control": "public, max-age=2592000, immutable",
        "X-Credit": "Card art (c) LocalThunk, via balatrowiki.org"})


@app.get("/api/joker/{name}/notes")
def joker_notes(name: str) -> dict:
    """The wiki's Synergies / Anti-Synergies / Strategy prose for one joker
    (balatrowiki.org, CC BY-NC-SA 3.0) — fetched lazily by the Codex."""
    key = name if name in NOTES else _canon_joker(name)
    return {"name": key, "notes": NOTES.get(key, []),
            "credit": "Text from balatrowiki.org, CC BY-NC-SA 3.0",
            "source": f"https://balatrowiki.org/w/{urllib.parse.quote(key.replace(' ', '_'))}"}


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
        "cards": {k: TABLES[k].fillna("").to_dict(orient="records")
                  for k in CARD_SETS if k in TABLES},
        "credit": IMAGES["credit"],
        "diag": DIAG,
        "demo": db.DEMO,
        "genie_ok": db.genie_ok(),
        "genie_agent_ok": db.genie_ok(),
        "ai_ok": db.ai_ok(),
        "chat_model": db.chat_endpoint() if db.ai_ok() else None,
        "search_mode": DIAG.get("search", ""),
        "lakebase_ok": _lakebase_ok(),
        "semantic_ok": _pg_ok() and str(DIAG.get("embeddings", "")).startswith("OK"),
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
        row = {
            "played": [c.label() for c in p["played"]],
            "held": [c.label() for c in p["held"]],
            "hand": r.hand, "level": r.level, "chips": r.chips, "mult": r.mult,
            "total": r.total, "steps": r.steps, "unknown": r.unknown_jokers,
            "mode": r.mode, "random_sources": r.random_sources,
        }
        if r.random_sources:
            # the same play scored with every chance effect missing / hitting:
            # an honest floor and ceiling around the expectation
            lo = score_hand(p["played"], p["held"], jokers, req.levels,
                            Rules(pessimist=True), extra)
            hi = score_hand(p["played"], p["held"], jokers, req.levels,
                            Rules(optimist=True), extra)
            row["floor"], row["ceiling"] = lo.total, hi.total
        out.append(row)
    best = out[0]
    verdict = ""
    if req.blind_req:
        if best["mode"] == "deterministic":
            verdict = "ok" if best["total"] >= req.blind_req \
                else f"short by {req.blind_req - best['total']:,}"
        else:
            if best.get("floor", 0) >= req.blind_req:
                verdict = "ok"                       # clears even if every roll misses
            elif best["total"] >= req.blind_req:
                verdict = "expected"                 # expected score clears; floor does not
            elif best.get("ceiling", 0) >= req.blind_req:
                verdict = "lucky"                    # only clears if the rolls hit
            else:
                verdict = f"short by {req.blind_req - best['total']:,}"
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
    if req.semantic and _pg_ok():
        try:
            if db._state.get("lbsearch"):
                hits = db.hybrid_search(q, top_n=24)
                return {"hits": [{"name": h[0], "sim": h[1], "vrank": h[2], "krank": h[3]} for h in hits],
                        "note": "hybrid · Lakebase Search (BM25 + vector, RRF)"}
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


class DecisionReq(BaseModel):
    ante: int = 0
    blind: str = ""
    hand: str = ""
    lineup: list[str] = []
    target: int = 0
    scored_before: int = 0
    options: list[dict] = []        # [{hand, played, total, mode}] — top plays on the table
    recommended: Optional[dict] = None   # {hand, played, total, mode}
    chosen: Optional[dict] = None        # {hand, played, total, mode}
    kind: str = "play"              # play | discard
    note: str = ""


@app.post("/api/decisions")
def log_decision(req: DecisionReq) -> dict:
    if not _lakebase_ok():
        raise HTTPException(503, "run log unavailable — decision not persisted")
    doc = req.model_dump() if hasattr(req, "model_dump") else req.dict()
    db.save_decision(req.ante, req.blind, doc)
    return {"ok": True}


@app.get("/api/decisions")
def decisions(limit: int = 40) -> dict:
    if not _lakebase_ok():
        return {"ok": False, "decisions": []}
    return {"ok": True, "decisions": db.list_decisions(max(1, min(200, limit)))}


def _decision_lines(limit: int = 12) -> list[str]:
    """Recorded decision moments for the coach's post-mortem, most recent first."""
    try:
        rows = db.list_decisions(limit)
    except Exception:
        return []
    out = []
    for r in rows:
        d = r.get("doc") or {}
        rec, ch = d.get("recommended") or {}, d.get("chosen") or {}
        same = rec.get("played") == ch.get("played")
        line = (f"- {str(r.get('ts',''))[:16]} ante {r.get('ante')} {r.get('blind','')}: "
                f"hand {d.get('hand','?')} | lineup {', '.join(d.get('lineup') or []) or 'none'} | "
                f"engine #1 {rec.get('hand','?')} {' '.join(rec.get('played') or [])} = {rec.get('total',0):,}"
                f"{' ('+rec.get('mode')+')' if rec.get('mode') and rec.get('mode')!='deterministic' else ''}"
                f" | played {'the same' if same else ch.get('hand','?')+' '+' '.join(ch.get('played') or [])+' = '+format(ch.get('total',0),',')}"
                f" | target {d.get('target',0):,}, scored before this hand {d.get('scored_before',0):,}")
        if d.get("note"):
            line += f" | note: {d['note']}"
        out.append(line)
    return out


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
    boss: str = ""          # current boss blind name (run mode), e.g. "The Wall"
    deck: str = ""
    stake: str = ""
    continue_from: str = ""  # a truncated previous answer to carry on from


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
    {"type": "function", "function": {
        "name": "card_lookup",
        "description": "Look up any Balatro card by name — joker, tarot, planet, "
                       "spectral, voucher, tag, deck, boss blind, seal, edition, "
                       "enhancement — and get its verified effect, cost, rarity, "
                       "unlock and (for jokers) the community wiki's Synergies / "
                       "Anti-Synergies / Strategy notes.",
        "parameters": {"type": "object", "properties": {
            "name": {"type": "string", "description": "card name, e.g. 'Blueprint' or 'The Wall'"}},
            "required": ["name"]}}},
]


def card_lookup(name: str) -> dict:
    """Exact-then-fuzzy lookup across every card table (also used by MCP)."""
    q = str(name or "").strip().lower()
    if not q:
        return {"error": "empty name"}
    tables = ["jokers"] + CARD_SETS
    best = None
    for t in tables:
        df = TABLES.get(t)
        if df is None:
            continue
        for r in df.fillna("").to_dict("records"):
            n = str(r["name"]).lower()
            score = 3 if n == q else 2 if n.startswith(q) or q.startswith(n) else 1 if q in n or n in q else 0
            if score and (best is None or score > best[0]):
                best = (score, t, r)
    if not best:
        return {"error": f"no card named '{name}'"}
    _, t, r = best
    out = {"table": t, **{k: v for k, v in r.items()
                          if k not in ("image", "wiki_number", "order", "test_idea")}}
    if t == "jokers":
        notes = NOTES.get(r["name"], [])
        out["wiki_notes"] = [f"{n['section']}: {n['text']}" for n in notes][:12]
        out["wiki_credit"] = "balatrowiki.org, CC BY-NC-SA 3.0"
    return out


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
    if name == "card_lookup":
        return card_lookup(str(args.get("name", "")))
    raise ValueError(f"unknown tool {name}")


_TOOL_NOTE = ("\n\n## Tools\nYou can call the deterministic engine directly: "
              "best_plays(hand), score_play(cards), discard_advisor(hand). "
              "Use them to verify any line you recommend instead of guessing "
              "numbers; then answer with the results. card_lookup(name) returns "
              "the wiki-verified text of any card plus community synergy notes — "
              "use it before asserting what a card does.")


_TRUNC: dict[str, bool] = {}     # last call's finish_reason == "length", per thread-less app


def _agentic_chat(prompt: str, req: "ChatReq", endpoint: str | None = None):
    msgs = [{"role": "user", "content": prompt + _TOOL_NOTE}]
    if req.continue_from:
        msgs.append({"role": "assistant", "content": req.continue_from})
        msgs.append({"role": "user", "content": "Your previous answer was cut off. Continue "
                     "exactly where it stopped — do not repeat anything already written."})
    trace: list[dict] = []
    model = endpoint
    _TRUNC["last"] = False
    with db.span("coach", {"question": req.question[:300]}, kind="AGENT") as root:
        for _ in range(3):
            with db.span("llm", {"messages": len(msgs)}, kind="LLM") as ls:
                m = db.chat_with_tools(msgs, _COACH_TOOLS, endpoint=endpoint, max_tokens=1400)
                model = m.get("_endpoint", model)
                ls.out({"endpoint": model, "tool_calls": len(m.get("tool_calls") or [])})
            calls = m.get("tool_calls") or []
            if not calls:
                ans = (m.get("content") or "").strip()
                _TRUNC["last"] = m.get("_finish") == "length"
                root.out({"answer": ans[:500], "tools": len(trace), "model": model})
                return ans, trace, model
            msgs.append({"role": "assistant", "content": m.get("content") or "",
                         "tool_calls": calls})
            for tc in calls[:4]:
                fn = tc.get("function", {}) or {}
                name = fn.get("name", "")
                try:
                    targs = json.loads(fn.get("arguments") or "{}")
                except Exception:
                    targs = {}
                with db.span(name or "tool", targs) as ts:
                    try:
                        result = _coach_tool_exec(name, targs, req)
                    except Exception as e:
                        result = {"error": str(e)[:200]}
                    ts.out(result)
                trace.append({"tool": name, "args": targs, "result": result})
                msgs.append({"role": "tool", "tool_call_id": tc.get("id", ""),
                             "content": json.dumps(result)[:4000]})
        msgs.append({"role": "user",
                     "content": "No more tool calls — answer now with what you have."})
        m = db.chat_with_tools(msgs, [], endpoint=endpoint, max_tokens=1400)
        ans = (m.get("content") or "").strip()
        _TRUNC["last"] = m.get("_finish") == "length"
        root.out({"answer": ans[:500], "tools": len(trace), "model": m.get("_endpoint", model)})
        return ans, trace, m.get("_endpoint", model)


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
             "checklist. If information is missing, say what you'd need.\n"
             "FORMAT: the player already sees a summary card with the engine's best play, "
             "its score, the target and whether it clears — do NOT repeat that arithmetic "
             "and do NOT produce tables of scores. Answer in exactly three short blocks, "
             "each under 90 words, with these bold headings: **Why** (why this line, or why "
             "a different one — if you recommend anything other than the engine's #1 play, "
             "say so explicitly and give its purpose and the assumption it rests on), "
             "**Main risk** (what goes wrong and how likely), **Next upgrade** (the single "
             "most valuable shop/lineup change). Lead with the decision; if the best play "
             "falls short of the target, say that first."]
    lines.append(f"\n## Run context\nAnte {req.ante}, ${req.money}, "
                 f"{req.hands_left} hands left, {req.discards_left} discards left"
                 + (f", {req.deck} deck" if req.deck else "")
                 + (f", {req.stake} stake" if req.stake else "") + ".")
    if req.shop:
        lines.append(f"Shop: {req.shop}")
    if req.boss and "blinds" in TABLES:
        b = TABLES["blinds"][TABLES["blinds"]["name"] == req.boss]
        if not b.empty:
            br = b.iloc[0]
            lines.append(f"Boss blind: {req.boss} — {br['effect']} "
                         f"(x{br['score_mult']} base score, ${br['reward']} reward).")
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
    if _lakebase_ok():
        dl = _decision_lines(12)
        if dl:
            lines.append("\n## Recorded decisions (what was on the table, what the engine "
                         "recommended, what I actually played)")
            lines.extend(dl)
            lines.append("For post-mortems, cite these moments specifically (date, ante, "
                         "hand) and only draw a lesson the records support — e.g. a "
                         "repeated choice of a lower-scoring play, an expected score that "
                         "missed the target, a lineup order that cost mult. Do not infer "
                         "the whole run from its final lineup.")
    lines.append(f"\n## Question\n{req.question}")
    prompt = "\n".join(lines)
    if db.ai_ok():
        try:
            answer, trace, model = _agentic_chat(prompt, req)
            if answer:
                return {"ok": True, "answer": answer, "prompt": prompt,
                        "tool_trace": trace, "model": model,
                        "truncated": bool(_TRUNC.get("last"))}
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


class BenchReq(BaseModel):
    endpoints: list[str] = []
    question: str = ("My hand is AH KH 9H 5H 2C AS 3C 7D with The Tribe. Should I play "
                     "now or discard first? Verify with the engine before answering.")


@app.post("/api/model/bench")
def model_bench(req: BenchReq, request: Request) -> dict:
    """Run the agentic coach once per candidate endpoint and report which
    ones call tools correctly, how fast, and what they say. Free Edition
    only exposes Foundation Model APIs, so this is how the default gets picked."""
    if not db.ai_ok():
        raise HTTPException(503, "AI not connected")
    if db.DEMO:
        _ai_throttle(request)
    eps = req.endpoints or db.chat_endpoints()
    creq = ChatReq(question=req.question, lineup=[LineupItem(name="The Tribe")])
    out = []
    for ep in eps[:6]:
        t0 = _time.time()
        try:
            ans, trace, model = _agentic_chat(req.question, creq, endpoint=ep)
            out.append({"endpoint": ep, "ok": True, "secs": round(_time.time() - t0, 1),
                        "tools": [t["tool"] for t in trace], "answer": ans[:240]})
        except Exception as e:
            out.append({"endpoint": ep, "ok": False, "secs": round(_time.time() - t0, 1),
                        "error": str(e)[:200]})
    return {"results": out, "current": db.chat_endpoint()}


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


# --- Genie Agent mode: multi-step reasoning, run in a worker thread and polled ---
_AGENT_JOBS: dict[str, dict] = {}


class GenieAgentReq(BaseModel):
    question: str
    conversation_id: Optional[str] = None


@app.post("/api/genie/agent/start")
def genie_agent_start(req: GenieAgentReq, request: Request) -> dict:
    if not db.genie_ok():
        raise HTTPException(503, "Genie is not wired up")
    if db.DEMO:
        _ai_throttle(request)
    job = uuid.uuid4().hex[:12]
    _AGENT_JOBS[job] = {"status": "running", "started": _time.time()}

    def work() -> None:
        try:
            r = db.genie_agent_run(req.question, req.conversation_id)
            _AGENT_JOBS[job].update(r)
            _AGENT_JOBS[job]["status"] = r.get("status") or "completed"
        except Exception as e:
            _AGENT_JOBS[job].update(status="failed", error=str(e)[:300])
        # keep the table small
        for k in [k for k, v in _AGENT_JOBS.items() if _time.time() - v.get("started", 0) > 1800]:
            _AGENT_JOBS.pop(k, None)
    threading.Thread(target=work, daemon=True).start()
    return {"job": job}


@app.get("/api/genie/agent/poll")
def genie_agent_poll(job: str) -> dict:
    j = _AGENT_JOBS.get(job)
    if not j:
        raise HTTPException(410, "job not found on this instance — ask again")
    return j


@app.get("/api/diag")
def diag() -> JSONResponse:
    return JSONResponse(DIAG)


# ---------------------------------------------------------------------------
# MCP: the engine as tools for any agent (Genie One, Claude, Cursor …).
# Mounted only when the `mcp` package is installed (Databricks deployment).
# ---------------------------------------------------------------------------
try:
    from mcp.server.fastmcp import FastMCP
    from mcp.server.transport_security import TransportSecuritySettings
    # served behind the Databricks Apps auth proxy on a public hostname, so the
    # localhost-only DNS-rebinding guard has to be relaxed
    _mcp = FastMCP("balatro-strategist", stateless_http=True, streamable_http_path="/",
                   transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False))

    @_mcp.tool()
    def best_plays_tool(hand: str, jokers: list[str] = []) -> dict:
        """Score every legal play from a Balatro hand with the deterministic
        engine. hand like 'AH KH 9H 5H 2C AS 3C 7D'; jokers left-to-right."""
        req = ChatReq(question="", lineup=[LineupItem(name=j) for j in jokers])
        return _coach_tool_exec("best_plays", {"hand": hand}, req)

    @_mcp.tool()
    def score_play_tool(cards: str, jokers: list[str] = []) -> dict:
        """Score exactly these played cards (1-5) with the given joker lineup."""
        req = ChatReq(question="", lineup=[LineupItem(name=j) for j in jokers])
        return _coach_tool_exec("score_play", {"cards": cards}, req)

    @_mcp.tool()
    def discard_advisor_tool(hand: str, jokers: list[str] = [], max_discard: int = 5) -> dict:
        """Monte-Carlo discard analysis: expected best-play score after redraw."""
        req = ChatReq(question="", lineup=[LineupItem(name=j) for j in jokers])
        return _coach_tool_exec("discard_advisor", {"hand": hand, "max_discard": max_discard}, req)

    @_mcp.tool()
    def card_lookup_tool(name: str) -> dict:
        """Wiki-verified text for any Balatro card (joker, tarot, planet,
        spectral, voucher, tag, deck, boss blind, seal, edition, enhancement)
        plus community synergy notes for jokers (balatrowiki.org, CC BY-NC-SA)."""
        return card_lookup(name)

    app.mount("/mcp", _mcp.streamable_http_app())
    import contextlib
    _mcp_stack = contextlib.AsyncExitStack()

    @app.on_event("startup")
    async def _mcp_start() -> None:
        await _mcp_stack.enter_async_context(_mcp.session_manager.run())

    @app.on_event("shutdown")
    async def _mcp_stop() -> None:
        await _mcp_stack.aclose()
    DIAG["mcp"] = "mounted at /mcp (streamable HTTP)"
except Exception as _e:               # package absent (Lambda) or API drift
    DIAG["mcp"] = f"off ({str(_e)[:80]})"
