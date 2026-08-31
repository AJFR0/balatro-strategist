"""
Lakebase data layer for Balatro Strategist.

Wiring:
  - Lakebase (managed Postgres) holds the run log and the joker-effect
    embeddings. Connection auth = short-lived OAuth token minted by the app's
    service principal via the Databricks SDK (no passwords anywhere).
  - Embeddings come from the pay-per-token FMAPI embedding endpoint.
  - pgvector is used when the extension is available; otherwise embeddings
    live in JSONB and cosine similarity runs in numpy (150 rows — same answer).
  - If Lakebase is unreachable entirely, the app degrades gracefully:
    runs are kept in session state and search falls back to TF-IDF.

Everything reports its health through diagnostics(), which app.py prints to
stdout at startup so the App's Logs tab shows exactly what is wired up.
"""
from __future__ import annotations
import json
import os
import time
import uuid

import numpy as np

INSTANCE = os.environ.get("LAKEBASE_INSTANCE", "balatro-base")
DBNAME = os.environ.get("LAKEBASE_DATABASE", "databricks_postgres")
EMBED_ENDPOINT = os.environ.get("EMBED_ENDPOINT", "databricks-gte-large-en")
CHAT_ENDPOINT = os.environ.get("STRATEGIST_ENDPOINT", "databricks-meta-llama-3-3-70b-instruct")
SCHEMA = "balatro"
TOKEN_TTL_S = 45 * 60


# ---------------------------------------------------------------------------
# Connection management
# ---------------------------------------------------------------------------
_state: dict = {"conn": None, "born": 0.0, "err": None, "pgvector": False, "ready": False}


def _workspace_client():
    from databricks.sdk import WorkspaceClient
    return WorkspaceClient()


def _pg_user(w) -> str:
    cid = os.environ.get("DATABRICKS_CLIENT_ID")
    if cid:
        return cid
    return w.current_user.me().user_name


def _fresh_conn():
    import psycopg2
    w = _workspace_client()
    inst = w.database.get_database_instance(name=INSTANCE)
    cred = w.database.generate_database_credential(
        request_id=str(uuid.uuid4()), instance_names=[INSTANCE])
    conn = psycopg2.connect(
        host=inst.read_write_dns, dbname=DBNAME, user=_pg_user(w),
        password=cred.token, sslmode="require", connect_timeout=10)
    conn.autocommit = True
    return conn


def get_conn():
    """Cached connection; re-minted before the OAuth token expires."""
    c = _state["conn"]
    if c is not None and (time.time() - _state["born"]) < TOKEN_TTL_S and not c.closed:
        return c
    try:
        if c is not None:
            try: c.close()
            except Exception: pass
        _state["conn"] = _fresh_conn()
        _state["born"] = time.time()
        _state["err"] = None
    except Exception as e:
        _state["conn"] = None
        _state["err"] = str(e)
    return _state["conn"]


def _exec(sql: str, params=None, fetch: bool = False):
    conn = get_conn()
    if conn is None:
        raise RuntimeError(f"Lakebase unavailable: {_state['err']}")
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchall() if fetch else None
    except Exception:
        # one retry on a broken/expired connection
        _state["conn"] = None
        conn = get_conn()
        if conn is None:
            raise
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchall() if fetch else None


def available() -> bool:
    return get_conn() is not None


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

def init_schema() -> dict:
    """Create schema/tables; detect or enable pgvector. Safe to call often."""
    out = {"lakebase": False, "pgvector": False, "error": None}
    if not available():
        out["error"] = _state["err"]
        return out
    out["lakebase"] = True
    try:
        _exec("CREATE EXTENSION IF NOT EXISTS vector")
        _state["pgvector"] = True
    except Exception:
        # extension may already exist, or we may lack rights — probe for it
        try:
            got = _exec("SELECT 1 FROM pg_extension WHERE extname='vector'", fetch=True)
            _state["pgvector"] = bool(got)
        except Exception:
            _state["pgvector"] = False
    out["pgvector"] = _state["pgvector"]

    _exec(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}")
    _exec(f"""
        CREATE TABLE IF NOT EXISTS {SCHEMA}.runs (
            id          BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            ts          TIMESTAMPTZ NOT NULL DEFAULT now(),
            ante        INT,
            deck        TEXT,
            stake       TEXT,
            lineup      JSONB,
            best_hand   TEXT,
            best_score  BIGINT,
            outcome     TEXT,           -- won | lost | in-progress
            notes       TEXT
        )""")
    if _state["pgvector"]:
        _exec(f"""
            CREATE TABLE IF NOT EXISTS {SCHEMA}.joker_embeddings (
                name  TEXT PRIMARY KEY,
                model TEXT NOT NULL,
                emb   vector(1024)
            )""")
    else:
        _exec(f"""
            CREATE TABLE IF NOT EXISTS {SCHEMA}.joker_embeddings_json (
                name  TEXT PRIMARY KEY,
                model TEXT NOT NULL,
                emb   JSONB NOT NULL
            )""")
    _state["ready"] = True
    return out


# ---------------------------------------------------------------------------
# Run log
# ---------------------------------------------------------------------------

def save_run(ante, deck, stake, lineup, best_hand, best_score, outcome, notes) -> bool:
    _exec(f"""INSERT INTO {SCHEMA}.runs
              (ante, deck, stake, lineup, best_hand, best_score, outcome, notes)
              VALUES (%s,%s,%s,%s,%s,%s,%s,%s)""",
          (ante, deck, stake, json.dumps(lineup), best_hand, best_score, outcome, notes))
    return True


def list_runs(limit: int = 100):
    rows = _exec(f"""SELECT id, ts, ante, deck, stake, lineup, best_hand,
                            best_score, outcome, notes
                     FROM {SCHEMA}.runs ORDER BY ts DESC LIMIT %s""",
                 (limit,), fetch=True)
    cols = ["id", "ts", "ante", "deck", "stake", "lineup", "best_hand",
            "best_score", "outcome", "notes"]
    return [dict(zip(cols, r)) for r in rows]


def run_stats():
    rows = _exec(f"""SELECT outcome, count(*), coalesce(avg(ante),0),
                            coalesce(max(best_score),0)
                     FROM {SCHEMA}.runs GROUP BY outcome""", fetch=True)
    return {r[0]: {"count": int(r[1]), "avg_ante": float(r[2]), "top_score": int(r[3])}
            for r in rows}


def delete_run(run_id: int):
    _exec(f"DELETE FROM {SCHEMA}.runs WHERE id = %s", (run_id,))


# ---------------------------------------------------------------------------
# Embeddings + semantic search
# ---------------------------------------------------------------------------

def _embed(texts: list[str]) -> list[list[float]]:
    w = _workspace_client()
    out = []
    for i in range(0, len(texts), 8):
        resp = w.serving_endpoints.query(name=EMBED_ENDPOINT, input=texts[i:i + 8])
        out.extend([d.embedding for d in resp.data])
    return out


def _emb_table() -> str:
    return f"{SCHEMA}.joker_embeddings" if _state["pgvector"] else f"{SCHEMA}.joker_embeddings_json"


def embedding_count() -> int:
    try:
        return int(_exec(f"SELECT count(*) FROM {_emb_table()}", fetch=True)[0][0])
    except Exception:
        return 0


def ensure_embeddings(jokers_df, budget_s: float = 300.0) -> dict:
    """Embed joker effect texts into Lakebase, incrementally.

    Works in small batches and commits each one, so progress survives
    timeouts and restarts — repeated boots converge on 150/150.
    """
    out = {"stored": 0, "total": len(jokers_df), "error": None}
    table = _emb_table()
    t0 = time.time()
    try:
        have = {r[0] for r in _exec(f"SELECT name FROM {table}", fetch=True)}
    except Exception as e:
        out["error"] = f"table read failed: {e}"
        return out
    missing = jokers_df[~jokers_df["name"].isin(have)]
    out["stored"] = len(have)
    if len(missing) == 0:
        return out
    rows = list(missing.iterrows())
    for i in range(0, len(rows), 8):
        if time.time() - t0 > budget_s:
            out["error"] = f"time budget hit at {out['stored']}/{out['total']} — will resume"
            return out
        chunk = rows[i:i + 8]
        texts = [f"{r['name']}. {r['effect']} Tags: {str(r['tags']).replace('|', ', ')}"
                 for _, r in chunk]
        try:
            embs = _embed(texts)
        except Exception as e:
            out["error"] = f"embedding batch failed at {out['stored']}/{out['total']}: {str(e)[:120]}"
            return out
        for (_, r), e in zip(chunk, embs):
            payload = str(list(e)) if _state["pgvector"] else json.dumps(list(e))
            _exec(f"""INSERT INTO {table} (name, model, emb) VALUES (%s,%s,%s)
                      ON CONFLICT (name) DO UPDATE SET emb = EXCLUDED.emb""",
                  (r["name"], EMBED_ENDPOINT, payload))
        out["stored"] += len(chunk)
    return out


def semantic_search(query: str, top_n: int = 12):
    """Return [(name, similarity)] — pgvector when present, numpy otherwise."""
    q = _embed([query])[0]
    if _state["pgvector"]:
        rows = _exec(
            f"""SELECT name, 1 - (emb <=> %s::vector) AS sim
                FROM {SCHEMA}.joker_embeddings ORDER BY emb <=> %s::vector LIMIT %s""",
            (str(list(q)), str(list(q)), top_n), fetch=True)
        return [(r[0], float(r[1])) for r in rows]
    rows = _exec(f"SELECT name, emb FROM {SCHEMA}.joker_embeddings_json", fetch=True)
    names = [r[0] for r in rows]
    M = np.array([json.loads(r[1]) if isinstance(r[1], str) else r[1] for r in rows], dtype=float)
    qv = np.array(q, dtype=float)
    sims = (M @ qv) / (np.linalg.norm(M, axis=1) * np.linalg.norm(qv) + 1e-9)
    order = np.argsort(-sims)[:top_n]
    return [(names[i], float(sims[i])) for i in order]


# ---------------------------------------------------------------------------
# TF-IDF fallback search (no network, no database — always works)
# ---------------------------------------------------------------------------
_tfidf: dict = {}


def _tokenize(s: str):
    import re
    return re.findall(r"[a-z0-9+×$]+", s.lower())


def build_tfidf(jokers_df):
    docs = [(r["name"], _tokenize(f"{r['name']} {r['effect']} {str(r['tags']).replace('|',' ')} "
                                  f"{r.get('archetype','')} {r.get('strategy','')}"))
            for _, r in jokers_df.iterrows()]
    df_count: dict = {}
    for _, toks in docs:
        for t in set(toks):
            df_count[t] = df_count.get(t, 0) + 1
    n = len(docs)
    idf = {t: np.log(n / (1 + c)) + 1 for t, c in df_count.items()}
    vecs = {}
    for name, toks in docs:
        v: dict = {}
        for t in toks:
            v[t] = v.get(t, 0) + 1
        vecs[name] = {t: c * idf[t] for t, c in v.items()}
    _tfidf.update({"idf": idf, "vecs": vecs})


def tfidf_search(query: str, top_n: int = 12):
    if not _tfidf:
        return []
    qt = _tokenize(query)
    qv = {}
    for t in qt:
        if t in _tfidf["idf"]:
            qv[t] = qv.get(t, 0) + _tfidf["idf"][t]
    if not qv:
        return []
    qn = np.sqrt(sum(x * x for x in qv.values()))
    scored = []
    for name, dv in _tfidf["vecs"].items():
        dot = sum(qv[t] * dv.get(t, 0) for t in qv)
        dn = np.sqrt(sum(x * x for x in dv.values()))
        if dot > 0:
            scored.append((name, float(dot / (qn * dn + 1e-9))))
    scored.sort(key=lambda x: -x[1])
    return scored[:top_n]


# ---------------------------------------------------------------------------
# Chat (AI strategist)
# ---------------------------------------------------------------------------

def chat(prompt: str, max_tokens: int = 900, temperature: float = 0.4) -> str:
    from databricks.sdk.service.serving import ChatMessage, ChatMessageRole
    w = _workspace_client()
    resp = w.serving_endpoints.query(
        name=CHAT_ENDPOINT,
        messages=[ChatMessage(role=ChatMessageRole.USER, content=prompt)],
        max_tokens=max_tokens, temperature=temperature)
    return resp.choices[0].message.content


# ---------------------------------------------------------------------------
# Diagnostics — printed to app logs at startup
# ---------------------------------------------------------------------------

def diagnostics(run_chat_test: bool = True) -> dict:
    d: dict = {}
    s = init_schema()
    d["lakebase"] = "OK" if s["lakebase"] else f"FAIL ({s['error']})"
    d["pgvector"] = "OK" if s["pgvector"] else "unavailable (JSONB fallback)"
    try:
        e = _embed(["diagnostic ping"])
        d["embeddings"] = f"OK ({EMBED_ENDPOINT}, dim={len(e[0])})"
    except Exception as ex:
        d["embeddings"] = f"FAIL ({str(ex)[:160]})"
    if run_chat_test:
        try:
            chat("Reply with the single word: ok", max_tokens=5, temperature=0.0)
            d["chat"] = f"OK ({CHAT_ENDPOINT})"
        except Exception as ex:
            d["chat"] = f"FAIL ({str(ex)[:160]})"
    return d
