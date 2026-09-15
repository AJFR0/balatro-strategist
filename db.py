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
# Coach model: STRATEGIST_ENDPOINT pins one; otherwise the first candidate that
# answers a tool-calling probe wins (all are Free Edition Foundation Model APIs).
CHAT_ENDPOINT = os.environ.get("STRATEGIST_ENDPOINT", "")
CHAT_CANDIDATES = [e.strip() for e in os.environ.get(
    "STRATEGIST_CANDIDATES",
    "databricks-gpt-oss-120b,databricks-qwen35-122b-a10b,"
    "databricks-llama-4-maverick,databricks-meta-llama-3-3-70b-instruct").split(",") if e.strip()]
LAKEBASE_PROJECT = os.environ.get("LAKEBASE_PROJECT", "projects/balatro-base")
LAKEBASE_BRANCH = os.environ.get("LAKEBASE_BRANCH", "production")
LAKEBASE_ENDPOINT = os.environ.get("LAKEBASE_ENDPOINT", "primary")
SCHEMA = "balatro"
TOKEN_TTL_S = 45 * 60

# Demo mode: run anywhere with zero Databricks dependencies. The engine,
# codex, synergy web and TF-IDF search are fully local already; this flag
# swaps the run log to a local SQLite file and skips every Databricks call.
DEMO = os.environ.get("DEMO_MODE", "").strip().lower() in ("1", "true", "yes")
DEMO_DB = os.environ.get(
    "DEMO_DB_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo_runs.sqlite3"))

# Hybrid mode: a demo deployment (e.g. the AWS review copy) with Databricks
# credentials supplied via the standard SDK env vars lights the AI back up —
# chat, embeddings, semantic search and Genie — while the run log stays in
# local SQLite so public visitors never write into the owner's Lakebase.
CONNECTED = bool(os.environ.get("DATABRICKS_HOST")) and bool(
    os.environ.get("DATABRICKS_TOKEN") or os.environ.get("DATABRICKS_CLIENT_ID"))
GENIE_SPACE_ID = os.environ.get("GENIE_SPACE_ID", "")


def ai_ok() -> bool:
    """AI features (chat/embeddings/Genie) are available."""
    return CONNECTED or not DEMO


def genie_ok() -> bool:
    return ai_ok() and bool(GENIE_SPACE_ID)


# ---------------------------------------------------------------------------
# Connection management
# ---------------------------------------------------------------------------
_state: dict = {"conn": None, "born": 0.0, "err": None, "pgvector": False, "ready": False,
               "lbsearch": False, "chat_ep": None, "bad_eps": set()}


def _workspace_client():
    from databricks.sdk import WorkspaceClient
    return WorkspaceClient()


def _pg_user(w) -> str:
    cid = os.environ.get("DATABRICKS_CLIENT_ID")
    if cid:
        return cid
    return w.current_user.me().user_name


def _endpoint_path() -> str:
    return f"/api/2.0/postgres/{LAKEBASE_PROJECT}/branches/{LAKEBASE_BRANCH}/endpoints/{LAKEBASE_ENDPOINT}"


def _pg_host(w) -> str:
    """Read-write host via the GA Postgres Projects API; legacy instance API as fallback."""
    try:
        ep = w.api_client.do("GET", _endpoint_path())
        host = (((ep or {}).get("status") or {}).get("hosts") or {}).get("host")
        if host:
            return host
    except Exception:
        pass
    return w.database.get_database_instance(name=INSTANCE).read_write_dns


def enable_endpoint() -> dict:
    """Re-enable a Lakebase endpoint the platform disabled after long idling
    (the app's own service principal does this — no human token involved)."""
    w = _workspace_client()
    r = w.api_client.do("PATCH", _endpoint_path(), query={"update_mask": "spec.disabled"},
                        body={"spec": {"disabled": False}})
    return {"ok": True, "op": str((r or {}).get("name", ""))[:80]}


def endpoint_status() -> dict:
    try:
        st = (_workspace_client().api_client.do("GET", _endpoint_path()) or {}).get("status") or {}
        return {"state": st.get("current_state"), "disabled": st.get("disabled")}
    except Exception as e:
        return {"error": str(e)[:120]}


def _fresh_conn(timeout: int = 10):
    import psycopg2
    w = _workspace_client()
    cred = w.database.generate_database_credential(
        request_id=str(uuid.uuid4()), instance_names=[INSTANCE])
    conn = psycopg2.connect(
        host=_pg_host(w), dbname=DBNAME, user=_pg_user(w),
        password=cred.token, sslmode="require", connect_timeout=timeout)
    conn.autocommit = True
    return conn


def wake(attempts: int = 6, wait: float = 15.0, timeout: int = 30) -> dict:
    """Keep connecting until an archived / suspended Lakebase branch comes
    back (the platform un-archives on the first real connection). Uses the
    app's own platform-minted credential — no tokens leave the service."""
    if DEMO and not CONNECTED:
        return {"ok": False, "error": "demo mode"}
    last = None
    for i in range(attempts):
        try:
            c = _fresh_conn(timeout)
            old = _state["conn"]
            _state.update(conn=c, born=time.time(), err=None)
            if old is not None:
                try: old.close()
                except Exception: pass
            return {"ok": True, "attempts": i + 1}
        except Exception as e:
            last = str(e)[:200]
            _state["err"] = last
            if "has been disabled" in last.lower():
                try:
                    enable_endpoint()          # platform disabled it after idling
                    last += " -> re-enabled endpoint, retrying"
                except Exception as ee:
                    last += f" -> enable failed: {str(ee)[:80]}"
                _state["err"] = last
            if i < attempts - 1:
                time.sleep(wait)
    return {"ok": False, "attempts": attempts, "error": last}


def get_conn():
    """Cached connection; re-minted before the OAuth token expires."""
    if DEMO and not CONNECTED:
        return None
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
# Run log  (Lakebase Postgres normally; DynamoDB on Lambda; SQLite fallback)
# ---------------------------------------------------------------------------
RUNS_TABLE = os.environ.get("RUNS_TABLE", "balatro-runs")
_ddb_state: dict = {"table": None, "ok": None}


def _dynamo():
    """DynamoDB table handle, or None when unavailable. Auto-detected on
    Lambda (boto3 is built in there); never used outside demo mode."""
    if _ddb_state["ok"] is False:
        return None
    if _ddb_state["table"] is None:
        try:
            import boto3
            t = boto3.resource("dynamodb").Table(RUNS_TABLE)
            t.load()                       # raises if missing/unauthorized
            _ddb_state["table"] = t
            _ddb_state["ok"] = True
        except Exception:
            _ddb_state["ok"] = False
            return None
    return _ddb_state["table"]


def _use_dynamo() -> bool:
    return DEMO and bool(os.environ.get("AWS_LAMBDA_FUNCTION_NAME")) \
        and _dynamo() is not None


def _ddb_save(ante, deck, stake, lineup, best_hand, best_score, outcome, notes):
    import datetime
    import uuid as _uuid
    now = datetime.datetime.now(datetime.timezone.utc)
    _dynamo().put_item(Item={
        "pk": "runs",
        "id": now.strftime("%Y%m%dT%H%M%S") + "-" + _uuid.uuid4().hex[:6],
        "ts": now.isoformat(timespec="seconds"),
        "ante": int(ante or 0), "deck": deck or "", "stake": stake or "",
        "lineup": json.dumps(lineup or []),
        "best_hand": best_hand or "", "best_score": int(best_score or 0),
        "outcome": outcome or "", "notes": notes or ""})


def _ddb_list(limit=100):
    r = _dynamo().query(KeyConditionExpression="pk = :p",
                        ExpressionAttributeValues={":p": "runs"},
                        ScanIndexForward=False, Limit=limit)
    out = []
    for it in r.get("Items", []):
        out.append({"id": it["id"], "ts": it.get("ts", ""),
                    "ante": int(it.get("ante", 0)), "deck": it.get("deck", ""),
                    "stake": it.get("stake", ""), "lineup": it.get("lineup", "[]"),
                    "best_hand": it.get("best_hand", ""),
                    "best_score": int(it.get("best_score", 0)),
                    "outcome": it.get("outcome", ""), "notes": it.get("notes", "")})
    return out


def _sqlite():
    import sqlite3
    conn = sqlite3.connect(DEMO_DB)
    conn.execute("""CREATE TABLE IF NOT EXISTS runs (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        ts         TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
        ante       INTEGER, deck TEXT, stake TEXT, lineup TEXT,
        best_hand  TEXT, best_score INTEGER, outcome TEXT, notes TEXT)""")
    return conn


def save_run(ante, deck, stake, lineup, best_hand, best_score, outcome, notes) -> bool:
    if DEMO:
        if _use_dynamo():
            try:
                _ddb_save(ante, deck, stake, lineup, best_hand, best_score,
                          outcome, notes)
                return True
            except Exception:
                _ddb_state["ok"] = False        # fall through to SQLite
        with _sqlite() as c:
            c.execute("""INSERT INTO runs (ante, deck, stake, lineup, best_hand,
                                           best_score, outcome, notes)
                         VALUES (?,?,?,?,?,?,?,?)""",
                      (ante, deck, stake, json.dumps(lineup), best_hand,
                       best_score, outcome, notes))
        return True
    _exec(f"""INSERT INTO {SCHEMA}.runs
              (ante, deck, stake, lineup, best_hand, best_score, outcome, notes)
              VALUES (%s,%s,%s,%s,%s,%s,%s,%s)""",
          (ante, deck, stake, json.dumps(lineup), best_hand, best_score, outcome, notes))
    return True


def list_runs(limit: int = 100):
    cols = ["id", "ts", "ante", "deck", "stake", "lineup", "best_hand",
            "best_score", "outcome", "notes"]
    if DEMO:
        if _use_dynamo():
            try:
                return _ddb_list(limit)
            except Exception:
                _ddb_state["ok"] = False
        with _sqlite() as c:
            rows = c.execute("""SELECT id, ts, ante, deck, stake, lineup, best_hand,
                                       best_score, outcome, notes
                                FROM runs ORDER BY ts DESC, id DESC LIMIT ?""",
                             (limit,)).fetchall()
        return [dict(zip(cols, r)) for r in rows]
    rows = _exec(f"""SELECT id, ts, ante, deck, stake, lineup, best_hand,
                            best_score, outcome, notes
                     FROM {SCHEMA}.runs ORDER BY ts DESC LIMIT %s""",
                 (limit,), fetch=True)
    return [dict(zip(cols, r)) for r in rows]


def run_stats():
    if DEMO:
        if _use_dynamo():
            try:
                agg: dict = {}
                for r in _ddb_list(1000):
                    o = r["outcome"] or ""
                    a = agg.setdefault(o, {"count": 0, "_ante": 0, "top_score": 0})
                    a["count"] += 1
                    a["_ante"] += r["ante"]
                    a["top_score"] = max(a["top_score"], r["best_score"])
                return {o: {"count": a["count"],
                            "avg_ante": a["_ante"] / a["count"] if a["count"] else 0.0,
                            "top_score": a["top_score"]} for o, a in agg.items()}
            except Exception:
                _ddb_state["ok"] = False
        with _sqlite() as c:
            rows = c.execute("""SELECT outcome, count(*), coalesce(avg(ante),0),
                                       coalesce(max(best_score),0)
                                FROM runs GROUP BY outcome""").fetchall()
        return {r[0]: {"count": int(r[1]), "avg_ante": float(r[2]),
                       "top_score": int(r[3])} for r in rows}
    rows = _exec(f"""SELECT outcome, count(*), coalesce(avg(ante),0),
                            coalesce(max(best_score),0)
                     FROM {SCHEMA}.runs GROUP BY outcome""", fetch=True)
    return {r[0]: {"count": int(r[1]), "avg_ante": float(r[2]), "top_score": int(r[3])}
            for r in rows}


def delete_run(run_id):
    if DEMO:
        if _use_dynamo():
            try:
                _dynamo().delete_item(Key={"pk": "runs", "id": str(run_id)})
                return
            except Exception:
                _ddb_state["ok"] = False
        with _sqlite() as c:
            c.execute("DELETE FROM runs WHERE id = ?", (run_id,))
        return
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


# ---------------------------------------------------------------------------
# Lakebase Search (Beta): BM25 keyword + ANN vector, fused with RRF in SQL
# ---------------------------------------------------------------------------
BM25_INDEX = f"{SCHEMA}.joker_docs_bm25"


def ensure_search(jokers_df) -> dict:
    """Install the lakebase_text / lakebase_vector extensions when the project
    has Lakebase Search enabled, keep a joker_docs table (name + effect +
    playbook + tags → tsvector) in sync, and build the BM25 + ANN indexes.
    Silently reports unavailable when the extensions are not offered."""
    out = {"lbsearch": False, "docs": 0, "error": None}
    if not available() or not _state["pgvector"]:
        out["error"] = "no pgvector connection"
        return out
    try:
        _exec("CREATE EXTENSION IF NOT EXISTS lakebase_vector CASCADE")
        _exec("CREATE EXTENSION IF NOT EXISTS lakebase_text CASCADE")
    except Exception as e:
        out["error"] = f"Lakebase Search not enabled ({str(e)[:80]})"
        _state["lbsearch"] = False
        return out
    _exec(f"""CREATE TABLE IF NOT EXISTS {SCHEMA}.joker_docs (
                name    TEXT PRIMARY KEY,
                doc     TEXT NOT NULL,
                doc_tsv TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', doc)) STORED)""")
    have = {r[0]: r[1] for r in _exec(f"SELECT name, md5(doc) FROM {SCHEMA}.joker_docs", fetch=True)}
    import hashlib
    changed = 0
    for _, r in jokers_df.fillna("").iterrows():
        doc = (f"{r['name']}. {r['effect']} Archetype: {r.get('archetype','')}. "
               f"Tags: {str(r['tags']).replace('|', ', ')}. Playbook: {r.get('strategy','')}")
        if have.get(r["name"]) != hashlib.md5(doc.encode()).hexdigest():
            _exec(f"""INSERT INTO {SCHEMA}.joker_docs (name, doc) VALUES (%s, %s)
                      ON CONFLICT (name) DO UPDATE SET doc = EXCLUDED.doc""", (r["name"], doc))
            changed += 1
    out["docs"] = len(jokers_df)
    # BM25 statistics are computed at build time → build after the corpus is loaded
    if changed or not _exec("SELECT 1 FROM pg_indexes WHERE indexname = 'joker_docs_bm25'", fetch=True):
        _exec("DROP INDEX IF EXISTS " + BM25_INDEX)
        _exec(f"""CREATE INDEX joker_docs_bm25 ON {SCHEMA}.joker_docs
                  USING lakebase_bm25 (doc_tsv) WITH (default_limit = 64)""")
    if not _exec("SELECT 1 FROM pg_indexes WHERE indexname = 'joker_embeddings_ann'", fetch=True):
        n = embedding_count()
        if n >= 100:
            _exec(f"""CREATE INDEX joker_embeddings_ann ON {SCHEMA}.joker_embeddings
                      USING lakebase_ann (emb vector_cosine_ops)""")
    _state["lbsearch"] = True
    out["lbsearch"] = True
    out["changed"] = changed
    return out


def hybrid_search(query: str, top_n: int = 12):
    """Reciprocal Rank Fusion of BM25 (lakebase_text) and ANN (lakebase_vector)
    rankings — one SQL statement, scores normalised to the best hit = 1.0."""
    q = str(list(_embed([query])[0]))
    rows = _exec(f"""
        WITH v AS (
            SELECT name, ROW_NUMBER() OVER (ORDER BY emb <=> %s::vector) AS r
            FROM {SCHEMA}.joker_embeddings ORDER BY emb <=> %s::vector LIMIT 40),
        k AS (
            SELECT name, ROW_NUMBER() OVER (
                ORDER BY doc_tsv <@> to_bm25query(to_tsvector('english', %s), '{BM25_INDEX}')) AS r
            FROM {SCHEMA}.joker_docs
            WHERE doc_tsv @@ to_tsquery('english', replace(plainto_tsquery('english', %s)::text, '&', '|'))
            ORDER BY doc_tsv <@> to_bm25query(to_tsvector('english', %s), '{BM25_INDEX}') LIMIT 40)
        SELECT COALESCE(v.name, k.name) AS name,
               COALESCE(1.0 / (60 + v.r), 0) + COALESCE(1.0 / (60 + k.r), 0) AS score,
               v.r AS vrank, k.r AS krank
        FROM v FULL OUTER JOIN k ON v.name = k.name
        ORDER BY score DESC LIMIT %s""",
        (q, q, query, query, query, top_n), fetch=True)
    if not rows:
        return []
    top = float(rows[0][1])
    return [(r[0], float(r[1]) / top, r[2], r[3]) for r in rows]


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
# MLflow tracing (coach tool calls) — no-op unless mlflow is importable and
# we are running on Databricks with an experiment we can write to
# ---------------------------------------------------------------------------
_trace: dict = {"on": False, "why": "not initialised"}


def init_tracing(experiment: str | None = None) -> dict:
    if DEMO:
        _trace.update(on=False, why="demo/hybrid mode — tracing only on the Databricks app")
        return _trace
    try:
        import mlflow
        mlflow.set_tracking_uri("databricks")
        exp = experiment or os.environ.get("COACH_EXPERIMENT", "")
        if not exp:
            me = _workspace_client().current_user.me().user_name
            exp = f"/Users/{me}/balatro-coach-traces"
        mlflow.set_experiment(exp)
        _trace.update(on=True, why=f"OK ({exp})")
    except Exception as e:
        _trace.update(on=False, why=f"off ({str(e)[:120]})")
    return _trace


class span:
    """`with db.span("name", inputs) as sp:` → MLflow span when tracing is on."""
    def __init__(self, name: str, inputs=None, kind: str = "TOOL"):
        self.name, self.inputs, self.kind, self._cm, self.sp = name, inputs, kind, None, None

    def __enter__(self):
        if _trace["on"]:
            try:
                import mlflow
                self._cm = mlflow.start_span(name=self.name, span_type=self.kind)
                self.sp = self._cm.__enter__()
                if self.inputs is not None:
                    self.sp.set_inputs(self.inputs)
            except Exception:
                self._cm = None
        return self

    def out(self, outputs):
        if self.sp is not None:
            try: self.sp.set_outputs(outputs)
            except Exception: pass

    def __exit__(self, *exc):
        if self._cm is not None:
            try: self._cm.__exit__(*exc)
            except Exception: pass
        return False


# ---------------------------------------------------------------------------
# Chat (AI strategist)
# ---------------------------------------------------------------------------

def chat_endpoints() -> list[str]:
    """Ordered list of chat endpoints to try: the pinned one, then candidates."""
    eps = ([CHAT_ENDPOINT] if CHAT_ENDPOINT else []) + [e for e in CHAT_CANDIDATES if e != CHAT_ENDPOINT]
    return [e for e in eps if e not in _state["bad_eps"]] or eps


def chat_endpoint() -> str:
    return _state["chat_ep"] or chat_endpoints()[0]


def _max_tokens_for(ep: str, n: int) -> int:
    # reasoning models spend tokens thinking before answering
    return max(n, 2000) if "gpt-oss" in ep else n


def chat(prompt: str, max_tokens: int = 900, temperature: float = 0.4) -> str:
    if not ai_ok():
        raise RuntimeError("demo mode — the AI coach runs on the Databricks deployment")
    m = chat_with_tools([{"role": "user", "content": prompt}], [],
                        max_tokens=max_tokens, temperature=temperature)
    return (m.get("content") or "").strip()


def chat_with_tools(messages: list[dict], tools: list[dict],
                    max_tokens: int = 900, temperature: float = 0.3,
                    endpoint: str | None = None) -> dict:
    """One OpenAI-style chat turn with function tools. Returns the raw
    assistant message dict ({"role","content","tool_calls":[...]}) with an
    extra "_endpoint" key. Falls through the candidate list on endpoint
    errors (missing model, no permission, rate limit) and remembers the
    first endpoint that works."""
    if not ai_ok():
        raise RuntimeError("demo mode — the AI coach runs on the Databricks deployment")
    w = _workspace_client()
    eps = [endpoint] if endpoint else chat_endpoints()
    last = None
    for ep in eps:
        body = {"messages": messages, "max_tokens": _max_tokens_for(ep, max_tokens),
                "temperature": temperature}
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        try:
            resp = w.api_client.do("POST", f"/serving-endpoints/{ep}/invocations", body=body)
            msg = dict(resp["choices"][0]["message"])
            c = msg.get("content")
            if isinstance(c, list):       # reasoning models return content parts
                msg["content"] = "".join(
                    (p.get("text") or "") if isinstance(p, dict) else str(p)
                    for p in c if not (isinstance(p, dict) and p.get("type") == "reasoning"))
            elif c is None:
                msg["content"] = ""
            if not endpoint:
                _state["chat_ep"] = ep
            msg["_endpoint"] = ep
            return msg
        except Exception as e:
            last = e
            if not endpoint:
                _state["bad_eps"].add(ep)
                continue
            raise
    raise RuntimeError(f"no chat endpoint answered: {str(last)[:160]}")


# ---------------------------------------------------------------------------
# Genie — natural-language questions over the workspace.balatro tables
# ---------------------------------------------------------------------------

def _genie_do(method: str, path: str, body: dict | None = None) -> dict:
    w = _workspace_client()
    return w.api_client.do(method, path, body=body) or {}


def genie_start(question: str) -> dict:
    """Kick off a Genie conversation; returns ids for polling."""
    r = _genie_do("POST", f"/api/2.0/genie/spaces/{GENIE_SPACE_ID}/start-conversation",
                  {"content": question[:1000]})
    return {"conversation_id": r.get("conversation_id"),
            "message_id": r.get("message_id")}


def genie_agent_run(question: str, conversation_id: str | None = None,
                    timeout_s: int = 240) -> dict:
    """Genie Agent mode (GA Aug 2026): multi-step reasoning over the space,
    streamed as server-sent events. Blocks until the response completes and
    returns the final report text, the SQL it ran, and a conversation id
    for follow-ups. Uses the app's own service-principal auth."""
    import requests
    w = _workspace_client()
    url = f"{w.config.host.rstrip('/')}/api/2.0/genie/agents/{GENIE_SPACE_ID}/responses"
    body = {"input": [{"type": "message", "role": "user",
                       "content": [{"type": "input_text", "text": question}]}]}
    if conversation_id:
        body["conversation_id"] = conversation_id
    headers = dict(w.config.authenticate())
    headers.update({"Content-Type": "application/json", "Accept": "text/event-stream"})
    r = requests.post(url, json=body, headers=headers, stream=True, timeout=(15, timeout_s))
    if r.status_code != 200:
        raise RuntimeError(f"agent mode HTTP {r.status_code}: {r.text[:200]}")
    items: dict = {}
    order: list = []
    final: dict = {}
    kinds: list = []
    for raw in r.iter_lines(decode_unicode=True):
        if not raw or not raw.startswith("data:"):
            continue
        try:
            ev = json.loads(raw[5:].strip())
        except Exception:
            continue
        typ = ev.get("type") or ev.get("event") or ""
        data = ev.get("data") if isinstance(ev.get("data"), dict) else ev
        if typ and typ not in kinds:
            kinds.append(typ)
        item = data.get("item") or data.get("output_item")
        if isinstance(item, dict):
            iid = item.get("id") or f"i{len(order)}"
            if iid not in items:
                order.append(iid)
            items[iid] = item
        if typ in ("response.completed", "response.failed"):
            final = data.get("response") if isinstance(data.get("response"), dict) else data
            break
    for it in (final.get("output") or []):
        if isinstance(it, dict):
            iid = it.get("id") or f"o{len(order)}"
            if iid not in items:
                order.append(iid)
            items[iid] = it
    text_parts, sqls, steps = [], [], 0
    for iid in order:
        it = items[iid]
        t = it.get("type", "")
        if t == "message":
            for c in it.get("content") or []:
                if isinstance(c, dict) and c.get("text"):
                    text_parts.append(c["text"])
        elif t == "function_call":
            steps += 1
            args = it.get("arguments")
            try:
                a = json.loads(args) if isinstance(args, str) else (args or {})
            except Exception:
                a = {"raw": str(args)[:400]}
            for k in ("sql", "query", "statement"):
                if isinstance(a, dict) and a.get(k):
                    sqls.append(str(a[k])[:2000])
                    break
        elif t == "reasoning":
            steps += 1
    status = final.get("status") or ("completed" if text_parts else "failed")
    err = final.get("error")
    return {"status": status, "text": "\n\n".join(text_parts).strip(),
            "sql": sqls[:6], "steps": steps,
            "conversation_id": final.get("conversation_id") or conversation_id,
            "error": (json.dumps(err)[:300] if err else None), "event_kinds": kinds[:12]}


def genie_poll(conversation_id: str, message_id: str) -> dict:
    """One poll step: status plus, when COMPLETED, text/sql/result rows."""
    base = f"/api/2.0/genie/spaces/{GENIE_SPACE_ID}/conversations/{conversation_id}/messages/{message_id}"
    m = _genie_do("GET", base)
    out: dict = {"status": m.get("status", "UNKNOWN")}
    if out["status"] != "COMPLETED":
        return out
    texts, sql, attach_id = [], "", None
    for a in m.get("attachments", []):
        if a.get("text"):
            texts.append(a["text"].get("content", ""))
        if a.get("query"):
            sql = a["query"].get("query", "")
            attach_id = a.get("attachment_id")
            if a["query"].get("description"):
                texts.append(a["query"]["description"])
    out["text"] = "\n\n".join(t for t in texts if t)
    out["sql"] = sql
    if attach_id:
        try:
            qr = _genie_do("GET", f"{base}/attachments/{attach_id}/query-result")
            sr = qr.get("statement_response", {})
            cols = [c.get("name", "") for c in
                    sr.get("manifest", {}).get("schema", {}).get("columns", [])]
            rows = sr.get("result", {}).get("data_array", []) or []
            out["columns"] = cols
            out["rows"] = rows[:50]
        except Exception as e:
            out["result_error"] = str(e)[:120]
    return out


# ---------------------------------------------------------------------------
# Diagnostics — printed to app logs at startup
# ---------------------------------------------------------------------------

def diagnostics(run_chat_test: bool = True) -> dict:
    if DEMO and not CONNECTED:
        return {
            "mode": "demo — running without Databricks (review copy)",
            "lakebase": "OK (demo: run log in local SQLite)",
            "pgvector": "off (demo)",
            "embeddings": "off (demo — keyword TF-IDF search)",
            "chat": "off (demo — deterministic playbook coach)",
        }
    d: dict = {}
    if DEMO and CONNECTED:
        d["mode"] = "hybrid — AI via Databricks, run log in local SQLite"
    s = init_schema()
    d["lakebase"] = "OK" if s["lakebase"] else f"FAIL ({s['error']})"
    d["pgvector"] = "OK" if s["pgvector"] else "unavailable (JSONB fallback)"
    try:
        e = _embed(["diagnostic ping"])
        d["embeddings"] = f"OK ({EMBED_ENDPOINT}, dim={len(e[0])})"
    except Exception as ex:
        d["embeddings"] = f"FAIL ({str(ex)[:160]})"
    d["search"] = "Lakebase Search (BM25 + ANN, RRF)" if _state["lbsearch"] else \
        ("pgvector" if s["pgvector"] else "keyword")
    if run_chat_test:
        try:
            chat("Reply with the single word: ok", max_tokens=5, temperature=0.0)
            d["chat"] = f"OK ({chat_endpoint()})"
        except Exception as ex:
            d["chat"] = f"FAIL ({str(ex)[:160]})"
    return d
