"""AWS Lambda entry point for the demo-mode review deployment.

Wraps the FastAPI app with Mangum so a Lambda Function URL (fronted by
CloudFront at balatro.ajf.codes) can serve it. Demo mode is forced on:
Lambda has no Databricks credentials, and its filesystem is read-only
outside /tmp, so the SQLite run log lives there (ephemeral per container —
fine for a review copy).
"""
import os

os.environ.setdefault("DEMO_MODE", "1")
os.environ.setdefault("DEMO_DB_PATH", "/tmp/demo_runs.sqlite3")

from mangum import Mangum  # noqa: E402

from app import app  # noqa: E402

handler = Mangum(app)
