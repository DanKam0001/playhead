"""Vercel entry point.

Vercel's Python runtime looks for an ASGI callable named `app`. The legacy
`builds`/`routes` form in vercel.json is used deliberately: it forwards the
original request path to the function, which the newer `rewrites` form does
not, and FastAPI needs the real path to route.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server.main import app  # noqa: E402

__all__ = ["app"]
