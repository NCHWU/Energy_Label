"""FastAPI web application for the Energy Label leaderboard."""

import json
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

app = FastAPI(title="LLM Energy Label", description="Energy efficiency ratings for LLMs")

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = Path(__file__).resolve().parents[3] / "results"

app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")


def _load_leaderboard(domain: str) -> List[Dict]:
    """Load pre-computed scoreboard for a domain."""
    path = RESULTS_DIR / domain / "scoreboard.json"
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _list_domains() -> List[Dict]:
    """List available benchmark domains with metadata."""
    meta_path = RESULTS_DIR / "domains.json"
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    # Auto-discover from directories
    domains = []
    if RESULTS_DIR.exists():
        for d in sorted(RESULTS_DIR.iterdir()):
            if d.is_dir() and (d / "scoreboard.json").exists():
                domains.append({"id": d.name, "name": d.name.replace("_", " ").title()})
    return domains


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    domains = _list_domains()
    return templates.TemplateResponse("index.html", {
        "request": request,
        "domains": domains,
    })


@app.get("/api/domains")
async def api_domains():
    return _list_domains()


@app.get("/api/leaderboard/{domain}")
async def api_leaderboard(domain: str):
    return _load_leaderboard(domain)
