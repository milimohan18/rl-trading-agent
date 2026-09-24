"""
FastAPI application — POST /ask endpoint for the Trading Agent Assistant.
"""

from __future__ import annotations

import os
import traceback
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# Load environment before importing agent
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from app.agent import ask  # noqa: E402

_INDEX_HTML = os.path.join(os.path.dirname(__file__), "static", "index.html")

# Any one of these enables the agent (see app.agent.ask for precedence)
_PROVIDER_KEYS = ("GROQ_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY")


def _api_key_configured() -> bool:
    return any(os.getenv(k) for k in _PROVIDER_KEYS)


# ── Lifespan ────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown hooks."""
    # Verify API key is configured (warn early, don't crash)
    if not _api_key_configured():
        print(
            "\n⚠️  WARNING: none of GROQ_API_KEY, ANTHROPIC_API_KEY or "
            "OPENAI_API_KEY is set. "
            "Requests to /ask will fail.\n"
            "   Copy .env.example to .env and add your key.\n"
        )
    yield


# ── App ─────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Trading Agent Assistant",
    description=(
        "An LLM-powered assistant that answers natural-language questions "
        "about an RL trading system's behavior, using tool calling to "
        "fetch real trade data and indicator values."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ── Request / Response models ───────────────────────────────────────────────

class AskRequest(BaseModel):
    """Request body for the /ask endpoint."""
    question: str = Field(
        ...,
        min_length=1,
        description="The natural-language question to ask the trading agent.",
        json_schema_extra={"examples": ["What trades were made in August 2025?"]},
    )
    conversation_history: Optional[list[dict]] = Field(
        default=None,
        description="Optional prior conversation turns for multi-turn dialogues.",
    )


class ToolCallRecord(BaseModel):
    """A single tool call made by the agent."""
    tool: str
    input: dict
    output: object


class AskResponse(BaseModel):
    """Response body from the /ask endpoint."""
    answer: str
    tool_calls: list[ToolCallRecord]


# ── Endpoints ───────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def index():
    """Chat page — a browser front end for POST /ask."""
    return FileResponse(_INDEX_HTML)


@app.get("/health")
async def health():
    """Health check — confirms the service is running."""
    return {
        "status": "ok",
        "service": "trading-agent-assistant",
        "api_key_configured": _api_key_configured(),
    }


@app.post("/ask", response_model=AskResponse)
async def ask_endpoint(request: AskRequest):
    """Ask the trading agent a natural-language question.

    The agent uses tool calling to fetch real trade data and indicator
    values, then composes a grounded answer.
    """
    question = request.question.strip()

    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    if not _api_key_configured():
        raise HTTPException(
            status_code=500,
            detail=(
                "No LLM API key is configured. Set GROQ_API_KEY, "
                "ANTHROPIC_API_KEY or OPENAI_API_KEY."
            ),
        )

    try:
        result = ask(
            question,
            conversation_history=request.conversation_history,
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Agent error: {str(e)}",
        )

    return AskResponse(
        answer=result["answer"],
        tool_calls=[
            ToolCallRecord(**tc) for tc in result["tool_calls"]
        ],
    )
