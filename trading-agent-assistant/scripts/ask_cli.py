#!/usr/bin/env python
"""
CLI interface for the Trading Agent Assistant.

Usage:
    python scripts/ask_cli.py "Why did the model buy on 2025-07-21?"
    python scripts/ask_cli.py --api "What trades were made in August 2025?"
    python scripts/ask_cli.py -i  # Interactive multi-turn chat mode

Modes:
    Default       — calls the agent directly (no server needed)
    --api         — sends the question to a running FastAPI server
    --interactive — interactive multi-turn conversation preserving context
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

# Add project root to sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Configure stdout/stderr for UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def _ask_direct(
    question: str,
    conversation_history: Optional[List[Dict[str, Any]]] = None,
) -> Optional[Dict[str, Any]]:
    """Call the agent directly (no HTTP server required)."""
    from app.agent import ask

    print(f"\n{'-' * 60}")
    print(f"  Question: {question}")
    print(f"{'-' * 60}\n")

    try:
        result = ask(question, conversation_history=conversation_history)
    except Exception as e:
        print(f"[Error]: {e}")
        return None

    # Print the answer
    print("Answer:\n")
    print(result["answer"])

    # Print tool calls summary
    if result["tool_calls"]:
        print(f"\n{'-' * 60}")
        print(f"  Tool calls made: {len(result['tool_calls'])}")
        print(f"{'-' * 60}")
        for i, tc in enumerate(result["tool_calls"], 1):
            print(f"\n  [{i}] {tc['tool']}")
            print(f"      Input:  {json.dumps(tc['input'], indent=None)}")
            output = tc["output"]
            if isinstance(output, list):
                print(f"      Output: {len(output)} record(s) returned")
            elif isinstance(output, dict) and "error" in output:
                print(f"      Output: Error: {output['error']}")
            else:
                print(f"      Output: {json.dumps(output, indent=None)[:200]}")

    print()
    return result


def _ask_api(
    question: str,
    conversation_history: Optional[List[Dict[str, Any]]] = None,
    base_url: str = "http://localhost:8000",
) -> Optional[Dict[str, Any]]:
    """Send the question to the running FastAPI server."""
    payload: Dict[str, Any] = {"question": question}
    if conversation_history:
        payload["conversation_history"] = conversation_history

    try:
        import requests
    except ImportError:
        # Fall back to urllib if requests is not installed
        import urllib.request
        import urllib.error

        req = urllib.request.Request(
            f"{base_url}/ask",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8")
            print(f"[HTTP Error] {e.code}: {body}")
            return None
        except urllib.error.URLError as e:
            print(f"[Connection error]: {e.reason}")
            print(f"   Is the server running at {base_url}?")
            return None
    else:
        try:
            resp = requests.post(
                f"{base_url}/ask",
                json=payload,
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.ConnectionError:
            print(f"[Connection error]: is the server running at {base_url}?")
            return None
        except requests.exceptions.HTTPError as e:
            print(f"[HTTP error]: {e}")
            print(f"   Response: {resp.text}")
            return None

    print(f"\n{'-' * 60}")
    print(f"  Question: {question}")
    print(f"{'-' * 60}\n")
    print("Answer:\n")
    print(data["answer"])

    if data.get("tool_calls"):
        print(f"\n  Tool calls: {len(data['tool_calls'])}")

    print()
    return data


def _run_interactive(use_api: bool = False, base_url: str = "http://localhost:8000") -> None:
    """Run an interactive multi-turn chat session with memory."""
    print("\n" + "=" * 60)
    print("  Trading Agent Assistant -- Interactive Chat")
    print(f"  Mode: {'API (' + base_url + ')' if use_api else 'Direct'}")
    print("  Type 'exit', 'quit', or Ctrl+C to stop.")
    print("=" * 60)

    history: List[Dict[str, Any]] = []

    while True:
        try:
            prompt = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not prompt:
            continue
        if prompt.lower() in ("exit", "quit"):
            print("\nGoodbye!")
            break

        if use_api:
            res = _ask_api(prompt, conversation_history=history, base_url=base_url)
        else:
            res = _ask_direct(prompt, conversation_history=history)

        if res and "answer" in res:
            history.append({"role": "user", "content": prompt})
            history.append({"role": "assistant", "content": res["answer"]})


def main() -> None:
    """Parse arguments and dispatch."""
    parser = argparse.ArgumentParser(
        description="Ask the Trading Agent Assistant a question.",
    )
    parser.add_argument(
        "question",
        type=str,
        nargs="?",
        default=None,
        help="The question to ask (wrap in quotes). Omit to enter interactive mode.",
    )
    parser.add_argument(
        "--api",
        action="store_true",
        help="Send the question to a running FastAPI server instead of calling the agent directly.",
    )
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:8000",
        help="Base URL of the FastAPI server (default: http://localhost:8000).",
    )
    parser.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="Start an interactive multi-turn chat session.",
    )

    args = parser.parse_args()

    if args.interactive or args.question is None:
        _run_interactive(use_api=args.api, base_url=args.url)
    else:
        if args.api:
            _ask_api(args.question, base_url=args.url)
        else:
            _ask_direct(args.question)


if __name__ == "__main__":
    main()
