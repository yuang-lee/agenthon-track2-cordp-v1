#!/usr/bin/env python
"""One chat-completions client for every text pipeline: retries, JSON, a call ledger, a budget.

## Executive summary (read this first)

`chat(messages, ...)` posts to an OpenAI-compatible endpoint (default: NVIDIA's hosted API with
the House model id, key read from `MODEL_API_KEY_FILE`, default `~/.nvidia_api_key`; at scoring
time the harness's `MODEL_ENDPOINT` / `MODEL_NAME` / `MODEL_TOKEN` take over), retries transient
errors (429, 5xx, timeouts; backoff 5, 15, 45 s), asks for JSON output and falls back to the last
balanced `{...}` in the reply, and returns the parsed object with the raw text, the reasoning
trace if any, and token usage. Thinking is OFF unless asked (`thinking=True`): the House model
thinks by default and spends the completion budget before any JSON (organizer's measurement).
`Budget` counts admitted requests per unit against the House cap (25) so no pipeline can exceed
it. Nothing here reads a document or touches a forecast.
"""
from __future__ import annotations

import json
import os
import pathlib
import threading
import time
import urllib.error
import urllib.request

DEFAULT_ENDPOINT = "https://integrate.api.nvidia.com/v1"
DEFAULT_MODEL = "nvidia/nemotron-3-super-120b-a12b"
HOUSE_CAP = 25
HOUSE_MAX_OUTPUT = 4000
TIMEOUT_S = float(os.environ.get("LLM_TIMEOUT", "90"))   # a hung hosted request once stalled a run for 20+ minutes
_PACE_LOCK, _LAST_CALL = threading.Lock(), [0.0]


def _pace() -> None:
    """Global pacing: LLM_MIN_INTERVAL seconds between call starts (the hosted API rate-limits per key)."""
    gap = float(os.environ.get("LLM_MIN_INTERVAL", "0") or 0)
    if gap <= 0:
        return
    with _PACE_LOCK:
        wait = _LAST_CALL[0] + gap - time.time()
        if wait > 0:
            time.sleep(wait)
        _LAST_CALL[0] = time.time()


def _key() -> str:
    tok = os.environ.get("MODEL_TOKEN") or os.environ.get("MODEL_API_KEY")
    if tok:
        return tok
    f = os.environ.get("MODEL_API_KEY_FILE", "~/.nvidia_api_key")
    p = pathlib.Path(os.path.expanduser(f))
    return p.read_text().strip() if p.exists() else "none"


def endpoint() -> str:
    e = os.environ.get("MODEL_ENDPOINT", DEFAULT_ENDPOINT).rstrip("/")
    return e if e.endswith("/v1") else e + "/v1"


def model_name() -> str:
    return os.environ.get("MODEL_NAME", DEFAULT_MODEL)


def extract_json(text: str, must_have=()):
    """Last balanced {...} in `text` that parses (and mentions an expected key, if given)."""
    best, stack, start = None, [], None
    for i, ch in enumerate(text):
        if ch == "{":
            if not stack:
                start = i
            stack.append(i)
        elif ch == "}" and stack:
            stack.pop()
            if not stack:
                try:
                    obj = json.loads(text[start:i + 1])
                    if isinstance(obj, dict) and (not must_have or any(k in obj for k in must_have)):
                        best = obj
                except ValueError:
                    pass
    return best


class Budget:
    """Admitted requests per unit; a failed or retried request is charged like the House does."""

    def __init__(self, cap: int = HOUSE_CAP):
        self.cap, self.used = cap, {}

    def charge(self, unit: str) -> None:
        self.used[unit] = self.used.get(unit, 0) + 1
        if self.used[unit] > self.cap:
            raise RuntimeError(f"{unit}: House budget of {self.cap} requests exceeded")


def chat(messages: list[dict], *, unit: str = "", budget: Budget | None = None, max_tokens: int = 1500,
         temperature: float = 0.0, seed: int = 0, thinking: bool = False, json_mode: bool = True,
         must_have=(), retries: int = 8, system: str | None = None, n: int = 1) -> dict:
    """Return {"parsed", "raw", "reasoning", "usage", "attempts", "error"}; parsed is None on failure."""
    if system:
        messages = [{"role": "system", "content": system}] + messages
    body = {"model": model_name(), "messages": messages, "temperature": temperature, "seed": seed,
            "max_tokens": min(max_tokens, HOUSE_MAX_OUTPUT),
            "chat_template_kwargs": {"enable_thinking": bool(thinking)}}
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    if n > 1:
        body["n"] = int(n)                     # several completions in ONE admitted request (the endpoint supports it)
    req = urllib.request.Request(endpoint() + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {_key()}"})
    out = {"parsed": None, "raw": "", "reasoning": "", "usage": {}, "attempts": 0, "error": ""}
    for attempt in range(retries + 1):
        out["attempts"] = attempt + 1
        if budget is not None:
            budget.charge(unit)
        _pace()
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
                payload = json.loads(r.read())
            if n > 1:                              # every completion, parsed separately
                out["choices"] = []
                for ch in payload.get("choices", []):
                    raw = (ch.get("message") or {}).get("content") or ""
                    try:
                        parsed = json.loads(raw)
                    except ValueError:
                        parsed = extract_json(raw, must_have)
                    out["choices"].append({"parsed": parsed, "raw": raw})
            msg = payload["choices"][0]["message"]
            out["raw"] = msg.get("content") or ""
            out["reasoning"] = msg.get("reasoning_content") or ""
            out["usage"] = payload.get("usage", {})
            try:
                out["parsed"] = json.loads(out["raw"])
            except ValueError:
                out["parsed"] = extract_json(out["raw"], must_have)
            if out["parsed"] is None:
                out["error"] = "no JSON in reply"
                if payload["choices"][0].get("finish_reason") == "length":
                    out["error"] = f"reply hit max_tokens={body['max_tokens']} before JSON"
            return out
        except urllib.error.HTTPError as exc:
            code = exc.code
            detail = exc.read()[:200].decode(errors="replace")
            out["error"] = f"HTTP {code}: {detail}"
            if code == 400 and json_mode:          # endpoint without JSON mode: retry plain
                body.pop("response_format", None)
                json_mode = False
                req = urllib.request.Request(req.full_url, data=json.dumps(body).encode(), headers=req.headers)
                continue
            if code not in (429, 500, 502, 503, 504) or attempt == retries:
                return out
            if code == 429:                        # rate limit: linear wait, LLM_429_WAIT seconds per attempt (default 6)
                time.sleep(float(os.environ.get("LLM_429_WAIT", "6")) * (attempt + 1))
                continue
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            out["error"] = f"{type(exc).__name__}: {exc}"
            if attempt == retries:
                return out
        time.sleep(5 * 3 ** attempt)
    return out
