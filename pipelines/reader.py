#!/usr/bin/env python
"""Corpus loading, the engine context every prompt states, and the prompt of each elicitation.

## Executive summary (read this first)

Every text pipeline reads the same things: the dated corpus of the unit (documents at or before
the as-of date, newest first, the organizer's 8 documents x 6,000 characters), and the engine's
own numbers per asset (as-of level, horizon standard deviation, recent moves in sd units, value
unit). What differs between pipelines is the *object* the model is asked for, one prompt each:

| prompt | asks for | consumed by |
|---|---|---|
| `knobs` | organizer's `drift_bp` + `vol_scale` per asset | R1 direct adjustment, gates |
| `scenarios` | up to 3 scenarios per asset {prob, shift in sd, width x} | L7 scenario mixture |
| `quantiles` | q05/q25/q50/q75/q95 per cell in target units | L2 blend with the engine |
| `doc` | ONE document at a time: P(down/flat/up), P(uncertainty lower/same/higher) per asset | L3 LEAP posterior |
| `readers` | the `knobs` object from k heterogeneous readers (personas, temperature) | L5 dispersion -> width, L8 aggregation |
| `rank` | scores 0-9 for K engine-proposed centre paths per asset | L1 planner re-weighting |

The card author's `[text].notes` never enter a prompt (they paraphrase the answer's framing).
"""
from __future__ import annotations

import json
import pathlib
import sys
import tomllib

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
from m0_baseline import panel_files  # noqa: E402

MAX_DOCS, MAX_DOC_CHARS = 8, 6000
PERSONAS = [
    "You are a hawkish rates trader who reads every central-bank text for signs of tightening.",
    "You are a dovish macro economist focused on growth risks and easing.",
    "You are a risk manager who cares only about how much uncertainty the documents imply.",
    "You are a momentum quant who trusts recent price action over narrative.",
    "You are a contrarian strategist who assumes the documents are already priced in.",
]
SHIFT_GRID = [-1.5, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 1.5]   # L1 candidate centres, in horizon sd


def load_corpus(unit_dir: pathlib.Path, asof: str) -> tuple[list[dict], int]:
    """Documents dated <= as-of, newest first, text truncated; (docs, n_excluded_by_date)."""
    idx = unit_dir / "text" / "corpus_index.json"
    if not idx.exists():
        return [], 0
    entries = json.loads(idx.read_text()).get("documents", [])
    docs, excluded = [], 0
    for e in entries:
        ts = str(e.get("timestamp", ""))[:10]
        f = unit_dir / "text" / e.get("file", "")
        if not ts or ts > asof or not f.is_file():
            excluded += 1
            continue
        docs.append({"doc_id": e.get("doc_id"), "timestamp": ts, "doc_type": e.get("doc_type", "doc"),
                     "text": f.read_text(errors="replace")[:MAX_DOC_CHARS]})
    docs.sort(key=lambda d: d["timestamp"], reverse=True)
    return docs, excluded


def doc_blob(docs: list[dict]) -> str:
    return "\n\n".join(f"--- {d['doc_id']} ({d['timestamp']}, {d['doc_type']}) ---\n{d['text']}" for d in docs[:MAX_DOCS]) or "(no documents)"


def doc_blob_clean(docs: list[dict]) -> str:
    """The f4event blob: boilerplate stripped (`clean_doc`), MAX_DOC_CHARS per document — the form
    `scripts/read_f4event.py` measured E10 with. The dev-09-text image sent the raw text (E10 addendum)."""
    return "\n\n".join(f"--- {d['doc_id']} ({d['timestamp']}, {d['doc_type']}) ---\n{clean_doc(d['text'])[:MAX_DOC_CHARS]}" for d in docs[:MAX_DOCS]) or "(no documents)"


def engine_context(unit_dir: pathlib.Path, samples: np.ndarray, cells: list, info: dict) -> dict:
    """Per asset: level, value unit, horizon sd (longest horizon, from the engine's draws), the
    last 21 and 63 observations' change in units and in sd; per cell: median and sd."""
    card = tomllib.loads((unit_dir / "card.toml").read_text())
    t = card["targets"]
    asof = pd.Timestamp(card["provenance"]["data_cutoff"])
    unit = t.get("value_unit", "")
    ctx = {"asof": str(asof.date()), "target_type": t.get("target_type", "level"), "value_unit": unit,
           "horizons": sorted(int(h) for h in t["horizons"]), "assets": {}, "cells": {}}
    for i, (a, h) in enumerate(cells):
        col = samples[:, i]
        ctx["cells"][f"{a}@{h}"] = {"median": float(np.median(col)), "sd": float(col.std())}
    for a in sorted(set(a for a, _ in cells)):
        hmax = max(h for x, h in cells if x == a)
        sd_h = ctx["cells"][f"{a}@{hmax}"]["sd"]
        s = None
        for p in panel_files(unit_dir):
            df = pd.read_parquet(p)
            colname = "asset" if "asset" in df.columns else "asset_id"
            g = df[df[colname] == a]
            if not g.empty:
                s = g.assign(date=pd.to_datetime(g["date"])).sort_values("date")
                s = s[s["date"] <= asof].set_index("date")["value"].astype(float)
                break
        level = float(s.iloc[-1]) if s is not None and len(s) else float("nan")
        moves = {}
        if s is not None:
            for k in (21, 63):
                if len(s) > k:
                    d = float(s.iloc[-1] - s.iloc[-1 - k]) if ctx["target_type"] == "level" else float(np.log1p(s.iloc[-k:]).sum())
                    moves[k] = {"change": d, "in_sd": d / sd_h if sd_h > 0 else 0.0}
        ctx["assets"][a] = {"level": level, "sd_h": sd_h, "hmax": hmax, "moves": moves,
                            "last_obs": str(s.index[-1].date()) if s is not None and len(s) else ""}
    return ctx


def _header(ctx: dict) -> list[str]:
    tt = ctx["target_type"]
    lines = [f"As-of date: {ctx['asof']}. Nothing after this date is known to you; use only the documents and numbers below.",
             f"Target type: {tt} ({'the level of the series' if tt == 'level' else 'cumulative log return from the as-of date'}), value unit: {ctx['value_unit'] or 'as in the panel'}. Horizons (business days): {ctx['horizons']}.",
             "Per asset: the as-of level, the statistical forecast's standard deviation at the longest horizon, and the realized change over the last 21 and 63 observations (in units and in multiples of that sd):"]
    for a, x in ctx["assets"].items():
        mv = "; ".join(f"last {k}: {m['change']:+.6g} ({m['in_sd']:+.2f} sd)" for k, m in x["moves"].items()) or "no recent history"
        lines.append(f"  {a}: level {x['level']:.6g} (last observation {x['last_obs']}), horizon sd {x['sd_h']:.6g}; {mv}")
    return lines


def prompt_knobs(ctx: dict, docs: list[dict]) -> str:
    """The organizer's reasoning-agent object (baselines/reasoning_agent.py), same wording for the numbers."""
    tt = ctx["target_type"]
    lines = ["You are adjusting a statistical forecast using dated documents."] + _header(ctx)
    lines += ["", f"Documents ({min(len(docs), MAX_DOCS)}), newest first:", doc_blob(docs), "",
              "For EACH asset, give two numbers:",
              "  drift_bp  : expected directional shift over the longest horizon.",
              ("              For log_return, 1 bp adds 0.0001 to the cumulative log return." if tt == "log_return"
               else "              Basis points of the MAGNITUDE of the current level; positive means up."),
              "              Use 0 if the documents say nothing. The shift is clamped to +-3 horizon standard deviations.",
              "  vol_scale : multiplier on the statistical standard deviation, in [0.5, 2.0].",
              "              >1 if the documents imply more uncertainty than usual, <1 if less.",
              "", "Reply with JSON only, no prose:",
              '{"assets": {"<asset>": {"drift_bp": <float>, "vol_scale": <float>, "because": "<one sentence citing a doc_id>"}}}']
    return "\n".join(lines)


def prompt_scenarios(ctx: dict, docs: list[dict]) -> str:
    lines = ["You are turning dated documents into a small set of scenarios for a statistical forecast."] + _header(ctx)
    lines += ["", f"Documents ({min(len(docs), MAX_DOCS)}), newest first:", doc_blob(docs), "",
              "For EACH asset give 1 to 3 scenarios that the documents support, with probabilities summing to 1.",
              "  shift_sd : where the scenario centres the asset at the longest horizon, in multiples of the horizon sd, on the grid -2, -1.5, -1, -0.5, -0.25, 0, 0.25, 0.5, 1, 1.5, 2.",
              "  width    : multiplier on the statistical spread inside the scenario, in [0.5, 2].",
              "If the documents say nothing decisive, give one scenario with shift_sd 0 and width 1.",
              "", "Reply with JSON only, no prose:",
              '{"assets": {"<asset>": {"scenarios": [{"label": "<short>", "prob": <float>, "shift_sd": <float>, "width": <float>, "doc_id": "<id>"}]}}}']
    return "\n".join(lines)


def prompt_quantiles(ctx: dict, docs: list[dict]) -> str:
    lines = ["You are forecasting directly from dated documents and the numbers below."] + _header(ctx)
    lines += ["Per (asset, horizon) cell, the statistical forecast's median and sd:"]
    lines += [f"  {k}: median {v['median']:.6g}, sd {v['sd']:.6g}" for k, v in ctx["cells"].items()]
    lines += ["", f"Documents ({min(len(docs), MAX_DOCS)}), newest first:", doc_blob(docs), "",
              "For EACH cell give your own 5th, 25th, 50th, 75th and 95th percentiles of the target value, in the target's units (not in sd), increasing.",
              "", "Reply with JSON only, no prose:",
              '{"cells": {"<asset>@<horizon>": {"q05": <float>, "q25": <float>, "q50": <float>, "q75": <float>, "q95": <float>}}}']
    return "\n".join(lines)


def prompt_doc(ctx: dict, doc: dict) -> str:
    lines = ["You are reading ONE dated document in isolation and stating what it implies for a statistical forecast."] + _header(ctx)
    lines += ["", f"--- {doc['doc_id']} ({doc['timestamp']}, {doc['doc_type']}) ---", doc["text"], "",
              "For EACH asset, from this document alone:",
              "  direction  : probabilities that the asset ends the longest horizon at least 0.5 sd below (down), within (flat), or at least 0.5 sd above (up) the statistical forecast's centre; they sum to 1. Use 1/3 each if the document is silent.",
              "  uncertainty: probabilities that the spread should be lower, same, higher than the statistical forecast; they sum to 1.",
              "  relevant   : true if the document bears on this asset at all.",
              "", "Reply with JSON only, no prose:",
              '{"assets": {"<asset>": {"relevant": <bool>, "direction": {"down": <p>, "flat": <p>, "up": <p>}, "uncertainty": {"lower": <p>, "same": <p>, "higher": <p>}}}}']
    return "\n".join(lines)


def prompt_rank(ctx: dict, docs: list[dict]) -> str:
    lines = ["A statistical engine proposes candidate centres for each asset; you score them from the documents."] + _header(ctx)
    lines += ["", f"Documents ({min(len(docs), MAX_DOCS)}), newest first:", doc_blob(docs), "",
              f"For EACH asset, the candidates are shifts of the centre at the longest horizon, in multiples of the horizon sd: {SHIFT_GRID}.",
              "Give each candidate an integer score from 0 (implausible given the documents) to 9 (most plausible). Score all 9 candidates. If the documents say nothing, score 0.0 highest and the rest by distance.",
              "", "Reply with JSON only, no prose:",
              '{"assets": {"<asset>": {"scores": {"-1.5": <int>, "-1.0": <int>, "-0.5": <int>, "-0.25": <int>, "0.0": <int>, "0.25": <int>, "0.5": <int>, "1.0": <int>, "1.5": <int>}}}}']
    return "\n".join(lines)


PROMPTS = {"knobs": prompt_knobs, "scenarios": prompt_scenarios, "quantiles": prompt_quantiles, "rank": prompt_rank, "f4event": lambda ctx, docs: prompt_f4event(ctx, docs)}


# ------------------------------------------------------------------ B2: screen -> extract -> vote (with abstention)
BOILERPLATE = (".gov means it", "ensures that you are connecting", "Share sensitive information",
               "Transmission of material in this", "Skip to main content", "official website")
SCREEN_HEAD_CHARS, EXTRACT_CHARS, B2_MAX_KEEP, B2_VOTES = 900, 9000, 4, 5


def clean_doc(text: str, min_words: int = 12) -> str:
    """Title line plus prose blocks only: the corpus is web-scraped and opens with navigation menus."""
    blocks = [b.strip() for b in text.replace("\r", "").split("\n\n")]
    keep = []
    for i, b in enumerate(blocks):
        if not b or any(s in b for s in BOILERPLATE):
            continue
        if i == 0 or len(b.split()) >= min_words:
            keep.append(" ".join(b.split()))
    return "\n\n".join(keep)


def prompt_screen(ctx: dict, docs: list[dict]) -> str:
    lines = ["You are triaging dated documents for a probabilistic forecast."] + _header(ctx)
    lines += ["", "Rate EACH document by how much it bears on how the target assets will move over the horizon:",
              "  0 = irrelevant or routine background; 1 = weak or indirect; 2 = directly relevant;",
              "  3 = decisive (a policy action, a shock, an explicit statement about the target).",
              "Judge from the text, not from the document type. One short reason each.", "", "Documents (newest first, first lines only):"]
    for d in docs[:MAX_DOCS]:
        lines += [f"--- {d['doc_id']} ({d['timestamp']}, {d['doc_type']}) ---", clean_doc(d["text"])[:SCREEN_HEAD_CHARS], ""]
    lines += ["Reply with JSON only, no prose:", '{"documents": {"<doc_id>": {"relevance": <0-3>, "reason": "<short>"}}}']
    return "\n".join(lines)


def prompt_extract(ctx: dict, doc: dict) -> str:
    lines = ["You are collecting evidence from ONE dated document for a probabilistic forecast."] + _header(ctx)
    lines += ["", f"--- {doc['doc_id']} ({doc['timestamp']}, {doc['doc_type']}) ---", clean_doc(doc["text"])[:EXTRACT_CHARS], "",
              "Copy up to 3 passages VERBATIM (each at most 60 words) that bear on how the target assets will move, or on stress in markets.",
              "Then list dated facts in two groups: `announced` = actions or numbers already public as of the document's date (already in prices),",
              "`expected` = actions the document says are likely LATER, with the date or meeting if given.",
              "Rate severity: 0 routine, 1 elevated uncertainty, 2 concrete adverse event or emergency measure, 3 crisis.",
              "", "Reply with JSON only, no prose:",
              '{"passages": ["<verbatim>"], "announced": ["<fact (date)>"], "expected": ["<fact (when)>"], "severity": <0-3>}']
    return "\n".join(lines)


def prompt_judge(ctx: dict, evidence: str) -> str:
    lines = ["You are deciding whether dated evidence should move a statistical forecast, and how much."] + _header(ctx)
    lines += ["", "Evidence extracted from the documents (verbatim passages, announced facts, expected facts):", evidence, "",
              "Rules. A fact that was public before the as-of date is ALREADY in the as-of level; it justifies a move only if the",
              "documents argue the market has not yet adjusted, or if a further action is expected inside the horizon. When the",
              "evidence is mixed or silent, answer flat with magnitude 0. Be as willing to say flat as to call a direction.",
              "For EACH asset:",
              "  direction : \"down\" | \"flat\" | \"up\" over the longest horizon relative to the statistical forecast's centre.",
              "  magnitude : 0.25 | 0.5 | 1.0 (multiples of the horizon sd; 0 if flat).",
              "  width     : 0.75 | 1.0 | 1.25 | 1.5 (multiplier on the statistical spread: >1 if uncertainty is elevated).",
              "  because   : one sentence citing a doc_id and saying whether the driver is announced or expected.",
              "", "Reply with JSON only, no prose:",
              '{"assets": {"<asset>": {"direction": "<down|flat|up>", "magnitude": <float>, "width": <float>, "because": "<...>"}}}']
    return "\n".join(lines)


def prompt_f4event(ctx: dict, docs: list[dict]) -> str:
    """E10: is the decisive event still ahead (keep the shock width) or already resolved / priced (take it back)?"""
    lines = ["You are assessing the SPREAD a statistical forecast should have, not its direction."] + _header(ctx)
    lines += ["The forecast is a distribution centred near today's value with the standard deviation shown; its 90% band is +-1.645 sd. It has been widened because cards of this kind are built around shocks. Your job: from the documents alone, decide whether the decisive event they describe is still AHEAD within the horizon and unresolved, so that a realized move beyond 2 sd is more likely than usual (class shock_ahead); or whether the decisive move has already happened / the decision is public and priced (resolved_or_priced); or whether the documents describe a routine period with nothing decisive inside the horizon (routine). Do not use any knowledge of what happened after the as-of date; cite the doc_ids that decide the class.",
              "", f"Documents ({min(len(docs), MAX_DOCS)}), newest first:", doc_blob_clean(docs), "",
              'Reply with JSON only:',
              '{"class": "shock_ahead" | "resolved_or_priced" | "routine", "p_beyond_2sd": <one of 0.2, 0.35, 0.5, 0.65, 0.8>, "decisive_event": "<short>", "when": "before_asof" | "inside_horizon" | "after_horizon" | "none", "evidence": ["<doc_id>", ...], "because": "<one sentence>"}']
    return "\n".join(lines)

