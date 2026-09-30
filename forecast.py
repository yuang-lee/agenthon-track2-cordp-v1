"""## Executive summary (read this first)

Run the frozen v1 six-call median corrector through the injected House endpoint.
Keep seed 0, 4000 joint draws and full-strength translation. Invalid final replies
retain the statistical forecast. Emit only the three forecast contract files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tomllib

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "pipelines"))
from cordp import baseline, run_reading, translate
import cordp
from engine import write

# Preserve the archived prompts and valid-output numerics, while treating malformed
# JSON element types as invalid corrections rather than crashing the container.
_archived_correction_values = cordp.correction_values


def checked_correction_values(parsed, context, docs):
    try:
        return _archived_correction_values(parsed, context, docs)
    except (AttributeError, OverflowError) as exc:
        raise ValueError("malformed correction element") from exc


cordp.correction_values = checked_correction_values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("verb", nargs="?", choices=["forecast"], default="forecast")
    parser.add_argument("--panels", type=Path, required=True)
    parser.add_argument("--text", type=Path, required=True)
    parser.add_argument("--asof", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    for name in ("MODEL_ENDPOINT", "MODEL_NAME", "MODEL_TOKEN"):
        if not os.environ.get(name, "").strip():
            parser.error(f"organizer-injected {name} is required")
    if not args.out.is_absolute() or args.out.name != "forecast.parquet":
        parser.error("--out must be an absolute path ending in forecast.parquet")
    unit = next((p for p in (args.panels, args.panels.parent)
                 if (p / "card.toml").is_file()), None)
    if unit is None or args.text.resolve() != (unit / "text").resolve():
        parser.error("panels and text must belong to the same mounted unit")
    cutoff = str(tomllib.loads((unit / "card.toml").read_text())["provenance"]["data_cutoff"])[:10]
    if args.asof[:10] != cutoff:
        parser.error("--asof disagrees with the card data_cutoff")
    freeze = json.loads((ROOT / "freeze.json").read_text())
    for name, digest in freeze["file_sha256"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"frozen source changed: {name}")
    os.environ["CORDP_JSON_MODE"] = "0"
    samples, cells, info = baseline(unit, 0)
    ledger = run_reading(unit, 0)
    revised = ledger["revised_medians"].get("agentic")
    shift = np.zeros(len(cells))
    if revised is not None:
        shifted = translate(samples, revised, 1.0)
        shift = np.asarray(revised) - np.median(samples, axis=0)
        samples = shifted
    write(samples, cells, info, args.out)
    meta_path = args.out.parent / "forecast_meta.json"
    meta = json.loads(meta_path.read_text())
    meta["rationale"]["method"] = "Frozen v1 dev-10 joint samples plus House LLM median translation, strength 1."
    meta["reasoning_applied"] = bool(np.any(shift != 0))
    meta["reasoning_skipped_reason"] = "invalid final correction" if revised is None else ""
    meta["house_requests"] = ledger["request_count"]
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    # Do not serialize endpoint URLs, credentials, raw errors, or proxy details.
    details = {"asof": cutoff, "seed": 0, "n_draws": len(samples),
               "prompt_version": 1, "strength": 1, "model_alias": os.environ["MODEL_NAME"],
               "house_requests": ledger["request_count"], "baseline_config": info["cfg"],
               "baseline_rule_notes": info.get("rule_notes", []),
               "cells": ledger["context"]["cells"], "shifts": shift.tolist(),
               "fallback": revised is None,
               "llm_proposal_unverified": ledger["calls"]["agentic"].get("parsed")}
    (args.out.parent / "forecast_rationale.md").write_text(
        "## Executive summary (read this first)\n\n"
        "The frozen statistical forecast is corrected using dated corpus evidence. "
        "Six calls include two comparison controls and the extractor, analyst, reviewer, "
        "and final corrector. Only the final corrector changes the forecast. "
        "Each marginal is translated; joint residual samples stay unchanged. "
        "Invalid final replies retain the baseline. LLM explanations below are "
        "unverified model statements, not established facts.\n\n"
        + json.dumps(details, indent=2, ensure_ascii=False) + "\n")
    print(f"Wrote {len(samples)} joint draws; House requests={ledger['request_count']}; fallback={revised is None}")


if __name__ == "__main__":
    main()
