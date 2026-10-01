#!/usr/bin/env python
"""The submission `forecast` verb: E-best (pipelines/ebest.json), text-blind.

## Executive summary (read this first)

The harness runs `forecast --panels /input/panels/ --text /input/text/ --asof <date> --out
/output/forecast.parquet` with the unit directory mounted at /input. This wrapper finds the unit
directory (the panels path itself, or its parent when the parquets sit at the unit root as the
shipped units do), loads the engine config `ebest.json` next to this file, simulates, and writes
exactly the three contract files beside `--out` — nothing else may appear in /output (gate g0).
`--text` is accepted and ignored: this arm reads no document. `--seed` and `--n-draws` are
accepted; the defaults reproduce the archived local run (`experiments/E3-classical-centre`).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from engine import simulate, write  # noqa: E402
from rules import apply_rules  # noqa: E402
import os  # noqa: E402


def unit_dir_of(panels: pathlib.Path, card: pathlib.Path | None) -> pathlib.Path:
    if card is not None:
        return card.resolve().parent
    if panels.name == "panels" and (panels.parent / "card.toml").exists():
        return panels.parent                     # the contract layout: /input/panels/ under the unit root
    for cand in (panels, panels.parent):
        if (cand / "card.toml").exists():
            return cand
    raise SystemExit(f"card.toml not found in {panels} or {panels.parent}; pass --card")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="forecast", description=__doc__.splitlines()[0])
    ap.add_argument("verb", nargs="?", default="forecast", help="tolerated leading verb")
    ap.add_argument("--panels", type=pathlib.Path, required=True)
    ap.add_argument("--text", type=pathlib.Path, default=None)
    ap.add_argument("--asof", required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--card", type=pathlib.Path, default=None)
    ap.add_argument("--n-draws", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--calibration", choices=["none", "gls-common75"],
                    default=os.environ.get("AGENT_CALIBRATION", "none"))
    ap.add_argument("--cfg", type=pathlib.Path, default=pathlib.Path(os.environ.get("AGENT_CFG", str(HERE / "ebest.json"))))
    a = ap.parse_args(argv)
    cfg = json.loads(a.cfg.read_text())
    if a.n_draws:
        cfg["n_draws"] = max(int(a.n_draws), 200)
    unit = unit_dir_of(a.panels, a.card)
    samples, cells, info = simulate(unit, cfg, a.seed)
    samples, notes = apply_rules(samples, cells, unit, info["asof"], cfg.get("rules"))
    info["rule_notes"] = notes
    text_cfg = cfg.get("text")
    if text_cfg and os.environ.get("MODEL_ENDPOINT", "").strip():
        from text_layer import apply_text_layer
        try:
            samples, ledger = apply_text_layer(samples, cells, info, unit, text_cfg)
        except Exception as exc:   # never let the text layer take the unit down
            ledger = {"reasoning_applied": False, "reasoning_skipped_reason": f"text layer failed ({type(exc).__name__})", "calls": 0}
        info["text_ledger"] = ledger
    elif text_cfg:
        info["text_ledger"] = {"reasoning_applied": False, "reasoning_skipped_reason": "MODEL_ENDPOINT is unset", "calls": 0}
    if a.calibration == "gls-common75":
        import pandas as pd
        from engine import _full_series
        from numerical_combinations import conditional_common
        positive = []
        for asset in info["asset_ids"]:
            _, stem = _full_series(unit, asset, pd.Timestamp(info["asof"]))
            if info["target"] == "level" and "fx" in stem.lower():
                positive.extend(i for i, cell in enumerate(cells) if cell[0] == asset)
        samples, calibration = conditional_common(samples, .75, positive)
        info["cfg"]["calibration"] = {"method": a.calibration, "scale": .75,
            "applied": calibration["applied"], "fallback": calibration.get("fallback")}
        info["rule_notes"].append("Covariance-weighted raw common component: " + json.dumps(calibration, sort_keys=True))
    if info["asof"] != a.asof[:10]:
        print(f"note: card data_cutoff {info['asof']} differs from --asof {a.asof}; the card's date was used")
    write(samples, cells, info, a.out)
    print(f"wrote {a.out.name}, forecast_meta.json and forecast_rationale.md to {a.out.parent}")
    print(f"  {len(info['asset_ids'])} asset(s) x {len(info['horizons'])} horizon(s), {samples.shape[0]} draws")
    return 0


if __name__ == "__main__":
    sys.exit(main())
