#!/usr/bin/env python
"""M0 replica — the organizer's text-blind joint Gaussian random walk, rebuilt from docs/M0-BASELINE.md.

## Executive summary (read this first)

Every official score is `our composite / M0's composite` per card. This script rebuilds M0's 500
draws for any card whose panels we hold, following the published procedure step by step (trailing
300 observations, gap rule, date-aligned covariance, path covariance min(s,g)·Sigma, crc32 seed,
cells sorted by asset then horizon). With a locally assembled realized file the ratio
`arm / m0` is then on the same scale as the leaderboard, which the reference CLI (about 29% worse
than M0) never was. It reads only the card and the panels: no corpus, no model, no network.

Usage (one unit, same verb shape as the reference CLI):

    python scripts/m0_baseline.py --unit units/t2-EXAMPLE-ust-curve-1m --out out/m0/t2-EXAMPLE-ust-curve-1m/forecast.parquet
    python scripts/m0_baseline.py --all --out-root out/m0      # every unit
    python scripts/m0_baseline.py --unit units/t2-EXAMPLE-ust-curve-1m --check   # print the §4 worked-example numbers

Deviations from the sealed generator that this script cannot close (M0-BASELINE.md §6): the target
date of a cell is sealed, so the horizon-to-steps rule uses `forecast_spec.json`'s
`observation_periods` on monthly panels (this reproduces the four published step counts) and the
declared horizon elsewhere; the cell order is the published rule (asset id, then horizon), which
holds on 100 of 103 answered cards.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import tomllib
import zlib

import numpy as np
import pandas as pd

N_DRAWS = 500
WINDOW = 300
LOG_RETURN_TRIPWIRE = 0.2
RATIONALE = "forecast_rationale.md"
# docs/M0-BASELINE.md §3.7: the four released monthly-panel cards and the panel steps M0 uses.
# The sealed target date is what M0 counts from; `observation_periods` reproduces three of the
# four, but on sahm-watch it gives 7/8 where the organizer states 8/9, so the published table wins.
PUBLISHED_MONTHLY_STEPS = {
    "t2-F1-cpi-glidepath-2023": {140: 8, 160: 9},
    "t2-F1-sahm-watch-2024": {145: 8, 165: 9},
    "t2-F4-covid-nfp-2020": {21: 2},
    "t2-F4-cpi-vintage-2022": {21: 2},
}


def panel_files(unit_dir: pathlib.Path) -> list[pathlib.Path]:
    """The unit's panel parquets: `panels/` first (a staged unit ships them there — the contract
    passes `--panels /input/panels/`), else the unit root (the shipped practice units). A unit
    root that holds a `panels/` subdirectory and no parquet of its own is the official layout;
    reading only the root crashed every unit of submit-dev-03 (2026-09-27)."""
    for d in (unit_dir / "panels", unit_dir):
        found = sorted(d.glob("*.parquet")) if d.is_dir() else []
        if found:
            return found
    return []


_panel_files = panel_files


def trailing_window(unit_dir: pathlib.Path, asset: str, asof: pd.Timestamp) -> pd.Series:
    """§3.1: the first parquet (sorted name) holding the asset; rows <= as-of; the last 300."""
    for p in _panel_files(unit_dir):
        df = pd.read_parquet(p)
        col = "asset" if "asset" in df.columns else ("asset_id" if "asset_id" in df.columns else None)
        if col is None or "date" not in df.columns:
            continue
        g = df[df[col] == asset]
        if g.empty:
            continue
        s = g.assign(date=pd.to_datetime(g["date"])).sort_values("date")
        s = s[s["date"] <= asof]
        s = s.set_index("date")["value"].astype(float)
        return s.iloc[-WINDOW:]
    raise SystemExit(f"{asset}: no panel in {unit_dir} carries this asset")


def steps_from_window(win: pd.Series, target_type: str) -> tuple[pd.Series, float, list[float]]:
    """§3.2 + §3.3: step series indexed by the date each step ends on, hole steps dropped.
    Returns (steps, median spacing in days, non-hole intervals in days)."""
    dates = win.index
    if len(dates) < 2:
        return pd.Series(dtype=float), float("nan"), []
    gaps = np.diff(dates.values).astype("timedelta64[D]").astype(float)  # intervals rows i-1 -> i
    med = float(np.median(gaps))
    hole = max(10.0 * med, 5.0)
    if target_type == "log_return":
        if float(np.median(np.abs(win.values))) >= LOG_RETURN_TRIPWIRE:
            raise SystemExit("log_return panel does not look like per-step returns (tripwire, §3.2)")
        vals = win.values[1:]  # row 0 has no preceding interval (§3.2)
    else:
        vals = np.diff(win.values)
    keep = gaps <= hole
    steps = pd.Series(vals[keep], index=dates[1:][keep])
    return steps, med, [float(g) for g in gaps[keep]]


def horizon_steps(win: pd.Series, horizon: int, spacing_days: float | None, target: pd.Timestamp | None,
                  period: str | None) -> int:
    """§3.7: the declared horizon unless the counted steps differ by 2x or more."""
    if len(win) < 3 or spacing_days is None or not np.isfinite(spacing_days):
        return horizon
    last = win.index[-1]
    if spacing_days > 20:  # monthly panel: count calendar months to the observation period
        if period is None:
            return horizon
        y1, m1 = (int(x) for x in period.split("-")[:2])
        steps = 12 * (y1 - last.year) + (m1 - last.month)
    else:
        if target is None:
            return horizon
        steps = int(round((target - last).days / spacing_days))
    if steps <= 0:
        return horizon
    ratio = max(steps, horizon) / max(min(steps, horizon), 1)
    return steps if ratio >= 2 else horizon


def build(unit_dir: pathlib.Path, *, check: bool = False) -> tuple[pd.DataFrame, dict]:
    card = tomllib.loads((unit_dir / "card.toml").read_text())
    tgt = card["targets"]
    unit_id = card["task"]["id"]
    asof = pd.Timestamp(card["provenance"]["data_cutoff"])
    target_type = tgt.get("target_type", "level")
    assets = sorted(tgt["asset_ids"])          # §3.9: cells sorted by asset id, then horizon
    horizons = sorted(int(h) for h in tgt["horizons"])
    periods: dict[int, str] = {}
    spec_path = unit_dir / "forecast_spec.json"
    if spec_path.exists():
        spec = json.loads(spec_path.read_text())
        st = spec.get("targets", {})
        per = st.get("observation_periods")
        if per and len(per) == len(st.get("horizons", [])):
            periods = {int(h): str(p) for h, p in zip(st["horizons"], per)}

    wins, steps, spacing = {}, {}, {}
    for a in assets:
        w = trailing_window(unit_dir, a, asof)
        s, med, gaps = steps_from_window(w, target_type)
        wins[a], steps[a] = w, s
        spacing[a] = float(np.mean(gaps)) if gaps else None   # §3.7.2: mean over non-hole intervals

    frame = pd.concat([steps[a].rename(a) for a in assets], axis=1, join="inner").dropna()  # §3.4
    if frame.empty:
        raise SystemExit(f"{unit_id}: no aligned step dates")
    mu = frame.mean(axis=0).to_dict()
    sigma = np.atleast_2d(np.cov(frame.values, rowvar=False, ddof=1))            # §3.5

    anchor = {a: (0.0 if target_type == "log_return" else float(wins[a].iloc[-1])) for a in assets}  # §3.6
    cells = [(a, h) for a in assets for h in horizons]
    s_cell = []
    for a, h in cells:
        target = asof + pd.tseries.offsets.BDay(h)
        published = PUBLISHED_MONTHLY_STEPS.get(unit_id, {}).get(h)
        s_cell.append(published if published is not None
                      else horizon_steps(wins[a], h, spacing[a], target, periods.get(h)))
    d = len(cells)
    ai = {a: i for i, a in enumerate(assets)}
    mean = np.array([anchor[a] + s * mu[a] for (a, _), s in zip(cells, s_cell)])
    cov = np.empty((d, d))
    for i, ((a, _), si) in enumerate(zip(cells, s_cell)):
        for j, ((b, _), sj) in enumerate(zip(cells, s_cell)):
            cov[i, j] = min(si, sj) * sigma[ai[a], ai[b]]                       # §3.8
    cov = cov + 1e-10 * np.eye(d)
    try:
        chol = np.linalg.cholesky(cov + 1e-9 * np.eye(d))
    except np.linalg.LinAlgError:
        chol = np.linalg.cholesky(np.diag(np.diag(cov + 1e-9 * np.eye(d))))
    seed = zlib.crc32(unit_id.encode()) & 0x7FFFFFFF                           # §3.9
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((N_DRAWS, d))
    samples = mean + z @ chol.T

    rows = [{"draw": k, "asset": a, "horizon": h, "value": float(samples[k, i])}
            for k in range(N_DRAWS) for i, (a, h) in enumerate(cells)]
    meta = {
        "unit_id": unit_id, "asof": str(asof.date()), "representation": "samples",
        "asset_ids": list(tgt["asset_ids"]), "horizons": [int(h) for h in tgt["horizons"]],
        "n_draws": N_DRAWS, "target": target_type,
        "rationale": {"file": RATIONALE, "method": "M0 replica: text-blind joint gaussian random walk (docs/M0-BASELINE.md)"},
    }
    info = {"seed": seed, "cells": cells, "steps": s_cell, "window_rows": {a: int(len(wins[a])) for a in assets},
            "aligned_rows": int(len(frame)), "spacing_days": spacing, "mu": mu, "anchor": anchor}
    if check:
        print(json.dumps(info, indent=1, default=str))
    return pd.DataFrame(rows), {"meta": meta, "info": info, "asof": str(asof.date()), "unit_id": unit_id}


def write(unit_dir: pathlib.Path, out: pathlib.Path, check: bool = False) -> None:
    df, extra = build(unit_dir, check=check)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    (out.parent / "forecast_meta.json").write_text(json.dumps(extra["meta"], indent=2) + "\n")
    info = extra["info"]
    (out.parent / RATIONALE).write_text(
        f"# {extra['unit_id']} — M0 replica (text-blind)\n\n"
        f"As of {extra['asof']}: joint Gaussian random walk on the trailing {WINDOW} observations, "
        f"drift = mean step, covariance = sample covariance of date-aligned steps, path covariance "
        f"min(s,g)·Sigma, {N_DRAWS} draws, seed crc32(unit id) = {info['seed']}. "
        f"Steps per cell: {dict(zip([f'{a}@{h}' for a, h in info['cells']], info['steps']))}. "
        "No document was read; this is the denominator of the official score, rebuilt from "
        "docs/M0-BASELINE.md, not a forecast that uses the text.\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--unit", type=pathlib.Path, help="one unit directory")
    ap.add_argument("--out", type=pathlib.Path, help="forecast.parquet path (sidecars written beside it)")
    ap.add_argument("--all", action="store_true", help="every unit under --units-root")
    ap.add_argument("--units-root", type=pathlib.Path, default=pathlib.Path("units"))
    ap.add_argument("--out-root", type=pathlib.Path, default=pathlib.Path("out/m0"))
    ap.add_argument("--check", action="store_true", help="print the construction details (§4 numbers)")
    a = ap.parse_args(argv)
    if a.all:
        units = sorted(p for p in a.units_root.iterdir() if (p / "card.toml").exists())
        for u in units:
            write(u, a.out_root / u.name / "forecast.parquet")
        print(f"wrote {len(units)} units under {a.out_root}")
        return 0
    if not a.unit:
        ap.error("--unit or --all")
    out = a.out or (a.out_root / a.unit.name / "forecast.parquet")
    write(a.unit, out, check=a.check)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
