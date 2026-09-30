#!/usr/bin/env python
"""Text-blind post-processing rules applied to the engine's draws inside the submission image.

## Executive summary (read this first)

Four rules: two measured in `experiments/E6-momentum-memory` (addenda T9, T10) and stacked in
`experiments/B1-integration`, (3) `calm_narrow` from `experiments/E9-calm-narrowing`, and (4) `post_fomc_unwiden` from `experiments/E10-resolved-gate`
(a scheduled statement within `k` business days before the as-of takes the F4 width prior back to 1.0); which narrows a leg of at most
`max_h` days by `factor` when the asset's recent-volatility ratio (sd of the last 20 steps over the trailing-300 sd) is
below `thr`; it reads only the panel. The first two: (1) `fomc_narrow`: a cell whose leg (as-of, as-of + h business days]
contains no scheduled FOMC statement is narrowed by the factor (legs above `max_h` always contain
one and are left alone); the calendar is `data/fomc_dates.csv` (statement dates 1997–2026 from the
Fed's site, plus the published 2025–2026 schedule; a `scheduled` flag excludes the 14 intermeeting
statements, which the as-of could not know about) — a static public table, allowed by the artifact
policy. (2) `stress_skew`: a corpus that holds a `corporate_8k` document (read from
`corpus_index.json`, the organizer's dated manifest — the document text is not read) widens every
asset by `width` and stretches the downside of yields and the equity factor by `skew`. Both read
nothing after the as-of date. `scripts/text_grid.py` holds the same mechanisms for offline scoring.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
STRESS_DOWN = ("UST", "MKT")
DEFAULT_RULES = {"fomc_narrow": {"factor": 1.15, "max_h": 63}, "stress_skew": {"types": ["corporate_8k"], "width": 1.25, "skew": 0.25}}


def fomc_dates() -> list[pd.Timestamp]:
    """Scheduled statement dates only: the `scheduled` column (0 = intermeeting / emergency statement) keeps
    the rule from knowing, at the as-of, that an unscheduled statement was about to come (look-ahead)."""
    for p in (HERE / "fomc_dates.csv", HERE.parent / "data" / "fomc_dates.csv"):
        if p.exists():
            df = pd.read_csv(p)
            if "scheduled" in df.columns:
                df = df[df["scheduled"].astype(int) == 1]
            return sorted(pd.to_datetime(df["date"]))
    return []


def apply_rules(X: np.ndarray, cells: list, unit_dir: pathlib.Path, asof: str, rules: dict | None) -> tuple[np.ndarray, list[str]]:
    """Return (draws, notes). `rules` = {"fomc_narrow": {...}, "stress_skew": {...}}; missing keys = off."""
    rules = rules or {}
    notes = []
    X = X.copy()
    med = np.median(X, axis=0)
    fn = rules.get("fomc_narrow")
    if fn:
        cal = fomc_dates()
        a0 = pd.Timestamp(asof)
        for i, (a, h) in enumerate(cells):
            if h > fn.get("max_h", 63) or not cal:
                continue
            end = a0 + pd.tseries.offsets.BDay(int(h))
            if not any(a0 < d <= end for d in cal):
                X[:, i] = med[i] + (X[:, i] - med[i]) / float(fn.get("factor", 1.15))
                notes.append(f"fomc_narrow: {a}@{h} has no scheduled statement in its leg -> spread / {fn.get('factor', 1.15)}")
    cn = rules.get("calm_narrow")
    if cn:
        # E9: a calm leg (recent-volatility ratio below `thr`) of at most `max_h` days is narrowed by `factor`; the
        # board's mean-of-ratios metric rewards a narrower spread on calm realizations (paper/theory_composite.py)
        from engine import _full_series
        try:
            import tomllib
            fam = str(tomllib.loads((unit_dir / "card.toml").read_text()).get("metadata", {}).get("category", "")).split("-")[-1]
        except Exception:
            fam = ""
        if fam in set(cn.get("families", ["F1", "F3", "F4"])):
            a0 = pd.Timestamp(asof)
            for a in sorted(set(x for x, _ in cells)):
                s_full, stem = _full_series(unit_dir, a, a0)
                x = np.log(s_full.values) if "fx" in stem else s_full.values
                d = np.diff(x)
                if len(d) < 25:
                    continue
                long = d[-int(cn.get("window", 300)):]
                vr = float(np.std(d[-int(cn.get("short", 20)):]) / np.std(long)) if np.std(long) > 0 else 1.0
                if vr >= float(cn.get("thr", 0.8)):
                    continue
                fac = cn.get("factor", 1.3)
                fac = float(fac.get(fam, 1.3)) if isinstance(fac, dict) else float(fac)   # scalar or per family
                for i, (aa, h) in enumerate(cells):
                    if aa == a and h <= int(cn.get("max_h", 63)):
                        X[:, i] = med[i] + (X[:, i] - med[i]) / fac
                        notes.append(f"calm_narrow: {a}@{h} vol ratio {vr:.2f} < {cn.get('thr', 0.8)} -> spread / {fac}")
    pf = rules.get("post_fomc_unwiden")
    if pf:
        # E10: a scheduled FOMC statement within the last `k` business days (the decision is public) -> the family's
        # width prior is taken back to `width_to` on every cell (fan and momentum untouched); F4 by default
        try:
            import tomllib
            fam = str(tomllib.loads((unit_dir / "card.toml").read_text()).get("metadata", {}).get("category", "")).split("-")[-1]
        except Exception:
            fam = ""
        if fam in set(pf.get("families", ["F4"])):
            cal = fomc_dates()
            a0 = pd.Timestamp(asof)
            k = int(pf.get("k", 10))
            if any(a0 - pd.tseries.offsets.BDay(k) <= d <= a0 for d in cal):
                f = float(pf.get("width_from", 1.5)) / float(pf.get("width_to", 1.0))
                for i, (a, h) in enumerate(cells):
                    X[:, i] = med[i] + (X[:, i] - med[i]) / f
                notes.append(f"post_fomc_unwiden: a scheduled statement within {k} BD before the as-of -> spread / {f:.3g} on every cell")
    ss = rules.get("stress_skew")
    if ss:
        idx = unit_dir / "text" / "corpus_index.json"
        try:
            docs = json.loads(idx.read_text()).get("documents", []) if idx.exists() else []
        except ValueError:
            docs = []
        hit = [d.get("doc_id") for d in docs if d.get("doc_type") in set(ss.get("types", ["corporate_8k"])) and str(d.get("timestamp", ""))[:10] <= asof]
        if hit:
            w, k = float(ss.get("width", 1.25)), float(ss.get("skew", 0.25))
            for i, (a, h) in enumerate(cells):
                x = X[:, i] - med[i]
                if k and any(a.startswith(s) for s in STRESS_DOWN):
                    x = np.where(x < 0, x * (1 + k), x * (1 - k))
                X[:, i] = med[i] + x * w
            notes.append(f"stress_skew: stress document(s) {hit} in the corpus -> spread x{w}, downside x(1+{k}) on yields/MKT")
    return X, notes
