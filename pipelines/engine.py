#!/usr/bin/env python
"""Text-blind engine: M0's construction with the old champion's options as switches (E1/E2).

## Executive summary (read this first)

One simulator, one config. With the default config it is the M0 procedure of
`docs/M0-BASELINE.md` (trailing 300 observations, drift = mean step, sample covariance,
Gaussian path, cells sorted by asset then horizon) drawn as a step-by-step path so that the old
engine's options can be switched on one at a time: Student-t shocks, an EWMA-blended scale over
the full history, a per-family width prior, a sub-sqrt fan on F3/F4, multiplicative FX. Experiment
E1 sweeps these switches; E2 is the old champion's switches plus M0's drift. Reads only the card
and the panels.

    cfg keys (defaults = M0):
      window      300   rows of history for scale and correlation (0 = full history)
      drift       1.0   multiplier on the trailing-300 mean step (0 = driftless walk); scalar or per family {"F3": 0.5}
      scale       cov   'cov' = sd of the window's steps; 'ewma' = 0.5*full std + 0.5*EWMA(0.97) std
      shock       gauss 'gauss' | 't4' (Student-t, unit variance)
      width       {}    per-family multiplier on the spread, e.g. {"F4": 1.5}
      fan         {}    per-family fan exponent, e.g. {"F3": 0.3, "F4": 0.3}; pivot 63 steps; daily panels only
      fx_log      False level targets on FX panels simulated in log space
      n_draws     4000

Usage:
    python pipelines/engine.py --unit units/t2-EXAMPLE-ust-curve-1m --out out/e/t2-EXAMPLE-ust-curve-1m/forecast.parquet [--cfg '{"shock":"t4"}'] [--seed 0]
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

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
from m0_baseline import (PUBLISHED_MONTHLY_STEPS, RATIONALE, horizon_steps, panel_files, steps_from_window,  # noqa: E402
                         trailing_window)

DEFAULT = {"window": 300, "drift": 1.0, "scale": "cov", "shock": "gauss", "width": {}, "fan": {},
           "fx_log": False, "n_draws": 4000}
EWMA_LAM, EWMA_MIX, FAN_PIVOT, T_DF = 0.97, 0.5, 63, 4.0


def _full_series(unit_dir: pathlib.Path, asset: str, asof: pd.Timestamp) -> tuple[pd.Series, str]:
    """All rows <= as-of for the asset, and the panel file's stem (to detect FX)."""
    for p in panel_files(unit_dir):
        df = pd.read_parquet(p)
        col = "asset" if "asset" in df.columns else ("asset_id" if "asset_id" in df.columns else None)
        if col is None or "date" not in df.columns:
            continue
        g = df[df[col] == asset]
        if g.empty:
            continue
        s = g.assign(date=pd.to_datetime(g["date"])).sort_values("date")
        s = s[s["date"] <= asof].set_index("date")["value"].astype(float)
        return s, p.stem
    raise SystemExit(f"{asset}: no panel in {unit_dir} carries this asset")


def _ewma_std(x: np.ndarray) -> float:
    w = EWMA_LAM ** np.arange(len(x) - 1, -1, -1)
    return float(np.sqrt(np.average(x ** 2, weights=w)))


def classical_centre(levels: np.ndarray, steps: np.ndarray, target_type: str, smax: int, model: str) -> np.ndarray:
    """Mean path for steps 1..smax from a statsforecast model fitted on the trailing window:
    on the levels for a `level` target, on the per-step returns (then cumulated) for `log_return`."""
    from statsforecast import StatsForecast
    from statsforecast.models import AutoARIMA, AutoETS, AutoTheta, Naive

    # A log_return target is a cumulative return, so the model is fitted on the cumulative index
    # of the per-step returns (a level-like path) and its forecast is taken relative to the last
    # value; fitting the returns themselves would make "naive" repeat the last day's return.
    y = np.asarray(levels if target_type == "level" else np.cumsum(steps), dtype=float)
    y = y[np.isfinite(y)]
    m = {"theta": AutoTheta(), "arima": AutoARIMA(), "ets": AutoETS(), "naive": Naive()}[model]
    df = pd.DataFrame({"unique_id": "s", "ds": np.arange(len(y)), "y": y})
    try:
        fc = StatsForecast(models=[m], freq=1, n_jobs=1).forecast(df=df, h=smax)
        path = fc.iloc[:, -1].to_numpy(dtype=float)
    except Exception:   # a model that fails to fit falls back to the naive level
        path = np.repeat(y[-1], smax)
    if target_type != "level":
        path = path - y[-1]
    return path


def simulate(unit_dir: pathlib.Path, cfg: dict, seed: int) -> tuple[np.ndarray, list[tuple[str, int]], dict]:
    """Return (samples[n_draws, cells], cells, info); cells sorted by asset id then horizon."""
    c = {**DEFAULT, **cfg}
    card = tomllib.loads((unit_dir / "card.toml").read_text())
    tgt, unit_id = card["targets"], card["task"]["id"]
    asof = pd.Timestamp(card["provenance"]["data_cutoff"])
    target_type = tgt.get("target_type", "level")
    fam = str(card.get("metadata", {}).get("category", "")).split("-")[-1]
    assets = sorted(tgt["asset_ids"])
    horizons = sorted(int(h) for h in tgt["horizons"])
    periods = {}
    spec_path = unit_dir / "forecast_spec.json"
    if spec_path.exists():
        st = json.loads(spec_path.read_text()).get("targets", {})
        per = st.get("observation_periods")
        if per and len(per) == len(st.get("horizons", [])):
            periods = {int(h): str(p) for h, p in zip(st["horizons"], per)}

    full, win300, steps300, stepsW, spacing, use_log = {}, {}, {}, {}, {}, {}
    for a in assets:
        s, stem = _full_series(unit_dir, a, asof)
        use_log[a] = bool(c["fx_log"]) and target_type == "level" and "fx" in stem.lower() and float(s.iloc[-1]) > 0
        src = np.log(s) if use_log[a] else s
        full[a] = src
        win300[a] = src.iloc[-300:]
        st3, _, gaps = steps_from_window(win300[a], target_type)
        steps300[a] = st3
        spacing[a] = float(np.mean(gaps)) if gaps else None
        w = src if int(c["window"]) == 0 else src.iloc[-int(c["window"]):]
        stepsW[a], _, _ = steps_from_window(w, target_type)

    dw = int(c.get("drift_window", 300) or 300)          # T1: rows of history behind the mean step (M0: 300)
    if dw != 300:
        steps300 = {}
        for a in assets:
            st_, _, _ = steps_from_window(full[a].iloc[-dw:], target_type)
            steps300[a] = st_
    f300 = pd.concat([steps300[a].rename(a) for a in assets], axis=1, join="inner").dropna()
    fW = pd.concat([stepsW[a].rename(a) for a in assets], axis=1, join="inner").dropna()
    drift = c["drift"].get(fam, 1.0) if isinstance(c["drift"], dict) else c["drift"]   # scalar or per family
    mu = {a: float(drift) * float(f300[a].mean()) for a in assets}                  # M0's drift, scaled
    lam_m = float(c.get("monthly_drift_lam", 0) or 0)   # >0: monthly panels use an EWMA mean step (recent prints) instead of the 25-year mean
    if lam_m:
        for a in assets:
            if spacing[a] is not None and spacing[a] > 20 and len(f300[a]) > 5:
                w = lam_m ** np.arange(len(f300[a]) - 1, -1, -1)
                mu[a] = float(drift) * float(np.average(f300[a].values, weights=w))
    lam = c.get("ewma_lam", EWMA_LAM); lam = lam.get(fam, EWMA_LAM) if isinstance(lam, dict) else lam       # T4: per family
    mix = c.get("ewma_mix", EWMA_MIX); mix = mix.get(fam, EWMA_MIX) if isinstance(mix, dict) else mix
    def _ewma(x, l):
        w = l ** np.arange(len(x) - 1, -1, -1)
        return float(np.sqrt(np.average(x ** 2, weights=w)))
    if c["scale"] == "ewma":
        sd = {a: (1 - mix) * float(stepsW[a].std(ddof=1)) + mix * _ewma(stepsW[a].values, lam) for a in assets}
    else:
        sd = {a: float(fW[a].std(ddof=1)) for a in assets}
    if c.get("monthly_scale") == "mad":                       # T8: robust scale on monthly panels (1.4826 x MAD)
        for a in assets:
            if spacing[a] is not None and spacing[a] > 20:
                x = stepsW[a].values
                sd[a] = float(1.4826 * np.median(np.abs(x - np.median(x))))
    n = len(assets)
    if n > 1:
        if c.get("corr_lam"):                                   # T5: EWMA correlation of the aligned steps
            w = float(c["corr_lam"]) ** np.arange(len(fW) - 1, -1, -1)
            xm = fW.values - np.average(fW.values, axis=0, weights=w)
            cov = (xm * w[:, None]).T @ xm / w.sum()
            dg = np.sqrt(np.clip(np.diag(cov), 1e-18, None))
            corr = cov / np.outer(dg, dg)
        else:
            corr = np.nan_to_num(fW.corr().values, nan=0.0)
        cs = float(c.get("corr_shrink", 0) or 0)                # T5: >0 toward independence, <0 toward one common factor
        if cs:
            target = np.eye(n) if cs > 0 else np.ones((n, n))
            corr = (1 - abs(cs)) * corr + abs(cs) * target
        np.fill_diagonal(corr, 1.0)
        w_, v_ = np.linalg.eigh(corr)
        chol = np.linalg.cholesky(v_ @ np.diag(np.clip(w_, 1e-6, None)) @ v_.T)
    else:
        chol = np.eye(1)

    anchor = {a: (0.0 if target_type == "log_return" else float(full[a].iloc[-1])) for a in assets}
    centre_cfg = c.get("centre", "m0")                      # 'm0' | 'theta' | ... | 'theta+m0' | per family {"F1": "theta"}
    centre_model = str(centre_cfg.get(fam, "m0") if isinstance(centre_cfg, dict) else centre_cfg)
    monthly = any(spacing[a] is not None and spacing[a] > 20 for a in assets)
    cells = [(a, h) for a in assets for h in horizons]
    s_cell = []
    for a, h in cells:
        pub = PUBLISHED_MONTHLY_STEPS.get(unit_id, {}).get(h)
        s_cell.append(pub if pub is not None else
                      horizon_steps(win300[a], h, spacing[a], asof + pd.tseries.offsets.BDay(h), periods.get(h)))
    smax = max(s_cell)
    n_draws = int(c["n_draws"])
    rng = np.random.default_rng(seed + (zlib.crc32(unit_id.encode()) & 0xFFFF))
    if c["shock"] == "t4":
        z = rng.standard_t(T_DF, size=(n_draws, smax, n)) / np.sqrt(T_DF / (T_DF - 2))
    else:
        z = rng.standard_normal((n_draws, smax, n))
    z = z @ chol.T
    width = float(c["width"].get(fam, 1.0))
    h_exp = float(c["fan"].get(fam, 0.5)) if not monthly else 0.5
    centre = {}
    if centre_model != "m0":   # E3: a classical model's mean path replaces anchor + s*mu
        for a in assets:
            m0_path = anchor[a] + np.arange(1, smax + 1) * mu[a]
            if centre_model.endswith("+m0"):          # equal-weight combination with M0's drift path
                centre[a] = 0.5 * (classical_centre(win300[a].values, steps300[a].values, target_type, smax, centre_model[:-3]) + m0_path)
            else:
                centre[a] = classical_centre(win300[a].values, steps300[a].values, target_type, smax, centre_model)
    out = np.empty((n_draws, len(cells)))
    for i, ((a, h), s) in enumerate(zip(cells, s_cell)):
        j = assets.index(a)
        cum = np.cumsum(z[:, :s, j], axis=1)[:, -1] * sd[a] * width
        if h_exp != 0.5:
            cum = cum * (s / FAN_PIVOT) ** (h_exp - 0.5)
        dd = c.get("drift_damp") or 0                                                   # T1: drift x min(1, H0/steps); scalar or per family
        dd = dd.get(fam, 0) if isinstance(dd, dict) else dd
        damp = min(1.0, float(dd) / s) if dd else 1.0
        mean_s = centre[a][s - 1] if centre else anchor[a] + s * mu[a] * damp
        mom = c.get("momentum", {}).get(fam, 0.0) if isinstance(c.get("momentum"), dict) else float(c.get("momentum") or 0)
        if mom and len(full[a]) > 21:                            # T6: +-mom x cell sd on the sign of the last 21 observations' change
            mean_s = mean_s + mom * np.sign(float(full[a].iloc[-1] - full[a].iloc[-22])) * sd[a] * np.sqrt(s) * width
        val = mean_s + cum
        out[:, i] = np.exp(val) if use_log[a] else val
    mix = float(c.get("mix_m0", 0) or 0)                        # T7: this share of draws comes from M0's own construction
    if mix > 0:
        m0cfg = {**DEFAULT, "n_draws": int(round(mix * n_draws))}
        Xm, _, _ = simulate(unit_dir, m0cfg, seed + 7919)
        out[: Xm.shape[0], :] = Xm
    info = {"cfg": c, "seed": seed, "cells": cells, "steps": s_cell, "mu": mu, "sd": sd, "anchor": anchor,
            "family": fam, "monthly": monthly, "unit_id": unit_id, "asof": str(asof.date()),
            "asset_ids": list(tgt["asset_ids"]), "horizons": [int(h) for h in tgt["horizons"]], "target": target_type}
    return out, cells, info


def write(samples: np.ndarray, cells: list[tuple[str, int]], info: dict, out: pathlib.Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    n = samples.shape[0]
    pd.DataFrame({"draw": np.repeat(np.arange(n, dtype=np.int32), len(cells)),
                  "asset": np.tile([a for a, _ in cells], n),
                  "horizon": np.tile(np.array([h for _, h in cells], dtype=np.int32), n),
                  "value": samples.reshape(-1)}).to_parquet(out, index=False)
    (out.parent / "forecast_meta.json").write_text(json.dumps({
        "unit_id": info["unit_id"], "asof": info["asof"], "representation": "samples",
        "asset_ids": info["asset_ids"], "horizons": info["horizons"], "n_draws": n, "target": info["target"],
        "rationale": {"file": RATIONALE, "method": ("text-blind engine " if not info.get("text_ledger", {}).get("reasoning_applied") else "engine + text layer (L3 per-document direction, L5 reader dispersion -> width) ") + json.dumps({k: v for k, v in info["cfg"].items() if k != "text"}, sort_keys=True)},
        **({"reasoning_applied": info["text_ledger"].get("reasoning_applied", False),
            "reasoning_skipped_reason": info["text_ledger"].get("reasoning_skipped_reason", ""),
            "house_requests": info["text_ledger"].get("calls", 0)} if info.get("text_ledger") is not None else {})},
        indent=2) + "\n")
    (out.parent / RATIONALE).write_text(
        f"# {info['unit_id']} — text-blind engine\n\nAs of {info['asof']}: random-walk path with config "
        f"{json.dumps(info['cfg'], sort_keys=True)}, seed {info['seed']}; drift = mean step of the trailing 300 "
        f"observations x {info['cfg']['drift']}; steps per cell {dict(zip([f'{a}@{h}' for a, h in cells], info['steps']))}. "
        + ("".join(f"\n- {n}" for n in info.get("rule_notes", [])) if info.get("rule_notes") else "")
        + (("\n\nText layer (House model, thinking off): " + json.dumps(info["text_ledger"], sort_keys=True) + "\n") if info.get("text_ledger") is not None
           else "\nNo document text was read; the corpus manifest's document types and a public FOMC calendar are the only text-side inputs.\n"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--unit", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--cfg", default="{}", help="JSON overriding the M0 defaults")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    samples, cells, info = simulate(a.unit, json.loads(a.cfg), a.seed)
    write(samples, cells, info, a.out)
    print(f"wrote {a.out} ({samples.shape[0]} draws x {len(cells)} cells)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
