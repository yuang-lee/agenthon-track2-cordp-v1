## Executive summary (read this first)

This source-only release adds covariance-weighted common-component shrinkage to
alpha-extractor's existing numerical forecast. It makes no language-model calls.
The image contains no task panels, text corpora, realized outcomes, cached
forecasts, pretrained neural weights or credentials. Source hashes are frozen in
freeze.json; dependencies and the Python base image are pinned.

## Forecast behavior

The inherited engine fits Theta and numerical moments to the supplied history,
simulates joint paths, and applies the existing calendar/corpus-metadata rules.
The new step shrinks the minimum-variance common component by0.75 in raw target
coordinates, preserving every within-draw cell difference and each column mean.
It estimates covariance from the generated draws, without target outcomes.
Single-cell cases retain the baseline. Degenerate covariance or invalid positive
FX support also retain the complete baseline. Default draw seed remains0.

The transformation preserves the variogram term algebraically. It changes the
marginal distributions; their medians and higher moments need not stay fixed.
This is a statistical calibration candidate, not evidence that news reasoning
improved forecasting. It uses no LLM or additional foundation-model checkpoint.

## Provenance and limitations

The original numerical engine/configuration and calendar derive from the team's
MIT-licensed Agenthon-2026-track2-forecasting project. Calendar publication vintages
have not been independently reconstructed. Research used local reconstructed
references and repeated public-practice comparisons; this release makes no claim
of sealed leaderboard improvement or organizer eligibility certification.

The default forecast entrypoint is tested against the frozen research candidate
under a read-only root filesystem, no network, and nonroot execution. Runtime
outputs contain only forecast.parquet, forecast_meta.json and forecast_rationale.md.
The upload descriptor discloses models:[], category api. A separate team claim is
created using the official toolkit; the raw Team Key never enters this repository.
