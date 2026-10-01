## Executive summary (read this first)

Frozen alpha-extractor Track2 candidate: numerical location control with House
scenario directions for Treasury grids, numerical persistence for single-asset
FX, and whole-scenario mixtures for monthly targets. Participants upload the
release ZIP; CodaBench runs the immutable image on freshly mounted inputs.
No model weights, cached forecasts, practice inputs or outcomes are bundled.

## Interface and verification

Build from the context directory with `docker build -t agenthon-scenario-hybrid-v3 .`.
Run `forecast --panels /input/panels --text /input/text --asof YYYY-MM-DD
--out /output/forecast.parquet`. Numerical-only tasks do not need a model endpoint.
House requests use the organizer-injected endpoint, model name and token.

Eligible tasks use at most 25 separate non-thinking requests, each with a 3000-token
output cap. Failure stops the request loop and retains the numerical fallback.
Read ARTIFACT_PROVENANCE.md for model disclosure and actual selection history.
The context freeze.json records all packaged source hashes and fixed settings.
Previous release tags preserve previous implementations.
