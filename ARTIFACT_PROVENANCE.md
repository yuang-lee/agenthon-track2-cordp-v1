## Executive summary (read this first)

This candidate combines a numerical joint forecast with independent House-model
scenarios. The image contains code, configuration and a public meeting calendar.
It contains no task panels, article corpus, outcomes, cached predictions, model
weights or credentials. Local reproducibility is verified; the organizer's House
proxy and sealed evaluation have not yet been tested with this candidate.

## Runtime behavior

The inherited numerical engine fits the supplied history, simulates 4,000 joint
draws and applies calendar and corpus-metadata rules. Generalized least squares
(GLS) calibration shrinks the covariance-weighted common component by 0.75 while
preserving each column mean and every within-draw coordinate difference.

Routing uses target frequency, asset units and target type, not known task IDs:

- Monthly targets retain the25 percent whole-scenario mixture.
- Daily Treasury level grids with multiple cells use a bounded25 percent mixture
  toward the numerical column-mean common center. House proposals select only
  relative orientation, by reflecting a whole row around its mean if its relative
  direction opposes the model scenario. Every within-row absolute difference is
  preserved; the LLM does not set the common level.
- Daily single-asset FX level grids with multiple horizons use a bounded25 percent
  common mixture toward the last observed value. They make zero House requests.
- Other grids retain GLS without model requests.

The common-shift limit is half the smallest numerical marginal80 percent width.
Final price support and finite values are checked. Numerical-only corrections
retain GLS if support fails. Model service failure retains the numerical common
center on rates and GLS on monthly tasks; the request loop stops without retry.

Eligible tasks make 25 separate requests, using seeds 0 through 24, temperature 1,
thinking disabled and a 3,000-token output limit. Each prompt requests one joint
future realization and supplies the last 60 observations plus frozen dated text.
The LLM does not receive the numerical forecast; the combination happens afterward.
Text is selected by the frozen inherited reader, rather than an external retrieval
service. There is no vendor web search or unrestricted network use.

Requests use MODEL_ENDPOINT, MODEL_NAME and MODEL_TOKEN supplied by the organizer.
Strict JSON Schema output is requested. Generic NVIDIA NIM documentation supports
this request format, but the actual House proxy/backend behavior remains untested.
HTTP or transport failure stops further requests and retains that route's numerical forecast;
no uncertain request is retried. Incomplete or invalid scenario replies use the
frozen history-based fallback. Outputs include the actual call ledger and fallback
information, without credentials or model endpoint values.

## Provenance and selection history

The MIT-licensed numerical engine, configuration and meeting calendar derive from
the alpha-extractor team's Agenthon-2026-track2-forecasting project. The calendar's
publication vintages have not been independently reconstructed. A packaging guard
prevents an uncovered future calendar interval from being treated as proof that
no FOMC meeting occurs; covered historical behavior is unchanged.

The source hashes, prompt code and numerical settings are recorded in freeze.json.
The Python base image and dependency versions are pinned. Research used public
practice inputs, locally reconstructed references, model/component ablations and
two separate model-generation blocks, matched no-news controls and fixed numerical
location controls on83 previously explored newer calendar windows. These are repeated comparisons on explored
cards, not independent held-out validation. The fixed routing and parameters were
selected during that research. We do not claim that post-cutoff selection is
exempt from the organizer's artifact policy or certify competition eligibility.

Local generation used NVIDIA Nemotron 3 Super 120B A12B FP8, Hugging Face revision
744b1880a37996c5d56bf454ae164dfd74d77c4e, with an Ampere-compatible local vLLM runtime.
Those weights are not packaged. The submission discloses only the injected House
model, nvidia/nemotron-3-super-120b-a12b, version/revision rl-030326-fp8, API access,
training cutoff unpublished. Local decoding need not match the House backend.

## Verification and identity

The actual container reproduces all 103 frozen practice outputs through exact
request matching and archived-reply replay, including 584 HTTP requests. This
tests code and transport, not new inference or House compatibility. The minimal
official exemplar, network-disabled numerical FX and an HTTP 503 numerical-center
fallback also pass local official gates. All106cases passed before publication. The
container runs nonroot, with a read-only root filesystem and bounded resources.

The upload ZIP contains a pinned image descriptor and the toolkit-generated
Team 153 claim. The raw Team Key is absent from this source, image and ZIP.
Only forecast.parquet, forecast_meta.json and forecast_rationale.md are emitted.
