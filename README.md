## Executive summary (read this first)

Frozen alpha-extractor Track 2 candidate: numerical GLS calibration plus 25 House
LLM scenarios on eligible target grids. CodaBench runs the published container;
participants upload the release ZIP containing its descriptor and team claim.
Model weights and evaluation inputs are not bundled.

## Build and interface

```bash
docker build -t agenthon-scenario-hybrid-v2 .
docker run --rm agenthon-scenario-hybrid-v2 forecast --help
```

The organizer mounts the task inputs, provides the House endpoint/model/token and
runs `forecast --panels /input/panels --text /input/text --asof YYYY-MM-DD
--out /output/forecast.parquet`. Numerical-only tasks need no model endpoint.
Eligible tasks stop on service failure and retain the numerical forecast.

Read ARTIFACT_PROVENANCE.md for selection history, runtime limits, verification
scope and unresolved House compatibility. freeze.json lists the packaged source
hashes. Earlier release tags retain their own frozen implementations.
