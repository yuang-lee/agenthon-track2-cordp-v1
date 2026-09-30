## Executive summary (read this first)

This release packages numerical forecasting code and fixed v1 prompts. Its only
language-model access is the organizer-injected House endpoint. No neural model
weights, task panels, corpus documents, realized outcomes or cached predictions
are included in the image. This record describes provenance; it does not certify
organizer eligibility or the account-to-team link.

## Source and reproducibility

The implementation derives from the team's MIT-licensed
`Agenthon-2026-track2-forecasting` workspace, copied from
`/data/annaliang/june/Agenthon-2026-track2-forecasting`. Frozen source hashes are in
`freeze.json`; the published source revision is
`fe7ca9c493dd875e7591cb1038515d6c1000dcf7`. Python's base image is pinned in Dockerfile,
and all numerical dependencies are pinned in requirements.txt. The participant
image is `ghcr.io/yuang-lee/agenthon-track2-cordp-v1@sha256:960c469f67f0726f89ecf79cd76c4748fd17f111e08efb2f7e466ba24d3c115f`.

The dev-10 numerical configuration in `pipelines/ebest.json` is inherited from the
team's prior experiments. The v1 prompts and full-strength median correction were
selected after local public-development comparisons using Nemotron 30B on
2026-09-30. The reference comparison used 29 scorable public-development cards;
32 cards received model calls. No held-out validation feedback was used to select
this release. These statements record actual selection history; they do not assert
that post-cutoff selection is exempt from the organizer's cutoff rules.

## Runtime artifacts and cutoff handling

- Numerical methods: on-input Theta fitting, simulation, covariance estimation,
  and the inherited rules in the frozen engine. No fitted neural checkpoint or
  serialized per-unit forecast is packaged. The exact fixed configuration and rule
  implementation are inspectable under `pipelines/`.
- Calendar: `pipelines/fomc_dates.csv` is an inherited static calendar, with its
  exact hash in `freeze.json`. Its original release dates and acquisition provenance
  were not independently reconstructed during packaging. The source can be inspected
  together with the rule that uses it.
- At inference, numerical histories are filtered to the card cutoff. Documents
  published after the cutoff are excluded. The last observation date, document age,
  business-day horizons and available observation periods are supplied to the LLM.
  No outcome file or cached LLM reply is read by the participant image.
- House disclosure: `nvidia/nemotron-3-super-120b-a12b`, reported revision and
  version `rl-030326-fp8`, training cutoff `unpublished`, access `api`. Runtime calls
  use MODEL_ENDPOINT, MODEL_NAME and MODEL_TOKEN injected by the organizer.
- The 30B research model and locally downloaded public 120B FP8 weights are not
  included or loaded by this image. Local 30B replay verifies packaging equivalence,
  not House 120B performance.

## Submission identity

The upload ZIP contains only `submission.json` and a toolkit-generated
`team-claim.json` for Team 153. The raw Team Key, API keys and other credentials are
absent from the source, image and ZIP. The claim authenticates this descriptor;
it does not independently prove that a new CodaBench account will be linked.
