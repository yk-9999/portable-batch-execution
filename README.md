# Portable Batch Execution

Portable Batch Execution is a backend-agnostic framework for finite,
reproducible batch workloads.

v1 implements a frozen Pydantic contract, planning/finalization kernel, local
filesystem data plane, five closed domain packs, and a GitHub Actions backend.
The kernel is the only RunManifest writer; workers emit attempts and backend
evidence only.

## Install

```bash
uv sync --dev
uv sync --extra tabular --extra ml
```

`httpx` and Pydantic are base dependencies. The tabular pack uses Polars and
DuckDB; the ML pack uses scikit-learn. Acquisition uses bounded HTTP through
the base dependency. Media operations require `ffmpeg` and `ffprobe` available
on `PATH`; no Python media runtime is bundled.

## v1 surface

- `portable_batch_execution.kernel` plans index/time shards and bounded waves,
  determines canonical successful attempts, and finalizes compare-and-swap
  manifests.
- `portable_batch_execution.data_plane.LocalFilesystemDataPlane` stores
  immutable artifacts and manifest revisions locally.
- `portable_batch_execution.packs` exports `TabularPack`, `AcquisitionPack`,
  `ReplayEvalPack`, `MLPack`, and `MediaPack`. Replay evaluation delegates
  project semantics exclusively to a trusted adapter.
- `GitHubActionsBackend` dispatches a workflow, returns a backend execution
  reference, then collects status evidence or cancels that same reference.

Generate contract schemas with:

```bash
uv run python scripts/export_schemas.py
```

## Non-goals

- Arbitrary shell execution
- Arbitrary Python source execution
- Arbitrary SQL supplied through a JobSpec
- Daemon or real-time workloads
- Storing private project data, models, or results in this public repository

## Private Data Plane

Trusted `workflow_dispatch` can retain the committed `wave-0000` public smoke
path, or select private mode with only opaque `run_id` and `wave_id`. Private
mode reads `PBE_PRIVATE_DATA_PLANE_BASE_URL` (HTTPS origin only) and
`PBE_PRIVATE_DATA_PLANE_BEARER_TOKEN` from the workflow environment. The
controller implements `GET /v1/runs/{run_id}/waves/{wave_id}`, artifact content
at `/v1/artifacts/{object_id}/content`, artifact upload, and run attempt
endpoints. It returns closed JobSpec, WaveSpec, and ShardSpec contracts; the
public worker never accepts a path, URL, command, code, import, or secret as an
input and only dispatches its fixed public operation registry. Artifact URIs
are metadata and must never carry credentials.

The kernel exposes `exhausted_shards` for current unsuccessful attempts that
reach the configured per-shard budget (the initial attempt counts). Project
controllers may use that state to choose their own fallback; fallback
implementations and private project logic remain outside this repository.
