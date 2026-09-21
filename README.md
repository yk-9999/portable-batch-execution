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

### HF-direct artifact mode

Trusted `workflow_dispatch` may also select `hf_direct` together with the
bounded HF store identity metadata `hf_bucket`, `hf_prefix`, and
`hf_object_layout` (`sha256-flat.v1`). In this mode the A1 controller remains the
control plane for closed-wave resolution, attempt/status metadata, and
orchestration, while artifact payload bytes move directly between the GitHub
Actions worker and the private HF bucket:

- `HfDirectDataPlane` delegates control methods (`resolve_wave`,
  `read_attempts`, `append_attempt`, manifest/status) to the A1 client and all
  artifact methods (`read`/`write`/`exists`/`verify`) to
  `HfBucketArtifactStore`; no artifact request is sent to `/v1/artifacts`.
- `sha256-flat.v1` maps `object_id` (lowercase SHA-256 hex) to
  `<prefix>/objects/sha256/<object_id>`. The identity is validated fail-closed
  against unsafe or out-of-scope bucket/prefix values before any remote call.
- Reads derive the object solely from the closed identity plus `ref.object_id`,
  download it directly, then verify the expected SHA-256 and exact size before
  returning bytes. Writes content-address the bytes, idempotently reuse an
  existing exact-size object, otherwise upload directly and re-check
  persistence, returning an `ArtifactRef` with an `hf://buckets/...` reference.
- The HF token is read only from the `HF_SYSTEM_TRADING_DATA_RW_TOKEN`
  environment variable injected by the workflow from the fixed repository
  secret. It is never placed on argv, in logs, in returned metadata, or in a
  persisted file. The pinned `hf` CLI (exact version `1.8.0`) is installed only
  for HF-direct jobs.
- Runtime-profile resolution still consumes A1 control metadata only
  (`resolve_wave`) and never fetches artifact bytes.

The existing A1 artifact mode (`private`) and public synthetic mode remain
unchanged for unrelated callers.
