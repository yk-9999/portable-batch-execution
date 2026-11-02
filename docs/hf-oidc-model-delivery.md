# Private model delivery through Hugging Face read-only OIDC

Status: **source-only / off by default**. This document is not approval to
create a Hugging Face repository, upload models, register a Trusted Publisher,
or change A1/Actions production configuration. These remain separate Commander
operational approvals.

## Current and target behavior

The existing `ml.distilbert_pair_binary_scores` broker binding contains exactly
four locally pinned model ArtifactRefs (config, weights, config, weights).
Without a `model_delivery` descriptor, the Public runner downloads all four
files through the authenticated A1 Private Data Plane, exactly as before.

With a validated descriptor, **newly registered** model jobs fetch those four
files directly from one HF model-only private repository. The A1 controller
still sends the tensor input and four immutable reference identities in the
private wave plan, but **does not serve model file contents**. Public verifies
each downloaded file's byte count and SHA-256 against the A1 static ArtifactRef
before loading it in the same local-only DistilBERT pack. Temporary files,
Xet/cache and exchange data live under a per-attempt temporary directory.

The model location is transport metadata in JobSpec, not client-controlled
`operation_params`, and does not change the request fingerprint or `public_sha`.
Already registered requests retain their previously stored JobSpec and
original delivery route across retries/restarts.

## Pending approved HF setup (not executed by this source PR)

1. Select or create a **model-only private model repository** under the
   existing approved HF account. Do not use the shared MySQL-backup bucket.
2. Publish *only* the four frozen model files at four unique paths, such as
   `model_a/config.json`, `model_a/model.safetensors`,
   `model_b/config.json`, `model_b/model.safetensors`.
3. Compare each remote file's SHA-256 and byte size to A1's existing pinned
   static ArtifactRef. Freeze the HF **40-character commit SHA**; no `main`,
   branch name, or tag is accepted for model identity.
4. In this private model repository's Settings > Trusted Publishers, add a
   **Read-only** repository publisher for GitHub Actions with claims restricted
   to GitHub repository `yk-9999/portable-batch-execution`, branch `main`,
   and workflow `.github/workflows/execute-wave.yml`. Do not create an account
   CI/CD identity: those do not read private model repos. Do not grant write.
5. Check account-wide private storage quota and per-job concurrent download
   behavior. Record 4/8-parallel performance and 401/403/429 behavior prior
   to changing production routing. No manual token or old HF backup token is
   installed into GitHub Secrets.
6. Only after rollout approval, add `model_delivery` to the A1-only protected
   broker config's existing DistilBERT static binding, for example:

```json
{
  "pack": "ml-batch",
  "operation": "ml.distilbert_pair_binary_scores",
  "input_refs": [
    {"object_id": "<existing-a-config-id>", "uri": "<existing-private-uri>",
     "sha256": "sha256:<existing-64-hex-digest>", "size_bytes": 0},
    {"object_id": "<existing-a-weights-id>", "uri": "<existing-private-uri>",
     "sha256": "sha256:<existing-64-hex-digest>", "size_bytes": 0},
    {"object_id": "<existing-b-config-id>", "uri": "<existing-private-uri>",
     "sha256": "sha256:<existing-64-hex-digest>", "size_bytes": 0},
    {"object_id": "<existing-b-weights-id>", "uri": "<existing-private-uri>",
     "sha256": "sha256:<existing-64-hex-digest>", "size_bytes": 0}
  ],
  "model_delivery": {
    "kind": "hf-private-repo-oidc",
    "repo_id": "<approved-namespace>/<approved-private-model-repo>",
    "revision": "<exact-40-hex-commit-sha>",
    "filenames": [
      "model_a/config.json", "model_a/model.safetensors",
      "model_b/config.json", "model_b/model.safetensors"
    ]
  }
}
```

The `input_refs` above are **shape placeholders**; preserve the actual
four A1 refs, including their real sizes and digests. The validation refuses
missing sizes, unsafe paths, unsupported operations, duplicate filenames, or
unpinned revisions. The new descriptor is **not accepted from the broker client**.

## Security and runtime boundaries

- `execute-wave.yml` grants `id-token: write` so the CLI can mint a GitHub
  OIDC identity. The HF repo's read-only Trusted Publisher checks claims; its
  token expires after ~60 minutes. No long-lived `HF_TOKEN` is supplied.
- The worker calls `hf auth token` with `HF_OIDC_RESOURCE` matching the
  private model repository; `huggingface_hub` uses that short-lived token for
  an exact-revision download. The token and CLI/HTTP error bodies are never
  printed, included in attempts, or persisted in the Public repository.
- The worker rejects missing GitHub OIDC context even if `HF_TOKEN` is
  present. It stores no model in Git commits, published Action artifacts,
  globally shared GitHub cache, or logs. All model files are removed with
  the per-attempt temporary directory. A1 retains local models as fallback.
- HF 401/403 => `hf_model_auth_failed`; 429 =>
  `hf_model_rate_limited`; other network failures =>
  `hf_model_download_failed`; byte mismatch =>
  `hf_model_digest_mismatch`. The existing bounded attempt/retry and
  parent Private fallback policy remains authoritative; Public never silently
  accepts or substitutes corrupted model content.
- Public code contains no private repository name, model checksum or model
  weights. Only the private broker configuration pins the actual source.
- `public_sha` remains untouched. A delivery-source-only change never
  rebinds an already registered request to a different execution fingerprint.

## Acceptance still gated on authorized resources

Offline tests validate transport metadata, identity/replay, sanitization,
401/403/429 behavior, interrupted fetch, SHA/size checks and actual numerical
parity using synthetic DistilBERT models. Real HF OIDC token exchange, model
repository permission, quota, 4/8 concurrent retrieval, actual A1 egress/CPU/
I/O and per-video latency measurements cannot be proven before repo/issuer
setup and model upload. Existing production is intentionally unchanged.

Reference: https://huggingface.co/docs/hub/trusted-publishers
