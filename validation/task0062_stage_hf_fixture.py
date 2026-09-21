from __future__ import annotations

import json

from portable_batch_execution.data_plane.hf import (
    OBJECT_LAYOUT_VERSION,
    HfBucketArtifactStore,
    HfBucketIdentity,
)

ROW_COUNT = 90_000
TEXT = "ab" * 50
MIN_INPUT_BYTES = 16 * 1024 * 1024
MAX_INPUT_BYTES = 65 * 1024 * 1024

IDENTITY = HfBucketIdentity.from_metadata(
    {
        "kind": "hf-buckets-direct",
        "bucket": "yamauchiJP/system-trading-data",
        "prefix": "live-stream-news/tmp/task0038/",
        "object_layout_version": OBJECT_LAYOUT_VERSION,
        "required_hf_cli_version": "1.8.0",
    }
)


def build_payload() -> bytes:
    rows = [
        {
            "row_id": f"task0062-{index:06d}",
            "partition_key": "task0062-stress",
            "segment_key": "s1",
            "event_time": "2026-09-21T12:00:00+00:00",
            "entity_id": f"entity-{index:06d}",
            "text": TEXT,
        }
        for index in range(ROW_COUNT)
    ]
    return json.dumps(rows, separators=(",", ":"), sort_keys=True).encode("utf-8")


def main() -> int:
    payload = build_payload()
    if not MIN_INPUT_BYTES <= len(payload) <= MAX_INPUT_BYTES:
        raise RuntimeError(f"fixture input size outside bound: {len(payload)}")
    store = HfBucketArtifactStore(IDENTITY)
    store.preflight()
    ref = store.write(payload, "application/json")
    if not store.verify(ref):
        raise RuntimeError("HF fixture verification failed")
    print(
        "TASK0062_FIXTURE_REF="
        + json.dumps(ref.model_dump(mode="json"), sort_keys=True)
    )
    print(
        "TASK0062_FIXTURE_META="
        + json.dumps(
            {
                "row_count": ROW_COUNT,
                "input_size_bytes": len(payload),
                "synthetic_non_outcome": True,
                "text_pattern": "ab-repeat",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
