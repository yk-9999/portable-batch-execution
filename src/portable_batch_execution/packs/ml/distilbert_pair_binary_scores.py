"""Closed DistilBERT pair binary class-1 probability scoring for private tensor input."""

from __future__ import annotations

import json
import math
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from portable_batch_execution.contracts import ArtifactRef

DistilbertStaticRefIdentity = tuple[str, str, int | None, str | None, str]
DistilbertStaticModelFingerprint = tuple[
    DistilbertStaticRefIdentity,
    DistilbertStaticRefIdentity,
    DistilbertStaticRefIdentity,
    DistilbertStaticRefIdentity,
]

INPUT_SCHEMA_VERSION = "pbe.ml.distilbert-pair-binary-scores.v1"
OUTPUT_SCHEMA_VERSION = "pbe.ml.distilbert-pair-binary-scores-output.v1"

_TOP_LEVEL_KEYS = frozenset({"schema_version", "row_ids", "model_a", "model_b"})
_MODEL_TENSOR_KEYS = frozenset({"input_ids", "attention_mask"})
_MAX_ROWS = 128
_MIN_SEQ_LEN = 1
_MAX_SEQ_LEN = 256


def _is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


_EXPECTED_ARCHITECTURE = "DistilBertForSequenceClassification"


def _validate_raw_distilbert_config_json(config: dict[str, Any]) -> None:
    if config.get("model_type") != "distilbert":
        raise ValueError("model config model_type must be distilbert")
    architectures = config.get("architectures")
    if architectures is not None:
        if not isinstance(architectures, list):
            raise TypeError("model config architectures must be an array")
        if architectures != [_EXPECTED_ARCHITECTURE]:
            raise ValueError(
                "model config architectures must be exactly DistilBertForSequenceClassification"
            )
    id2label = config.get("id2label")
    if id2label is not None and (not isinstance(id2label, dict) or len(id2label) != 2):
        raise ValueError("model config id2label must contain exactly 2 labels")
    label2id = config.get("label2id")
    if label2id is not None and (not isinstance(label2id, dict) or len(label2id) != 2):
        raise ValueError("model config label2id must contain exactly 2 labels")


def _validate_staged_distilbert_classifier_config(directory: Path) -> None:
    from transformers import DistilBertConfig

    parsed = DistilBertConfig.from_pretrained(directory, local_files_only=True)
    if parsed.num_labels != 2:
        raise ValueError("model config num_labels must be 2")


def validate_distilbert_classifier_config(config: Any) -> None:
    if not isinstance(config, dict):
        raise TypeError("model config must be an object")
    _validate_raw_distilbert_config_json(config)


def _parse_model_tensors(name: str, side: Any, row_count: int) -> tuple[list[list[int]], list[list[int]]]:
    if not isinstance(side, dict):
        raise TypeError(f"{name} must be an object")
    if frozenset(side.keys()) != _MODEL_TENSOR_KEYS:
        raise ValueError(f"{name} has unexpected fields")
    input_ids = side.get("input_ids")
    attention_mask = side.get("attention_mask")
    if not isinstance(input_ids, list) or not isinstance(attention_mask, list):
        raise TypeError(f"{name} tensors must be arrays")
    if len(input_ids) != row_count or len(attention_mask) != row_count:
        raise ValueError(f"{name} row dimension must match row_ids")
    parsed_ids: list[list[int]] = []
    parsed_mask: list[list[int]] = []
    seq_len: int | None = None
    for row_index in range(row_count):
        id_row = input_ids[row_index]
        mask_row = attention_mask[row_index]
        if not isinstance(id_row, list) or not isinstance(mask_row, list):
            raise TypeError(f"{name} tensor rows must be arrays")
        if len(id_row) != len(mask_row):
            raise ValueError(f"{name} input_ids and attention_mask shapes must match")
        if not (_MIN_SEQ_LEN <= len(id_row) <= _MAX_SEQ_LEN):
            raise ValueError(f"{name} sequence length must be between {_MIN_SEQ_LEN} and {_MAX_SEQ_LEN}")
        if seq_len is None:
            seq_len = len(id_row)
        elif len(id_row) != seq_len:
            raise ValueError(f"{name} sequence length must be consistent across rows")
        parsed_id_row: list[int] = []
        parsed_mask_row: list[int] = []
        for token_id, mask_value in zip(id_row, mask_row, strict=True):
            if not isinstance(token_id, int) or isinstance(token_id, bool) or token_id < 0:
                raise ValueError(f"{name} input_ids must be non-negative integers")
            if mask_value not in (0, 1):
                raise ValueError(f"{name} attention_mask values must be 0 or 1")
            if not isinstance(mask_value, int) or isinstance(mask_value, bool):
                raise TypeError(f"{name} attention_mask must be integer 0 or 1")
            parsed_id_row.append(token_id)
            parsed_mask_row.append(mask_value)
        parsed_ids.append(parsed_id_row)
        parsed_mask.append(parsed_mask_row)
    return parsed_ids, parsed_mask


def validate_distilbert_pair_binary_scores_input(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise TypeError("input must be an object")
    if frozenset(payload.keys()) != _TOP_LEVEL_KEYS:
        raise ValueError("input has unexpected fields")
    if payload.get("schema_version") != INPUT_SCHEMA_VERSION:
        raise ValueError("unsupported input schema_version")
    row_ids = payload.get("row_ids")
    if not isinstance(row_ids, list) or not row_ids:
        raise ValueError("row_ids must be a non-empty array")
    if len(row_ids) > _MAX_ROWS:
        raise ValueError("row_ids exceeds supported row bound")
    seen: set[str] = set()
    normalized_ids: list[str] = []
    for row_id in row_ids:
        if not isinstance(row_id, str) or not row_id:
            raise ValueError("row_ids must be unique non-empty strings")
        if row_id in seen:
            raise ValueError("row_ids must be unique non-empty strings")
        seen.add(row_id)
        normalized_ids.append(row_id)
    row_count = len(normalized_ids)
    model_a_ids, model_a_mask = _parse_model_tensors("model_a", payload.get("model_a"), row_count)
    model_b_ids, model_b_mask = _parse_model_tensors("model_b", payload.get("model_b"), row_count)
    return {
        "row_ids": tuple(normalized_ids),
        "model_a": (model_a_ids, model_a_mask),
        "model_b": (model_b_ids, model_b_mask),
    }


def _stage_model_bundle(config_bytes: bytes, weights_bytes: bytes, directory: Path) -> None:
    try:
        config = json.loads(config_bytes.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ValueError("model config is not valid JSON") from exc
    _validate_raw_distilbert_config_json(config)
    config_path = directory / "config.json"
    weights_path = directory / "model.safetensors"
    config_path.write_bytes(config_bytes)
    weights_path.write_bytes(weights_bytes)
    _validate_staged_distilbert_classifier_config(directory)


def _infer_class1_probabilities_from_model(
    model: Any,
    input_ids: list[list[int]],
    attention_mask: list[list[int]],
) -> list[float]:
    import torch

    model.eval()
    ids_tensor = torch.tensor(input_ids, dtype=torch.long)
    mask_tensor = torch.tensor(attention_mask, dtype=torch.long)
    with torch.no_grad():
        logits = model(input_ids=ids_tensor, attention_mask=mask_tensor).logits
        probabilities = torch.softmax(logits, dim=1)[:, 1]
    scores = probabilities.tolist()
    if not all(_is_finite_number(score) for score in scores):
        raise ValueError("model output is not finite")
    return [float(score) for score in scores]


def _load_classifier_from_staged_directory(model_dir: Path) -> Any:
    from transformers import DistilBertForSequenceClassification

    return DistilBertForSequenceClassification.from_pretrained(
        model_dir,
        local_files_only=True,
    )


def distilbert_static_artifact_ref_identity(ref: ArtifactRef) -> DistilbertStaticRefIdentity:
    """Immutable ArtifactRef fields used for per-shard static model cache identity."""
    return (
        ref.object_id,
        ref.sha256,
        ref.size_bytes,
        ref.media_type,
        ref.uri,
    )


def distilbert_static_model_fingerprint(
    model_a_config: ArtifactRef,
    model_a_weights: ArtifactRef,
    model_b_config: ArtifactRef,
    model_b_weights: ArtifactRef,
) -> DistilbertStaticModelFingerprint:
    """Fingerprint the four static model ArtifactRefs shared across DistilBERT shards."""
    return (
        distilbert_static_artifact_ref_identity(model_a_config),
        distilbert_static_artifact_ref_identity(model_a_weights),
        distilbert_static_artifact_ref_identity(model_b_config),
        distilbert_static_artifact_ref_identity(model_b_weights),
    )


def distilbert_static_model_fingerprint_from_bytes(
    model_a_config: bytes,
    model_a_weights: bytes,
    model_b_config: bytes,
    model_b_weights: bytes,
) -> DistilbertStaticModelFingerprint:
    """Content-only fingerprint for callers without ArtifactRef metadata."""
    from hashlib import sha256

    def _content_identity(payload: bytes, slot: str) -> DistilbertStaticRefIdentity:
        digest = f"sha256:{sha256(payload).hexdigest()}"
        return (
            f"content:{slot}:{digest}",
            digest,
            len(payload),
            None,
            f"pbe://inline/{slot}",
        )

    return (
        _content_identity(model_a_config, "model-a-config"),
        _content_identity(model_a_weights, "model-a-weights"),
        _content_identity(model_b_config, "model-b-config"),
        _content_identity(model_b_weights, "model-b-weights"),
    )


@dataclass
class DistilbertStaticModelPair:
    """Staged and loaded DistilBERT classifiers reused across shard tensor inputs."""

    fingerprint: DistilbertStaticModelFingerprint
    _temporary: tempfile.TemporaryDirectory[str]
    _model_a: Any
    _model_b: Any

    @classmethod
    def from_verified_artifact_bytes(
        cls,
        *,
        model_a_config: bytes,
        model_a_weights: bytes,
        model_b_config: bytes,
        model_b_weights: bytes,
        fingerprint: DistilbertStaticModelFingerprint,
    ) -> DistilbertStaticModelPair:
        temporary = tempfile.TemporaryDirectory()
        root = Path(temporary.name)
        model_a_dir = root / "model_a"
        model_b_dir = root / "model_b"
        model_a_dir.mkdir()
        model_b_dir.mkdir()
        _stage_model_bundle(model_a_config, model_a_weights, model_a_dir)
        _stage_model_bundle(model_b_config, model_b_weights, model_b_dir)
        return cls(
            fingerprint=fingerprint,
            _temporary=temporary,
            _model_a=_load_classifier_from_staged_directory(model_a_dir),
            _model_b=_load_classifier_from_staged_directory(model_b_dir),
        )

    def execute_pair_binary_scores(self, payload: Any) -> dict[str, Any]:
        validated = validate_distilbert_pair_binary_scores_input(payload)
        model_a_ids, model_a_mask = validated["model_a"]
        model_b_ids, model_b_mask = validated["model_b"]
        model_a_scores = _infer_class1_probabilities_from_model(
            self._model_a, model_a_ids, model_a_mask
        )
        model_b_scores = _infer_class1_probabilities_from_model(
            self._model_b, model_b_ids, model_b_mask
        )
        if len(model_a_scores) != len(validated["row_ids"]) or len(model_b_scores) != len(
            validated["row_ids"]
        ):
            raise ValueError("model output row count mismatch")
        return {
            "schema_version": OUTPUT_SCHEMA_VERSION,
            "row_ids": list(validated["row_ids"]),
            "model_a_scores": model_a_scores,
            "model_b_scores": model_b_scores,
        }


def execute_distilbert_pair_binary_scores(
    payload: Any,
    *,
    model_a_config: bytes,
    model_a_weights: bytes,
    model_b_config: bytes,
    model_b_weights: bytes,
) -> dict[str, Any]:
    fingerprint = distilbert_static_model_fingerprint_from_bytes(
        model_a_config,
        model_a_weights,
        model_b_config,
        model_b_weights,
    )
    pair = DistilbertStaticModelPair.from_verified_artifact_bytes(
        model_a_config=model_a_config,
        model_a_weights=model_a_weights,
        model_b_config=model_b_config,
        model_b_weights=model_b_weights,
        fingerprint=fingerprint,
    )
    return pair.execute_pair_binary_scores(payload)
