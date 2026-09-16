import base64
import json
from unittest.mock import patch

import httpx
import pytest

from portable_batch_execution.backends.github_actions import GitHubActionsBackend
from portable_batch_execution.broker.config import BrokerConfig
from portable_batch_execution.broker.planning import canonical_operation_params
from portable_batch_execution.broker.protocol import BrokerExecuteResponse
from portable_batch_execution.broker.service import UnixBrokerService
from portable_batch_execution.controller.a1_controller import A1Controller
from portable_batch_execution.packs.ml import MLPack

_PUBLIC_SHA = "ac3a69d2c818526b87f38c848d324221e2dc2775"
_UID = 1000

_CLOSED_ML_OPS = (
    "ml.char_wb_tfidf_logistic_score",
    "ml.cosine_similarity_matrix",
)


@pytest.mark.parametrize("operation", _CLOSED_ML_OPS)
def test_closed_ml_operations_accept_empty_params(operation: str):
    assert canonical_operation_params("ml-batch", operation, {}) == {}


@pytest.mark.parametrize("operation", _CLOSED_ML_OPS)
def test_closed_ml_operations_reject_non_empty_params(operation: str):
    with pytest.raises(ValueError, match="closed operation parameters"):
        canonical_operation_params("ml-batch", operation, {"extra": 1})


def test_legacy_ml_operation_still_uses_mlpack_validation():
    validated = canonical_operation_params(
        "ml-batch", "ml.tfidf", {"max_features": 8}
    )
    assert validated == MLPack().validate_params("ml.tfidf", {"max_features": 8})


def test_char_wb_service_request_passes_param_validation(tmp_path):
    config_path = tmp_path / "broker-config.json"
    config_path.write_text(
        json.dumps(
            {
                "schema_version": "pbe.a1-unix-broker.config.v1",
                "public_sha": _PUBLIC_SHA,
                "max_input_bytes": 1_048_576,
                "allowed_operations_by_uid": {
                    str(_UID): [
                        ["ml-batch", "ml.char_wb_tfidf_logistic_score"],
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    config = BrokerConfig.load(config_path)
    backend = GitHubActionsBackend(
        "owner",
        "repo",
        "execute-wave.yml",
        private_data_plane=True,
        client=httpx.Client(
            base_url="https://api.github.com",
            transport=httpx.MockTransport(lambda request: httpx.Response(500)),
        ),
    )
    controller = A1Controller(tmp_path, backend=backend)
    service = UnixBrokerService(
        state_root=tmp_path,
        config=config,
        controller=controller,
        poll_interval_seconds=0.0,
    )
    ml_input = json.dumps(
        {
            "schema_version": "pbe.ml.char-wb-tfidf-logistic-score.v1",
            "model": {
                "features": [{"feature": "ab", "idf": 1.0, "coefficient": 0.0}],
                "intercept": 0.0,
            },
            "rows": [{"row_id": "a", "text": "ab"}],
        }
    ).encode()
    request = {
        "schema_version": "pbe.a1-unix-broker.request.v1",
        "request_id": "req-char-wb",
        "pack": "ml-batch",
        "operation": "ml.char_wb_tfidf_logistic_score",
        "operation_params": {},
        "input_media_type": "application/json",
        "input_b64": base64.b64encode(ml_input).decode("ascii"),
    }
    stub = BrokerExecuteResponse(request_id="req-char-wb", status="succeeded")
    with patch.object(
        UnixBrokerService, "_drive_to_terminal", return_value=stub
    ):
        response = service.handle_payload(_UID, request)
    assert response.error_code != "operation_params_invalid"
    assert response.status == "succeeded"
