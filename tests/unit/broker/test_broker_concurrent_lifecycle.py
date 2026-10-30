"""Concurrency protections of the AF_UNIX Public broker."""

import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

from portable_batch_execution.broker.server import MAX_ACTIVE_CONNECTIONS
from portable_batch_execution.broker.service import (
    UnixBrokerService,
    _RequestIdSerializer,
)


def test_same_request_id_is_serialized_and_registry_released():
    serializer = _RequestIdSerializer()
    first_inside = threading.Event()
    release_first = threading.Event()
    second_inside = threading.Event()
    calls = []

    def first():
        with serializer.hold("req-1"):
            calls.append("first")
            first_inside.set()
            assert release_first.wait(3)

    def second():
        with serializer.hold("req-1"):
            calls.append("second")
            second_inside.set()

    with ThreadPoolExecutor(max_workers=2) as executor:
        a = executor.submit(first)
        assert first_inside.wait(3)
        b = executor.submit(second)
        assert not second_inside.wait(0.1)
        release_first.set()
        a.result(timeout=4)
        b.result(timeout=4)
    assert calls == ["first", "second"]
    assert serializer._locks == {}


def test_independent_ids_overlap_and_registry_released():
    serializer = _RequestIdSerializer()
    both_inside = threading.Barrier(2, timeout=4)

    def run(request_id):
        with serializer.hold(request_id):
            both_inside.wait()

    with ThreadPoolExecutor(max_workers=2) as executor:
        a = executor.submit(run, "req-A")
        b = executor.submit(run, "req-B")
        a.result(timeout=5)
        b.result(timeout=5)
    assert serializer._locks == {}


def test_actual_handle_message_is_serialized_for_valid_request_id():
    service = UnixBrokerService.__new__(UnixBrokerService)
    service._request_serializer = _RequestIdSerializer()
    entered = threading.Event()
    release = threading.Event()
    other = threading.Event()
    visits = []

    def run_inner(_uid, payload):
        visits.append(payload["request_id"])
        if len(visits) == 1:
            entered.set()
            assert release.wait(4)
        else:
            other.set()
        return b"ok\n"

    with (
        patch.object(service, "_handle_message_unlocked", side_effect=run_inner),
        ThreadPoolExecutor(max_workers=2) as executor,
    ):
        a = executor.submit(service.handle_message, 917, {"request_id": "req-same"})
        assert entered.wait(3)
        b = executor.submit(service.handle_message, 917, {"request_id": "req-same"})
        assert not other.wait(0.1)
        release.set()
        assert a.result(timeout=5) == b"ok\n"
        assert b.result(timeout=5) == b"ok\n"
    assert visits == ["req-same", "req-same"]
    assert service._request_serializer._locks == {}


def test_active_connection_cap_remains_bounded():
    assert 1 <= MAX_ACTIVE_CONNECTIONS <= 8
