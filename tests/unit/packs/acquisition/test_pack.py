from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest

from portable_batch_execution.packs.acquisition import AcquisitionPack

FIXTURES = Path(__file__).parents[4] / "fixtures" / "public" / "acquisition"


@pytest.fixture()
def fixture_server():
    requests: list[str] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append(self.path)
            path = urlsplit(self.path).path
            if path == "/rest/page-1":
                body = (FIXTURES / "rest-page-1.json").read_bytes()
            elif path == "/rest/page-2":
                body = (FIXTURES / "rest-page-2.json").read_bytes()
            elif path == "/html/page-1":
                body = (FIXTURES / "catalog.html").read_bytes()
            elif path == "/html/page-2":
                body = (FIXTURES / "catalog-page-2.html").read_bytes()
            elif path == "/incremental":
                since = int(parse_qs(urlsplit(self.path).query).get("since", [0])[0])
                body = json.dumps(
                    {"items": [{"id": n} for n in range(since + 1, 4)]}
                ).encode()
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header(
                "Content-Type",
                "application/json"
                if path.startswith("/rest") or path == "/incremental"
                else "text/html",
            )
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", requests
    finally:
        server.shutdown()
        thread.join()


def test_rest_link_pagination_is_finite_and_uses_local_server(fixture_server):
    base_url, requests = fixture_server
    result = AcquisitionPack().acquire(
        "acquisition.rest",
        {
            "url": f"{base_url}/rest/page-1",
            "items_path": "items",
            "max_pages": 4,
            "pagination": {"kind": "link", "next_field": "next"},
        },
    )
    assert [record["id"] for record in result.records] == [1, 2, 3]
    assert result.pages_fetched == 2
    assert requests == ["/rest/page-1", "/rest/page-2"]


def test_html_link_pagination_extracts_configured_fields(fixture_server):
    base_url, _ = fixture_server
    result = AcquisitionPack().acquire(
        "acquisition.html",
        {
            "url": f"{base_url}/html/page-1",
            "item_selector": "article.product",
            "fields": {"id": "@data-id", "name": "h2", "price": ".price"},
            "max_pages": 4,
            "pagination": {"kind": "link", "next_selector": "a.next"},
        },
    )
    assert list(result.records) == [
        {"id": "a1", "name": "Alpha", "price": "10"},
        {"id": "b2", "name": "Beta", "price": "20"},
        {"id": "c3", "name": "Gamma", "price": "30"},
    ]


def test_incremental_uses_and_advances_local_context_state(fixture_server):
    base_url, requests = fixture_server
    context = {"acquisition_state": {"catalog": 1}}
    params = {
        "url": f"{base_url}/incremental",
        "items_path": "items",
        "incremental": {
            "cursor_field": "id",
            "cursor_param": "since",
            "state_key": "catalog",
        },
    }
    result = AcquisitionPack().acquire("acquisition.incremental", params, context)
    assert [record["id"] for record in result.records] == [2, 3]
    assert context["acquisition_state"]["catalog"] == 3
    assert requests == ["/incremental?since=1"]


@pytest.mark.parametrize(
    "params",
    [
        {"url": "file:///not-http"},
        {"url": "http://user:pass@example.test/x"},
        {"url": "http://example.test/x", "max_pages": 0},
    ],
)
def test_validation_rejects_unbounded_or_credential_urls(params):
    with pytest.raises(ValueError):
        AcquisitionPack().validate_params("acquisition.rest", params)
