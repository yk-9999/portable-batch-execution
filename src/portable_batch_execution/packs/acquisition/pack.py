"""A small, bounded web acquisition pack.

The pack deliberately performs a finite number of ordinary HTTP requests.  It
does not start a crawler process, retain a connection, or schedule polling.
"""

from __future__ import annotations

from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Any
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit

import httpx

_OPERATIONS = ("acquisition.rest", "acquisition.html", "acquisition.incremental")
_MAX_PAGES = 1_000


@dataclass(frozen=True)
class AcquisitionResult:
    """Records acquired during one finite execution."""

    records: tuple[dict[str, Any], ...]
    pages_fetched: int
    next_cursor: Any | None = None


@dataclass
class _Node:
    tag: str
    attrs: dict[str, str]
    parent: _Node | None = None
    children: list[_Node] | None = None
    text_parts: list[str] | None = None

    def __post_init__(self) -> None:
        self.children = [] if self.children is None else self.children
        self.text_parts = [] if self.text_parts is None else self.text_parts

    def text(self) -> str:
        parts = list(self.text_parts)
        for child in self.children:
            parts.append(child.text())
        return " ".join(" ".join(parts).split())


class _HtmlTree(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.root = _Node("document", {})
        self.current = self.root

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        node = _Node(tag, {key: value or "" for key, value in attrs}, self.current)
        self.current.children.append(node)
        self.current = node

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        self.handle_endtag(tag)

    def handle_endtag(self, tag: str) -> None:
        node = self.current
        while node.parent is not None:
            if node.tag == tag:
                self.current = node.parent
                return
            node = node.parent

    def handle_data(self, data: str) -> None:
        self.current.text_parts.append(data)


def _descendants(node: _Node) -> list[_Node]:
    found: list[_Node] = []
    for child in node.children:
        found.append(child)
        found.extend(_descendants(child))
    return found


def _matches(node: _Node, selector: str) -> bool:
    if selector.startswith("#"):
        return node.attrs.get("id") == selector[1:]
    if selector.startswith("."):
        return selector[1:] in node.attrs.get("class", "").split()
    if "." in selector:
        tag, class_name = selector.split(".", 1)
        return node.tag == tag and class_name in node.attrs.get("class", "").split()
    return node.tag == selector


def _select(node: _Node, selector: str) -> list[_Node]:
    """Select a deliberately small CSS subset: tag, .class, #id, and ancestry."""
    current = [node]
    for part in selector.split():
        current = [
            candidate
            for parent in current
            for candidate in _descendants(parent)
            if _matches(candidate, part)
        ]
    return current


def _nested(value: Any, path: str | None, default: Any = None) -> Any:
    if not path:
        return value
    current = value
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def _with_query(url: str, key: str, value: Any) -> str:
    parsed = urlsplit(url)
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    query[key] = str(value)
    return urlunsplit(
        (parsed.scheme, parsed.netloc, parsed.path, urlencode(query), parsed.fragment)
    )


def _safe_url(value: Any, name: str = "url") -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    parsed = urlsplit(value)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.username
        or parsed.password
    ):
        raise ValueError(f"{name} must be an absolute HTTP URL without credentials")
    return value


class AcquisitionPack:
    pack_id = "acquisition-batch"
    supported_operations = _OPERATIONS

    def validate_params(self, operation: str, params: dict) -> dict:
        if operation not in self.supported_operations:
            raise ValueError("unsupported acquisition operation")
        if not isinstance(params, dict):
            raise TypeError("acquisition parameters must be a dictionary")
        validated = dict(params)
        validated["url"] = _safe_url(validated.get("url"))
        max_pages = validated.get("max_pages", 100)
        if (
            isinstance(max_pages, bool)
            or not isinstance(max_pages, int)
            or not 1 <= max_pages <= _MAX_PAGES
        ):
            raise ValueError(f"max_pages must be an integer from 1 to {_MAX_PAGES}")
        validated["max_pages"] = max_pages
        timeout = validated.get("timeout_seconds", 10.0)
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not 0 < timeout <= 60
        ):
            raise ValueError("timeout_seconds must be between 0 and 60")
        validated["timeout_seconds"] = float(timeout)
        headers = validated.get("headers", {})
        if not isinstance(headers, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in headers.items()
        ):
            raise ValueError("headers must be string pairs")
        validated["headers"] = headers
        pagination = validated.get("pagination", {"kind": "none"})
        if not isinstance(pagination, dict) or pagination.get("kind", "none") not in {
            "none",
            "link",
            "page",
            "cursor",
        }:
            raise ValueError("pagination.kind must be none, link, page, or cursor")
        validated["pagination"] = dict(pagination)
        if operation == "acquisition.html" and not isinstance(
            validated.get("item_selector"), str
        ):
            raise ValueError("item_selector is required for HTML acquisition")
        if operation == "acquisition.incremental":
            incremental = validated.get("incremental")
            if not isinstance(incremental, dict) or not isinstance(
                incremental.get("cursor_field"), str
            ):
                raise ValueError("incremental.cursor_field is required")
            if not isinstance(incremental.get("cursor_param", "since"), str):
                raise ValueError("incremental.cursor_param must be a string")
        return validated

    def execute(
        self, job: Any, shard: Any, params: dict, context: Any
    ) -> AcquisitionResult:
        operation = getattr(job, "operation", None) or params.get(
            "operation", "acquisition.rest"
        )
        validated = self.validate_params(operation, params)
        return self.acquire(operation, validated, context)

    def acquire(
        self, operation: str, params: dict, context: Any = None
    ) -> AcquisitionResult:
        params = self.validate_params(operation, params)
        state, state_key, cursor_field, prior_cursor = self._incremental_state(
            operation, params, context
        )
        if operation == "acquisition.incremental" and prior_cursor is not None:
            params["url"] = _with_query(
                params["url"],
                params["incremental"].get("cursor_param", "since"),
                prior_cursor,
            )

        records: list[dict[str, Any]] = []
        url = params["url"]
        next_cursor = prior_cursor
        with httpx.Client(
            timeout=params["timeout_seconds"],
            headers=params["headers"],
            follow_redirects=True,
        ) as client:
            for page_number in range(params["max_pages"]):
                response = client.get(url)
                response.raise_for_status()
                page_records, payload = self._records(
                    operation,
                    response.text,
                    response.headers.get("content-type", ""),
                    params,
                )
                if operation == "acquisition.incremental" and cursor_field:
                    page_records = [
                        item
                        for item in page_records
                        if self._newer(item.get(cursor_field), prior_cursor)
                    ]
                    for item in page_records:
                        candidate = item.get(cursor_field)
                        if self._newer(candidate, next_cursor):
                            next_cursor = candidate
                records.extend(page_records)
                url = self._next_url(
                    url, payload, params["pagination"], page_number, bool(page_records)
                )
                if url is None:
                    break
        if state is not None and state_key is not None and next_cursor is not None:
            state[state_key] = next_cursor
        return AcquisitionResult(tuple(records), page_number + 1, next_cursor)

    def finalize(self, job: Any, canonical_attempts: Any, context: Any) -> None:
        """Acquisition has no cross-shard finalization work."""

    @staticmethod
    def _incremental_state(
        operation: str, params: dict, context: Any
    ) -> tuple[dict | None, str | None, str | None, Any]:
        if operation != "acquisition.incremental":
            return None, None, None, None
        incremental = params["incremental"]
        state_key = incremental.get("state_key", params["url"])
        if not isinstance(state_key, str):
            raise TypeError("incremental.state_key must be a string")
        if not isinstance(context, dict):
            return (
                {},
                state_key,
                incremental["cursor_field"],
                incremental.get("start_value"),
            )
        state = context.setdefault("acquisition_state", {})
        if not isinstance(state, dict):
            raise TypeError("context.acquisition_state must be a dictionary")
        return (
            state,
            state_key,
            incremental["cursor_field"],
            state.get(state_key, incremental.get("start_value")),
        )

    @staticmethod
    def _newer(candidate: Any, prior: Any) -> bool:
        if candidate is None:
            return False
        if prior is None:
            return True
        try:
            return candidate > prior
        except TypeError:
            return str(candidate) > str(prior)

    @staticmethod
    def _records(
        operation: str, body: str, content_type: str, params: dict
    ) -> tuple[list[dict[str, Any]], Any]:
        if operation == "acquisition.html":
            tree = _HtmlTree()
            tree.feed(body)
            items = []
            fields = params.get("fields", {})
            if not isinstance(fields, dict):
                raise ValueError("fields must be a dictionary")
            for node in _select(tree.root, params["item_selector"]):
                item: dict[str, Any] = {}
                for name, selector in fields.items():
                    if not isinstance(name, str) or not isinstance(selector, str):
                        raise TypeError("fields must map strings to strings")
                    if selector == "text":
                        item[name] = node.text()
                    elif selector.startswith("@"):
                        item[name] = node.attrs.get(selector[1:])
                    else:
                        matches = _select(node, selector)
                        item[name] = matches[0].text() if matches else None
                items.append(item if fields else {"text": node.text()})
            return items, tree
        try:
            payload = __import__("json").loads(body)
        except ValueError as exc:
            raise ValueError("REST acquisition requires a JSON response") from exc
        items = _nested(payload, params.get("items_path"), payload)
        if not isinstance(items, list) or not all(
            isinstance(item, dict) for item in items
        ):
            raise ValueError("items_path must resolve to a list of objects")
        return items, payload

    @staticmethod
    def _next_url(
        current_url: str,
        payload: Any,
        pagination: dict,
        page_number: int,
        has_records: bool,
    ) -> str | None:
        kind = pagination.get("kind", "none")
        if kind == "none":
            return None
        if kind == "page":
            if not has_records:
                return None
            if (
                isinstance(payload, dict)
                and pagination.get("has_more_field")
                and not _nested(payload, pagination["has_more_field"])
            ):
                return None
            return _with_query(
                current_url,
                pagination.get("param", "page"),
                page_number + pagination.get("start", 1) + 1,
            )
        if kind == "cursor":
            cursor = _nested(
                payload, pagination.get("next_cursor_field", "next_cursor")
            )
            return (
                _with_query(
                    current_url, pagination.get("cursor_param", "cursor"), cursor
                )
                if cursor not in (None, "")
                else None
            )
        if kind == "link":
            link = (
                _nested(payload, pagination.get("next_field", "next"))
                if isinstance(payload, dict)
                else None
            )
            if link is None and isinstance(payload, _HtmlTree):
                selector = pagination.get("next_selector", "a.next")
                matches = _select(payload.root, selector)
                link = matches[0].attrs.get("href") if matches else None
            if not isinstance(link, str) or not link:
                return None
            return _safe_url(urljoin(current_url, link), "pagination link")
        return None
