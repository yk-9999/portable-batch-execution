"""CLI for the bounded A1 private data plane controller."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from portable_batch_execution.backends.github_actions import GitHubActionsBackend
from portable_batch_execution.broker.server import serve_unix_broker
from portable_batch_execution.broker.service import build_service_from_environment
from portable_batch_execution.controller.a1_controller import A1Controller
from portable_batch_execution.data_plane.http_server import (
    serve_private_data_plane_from_environment,
)

_GITHUB_TOKEN_FILE_ERROR = "github token file is not available"


def _read_nonempty_utf8_secret_file(path: str) -> str:
    try:
        raw = Path(path).read_text(encoding="utf-8")
    except OSError:
        raise ValueError(_GITHUB_TOKEN_FILE_ERROR) from None
    token = raw.strip()
    if not token:
        raise ValueError(_GITHUB_TOKEN_FILE_ERROR)
    return token


def _github_token_from_args(args: argparse.Namespace) -> str | None:
    path = getattr(args, "github_token_file", None)
    if path is None:
        return None
    return _read_nonempty_utf8_secret_file(path)


def _controller(args: argparse.Namespace) -> A1Controller:
    backend = None
    github_owner = getattr(args, "github_owner", None)
    github_repo = getattr(args, "github_repo", None)
    github_workflow = getattr(args, "github_workflow", None)
    if github_owner and github_repo and github_workflow:
        token = _github_token_from_args(args)
        backend_kwargs: dict[str, object] = {
            "dispatch_ref": getattr(args, "github_ref", "main"),
            "private_data_plane": True,
        }
        if token is not None:
            backend_kwargs["token"] = token
        backend = GitHubActionsBackend(
            github_owner,
            github_repo,
            github_workflow,
            **backend_kwargs,
        )
    return A1Controller(Path(args.state_root), backend=backend)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="A1 private batch controller")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser(
        "serve-data-plane", help="Run loopback private data plane HTTP"
    )
    serve.set_defaults(command="serve-data-plane")

    prepare = sub.add_parser(
        "prepare-synthetic", help="Register a private synthetic run"
    )
    prepare.add_argument("--state-root", required=True)
    prepare.add_argument("--run-id")
    prepare.add_argument("--wave-id")
    prepare.set_defaults(command="prepare-synthetic")

    dispatch = sub.add_parser("dispatch", help="Dispatch one private wave")
    dispatch.add_argument("--state-root", required=True)
    dispatch.add_argument("--run-id", required=True)
    dispatch.add_argument("--wave-id", required=True)
    dispatch.add_argument("--github-owner", required=True)
    dispatch.add_argument("--github-repo", required=True)
    dispatch.add_argument("--github-workflow", required=True)
    dispatch.add_argument("--github-ref", default="main")
    dispatch.add_argument("--github-token-file")
    dispatch.set_defaults(command="dispatch")

    inspect = sub.add_parser("inspect", help="Inspect controller state")
    inspect.add_argument("--state-root", required=True)
    inspect.add_argument("--run-id", required=True)
    inspect.add_argument("--github-owner")
    inspect.add_argument("--github-repo")
    inspect.add_argument("--github-workflow")
    inspect.add_argument("--github-ref", default="main")
    inspect.add_argument("--github-token-file")
    inspect.set_defaults(command="inspect")

    reconcile = sub.add_parser("reconcile", help="Refresh canonical manifest")
    reconcile.add_argument("--state-root", required=True)
    reconcile.add_argument("--run-id", required=True)
    reconcile.set_defaults(command="reconcile")

    broker = sub.add_parser(
        "serve-unix-broker",
        help="Run the A1-local Unix-domain execution broker",
    )
    broker.add_argument("--state-root", required=True)
    broker.add_argument("--socket-path", required=True)
    broker.add_argument("--poll-interval-seconds", type=float, default=1.0)
    broker.add_argument("--github-owner", required=True)
    broker.add_argument("--github-repo", required=True)
    broker.add_argument("--github-workflow", required=True)
    broker.add_argument("--github-ref", default="main")
    broker.add_argument("--github-token-file")
    broker.add_argument(
        "--max-concurrent-requests",
        type=int,
        default=1,
        metavar="N",
        help="Maximum in-flight broker requests (default: 1, serial)",
    )
    broker.add_argument(
        "--artifact-read-base-url",
        help="Optional HTTPS origin or loopback HTTP origin for broker output artifact reads",
    )
    broker.add_argument(
        "--artifact-read-token-file",
        help="Bearer token file for --artifact-read-base-url (never logged)",
    )
    broker.set_defaults(command="serve-unix-broker")

    args = parser.parse_args(argv)

    if args.command == "serve-data-plane":
        server = serve_private_data_plane_from_environment()
        host, port = server.server_address
        print(json.dumps({"bind": f"{host}:{port}"}))
        server.serve_forever()
        return 0

    if args.command == "prepare-synthetic":
        prepared = _controller(args).prepare_private_synthetic_run(
            logical_run_id=args.run_id,
            wave_id=args.wave_id,
        )
        print(
            json.dumps(
                {
                    "logical_run_id": prepared.logical_run_id,
                    "wave_id": prepared.wave_id,
                    "manifest_revision": prepared.manifest.revision,
                }
            )
        )
        return 0

    if args.command == "dispatch":
        execution = _controller(args).dispatch_private_wave(args.run_id, args.wave_id)
        print(
            json.dumps(
                {
                    "backend_id": execution.backend_id,
                    "execution_id": execution.execution_id,
                }
            )
        )
        return 0

    if args.command == "inspect":
        payload = _controller(args).inspect_run(args.run_id)
        print(json.dumps(payload))
        return 0

    if args.command == "reconcile":
        manifest = _controller(args).reconcile_run(args.run_id)
        print(json.dumps({"revision": manifest.revision, "status": manifest.status}))
        return 0

    if args.command == "serve-unix-broker":
        import os

        os.environ.setdefault(
            "PBE_BROKER_CONFIG", os.environ.get("PBE_BROKER_CONFIG", "")
        )
        if not os.environ.get("PBE_BROKER_CONFIG"):
            parser.error("serve-unix-broker requires PBE_BROKER_CONFIG")
        backend = _controller(args).backend
        artifact_read_token_file = (
            Path(args.artifact_read_token_file)
            if args.artifact_read_token_file
            else None
        )
        if bool(args.artifact_read_base_url) != bool(artifact_read_token_file):
            parser.error(
                "serve-unix-broker requires both --artifact-read-base-url and "
                "--artifact-read-token-file when configuring artifact reads"
            )
        service = build_service_from_environment(
            state_root=Path(args.state_root),
            backend=backend,
            poll_interval_seconds=args.poll_interval_seconds,
            artifact_read_base_url=args.artifact_read_base_url,
            artifact_read_token_file=artifact_read_token_file,
        )
        if args.max_concurrent_requests < 1:
            parser.error("--max-concurrent-requests must be at least 1")
        serve_unix_broker(
            socket_path=Path(args.socket_path),
            service=service,
            socket_mode=service.config.socket_mode,
            max_concurrent_requests=args.max_concurrent_requests,
        )
        return 0

    parser.error("unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
