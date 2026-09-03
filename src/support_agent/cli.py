"""Command-line entry point: ``support-agent <command>``."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from support_agent.config import get_settings
from support_agent.db import create_demo_database
from support_agent.logging_config import configure_logging


def _seed_db(args: argparse.Namespace) -> int:
    path: Path = args.path or get_settings().database_path
    created = create_demo_database(path, reset=args.reset)
    if created:
        print(f"Seeded demo database at {path}")
    else:
        print(f"{path} already exists; pass --reset to recreate it")
    return 0


def _serve(args: argparse.Namespace) -> int:
    import uvicorn

    from support_agent.api import create_app

    settings = get_settings()
    configure_logging(settings.log_level)
    # log_config=None keeps uvicorn's loggers on the root handler, which masks PII.
    uvicorn.run(
        create_app(settings),
        host=args.host or settings.host,
        port=args.port or settings.port,
        log_config=None,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="support-agent", description="Multi-agent customer-support workflow."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    seed = commands.add_parser("seed-db", help="create the SQLite demo database")
    seed.add_argument("--path", type=Path, help="database file (default: SUPPORT_DATABASE_PATH)")
    seed.add_argument("--reset", action="store_true", help="delete and recreate the database")
    seed.set_defaults(handler=_seed_db)

    serve = commands.add_parser("serve", help="run the HTTP API")
    serve.add_argument("--host", help="bind address (default: SUPPORT_HOST)")
    serve.add_argument("--port", type=int, help="port (default: SUPPORT_PORT)")
    serve.set_defaults(handler=_serve)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
