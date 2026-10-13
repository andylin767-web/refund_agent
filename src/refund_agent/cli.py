"""Command-line interface for safe, local policy evaluation."""

import argparse
import json
from collections.abc import Sequence
from decimal import Decimal, InvalidOperation
from pathlib import Path

from refund_agent import __version__
from refund_agent.order_service import OrderNotFoundError, OrderService
from refund_agent.policy import RefundPolicy


def _decimal(value: str) -> Decimal:
    try:
        return Decimal(value)
    except InvalidOperation as error:
        raise argparse.ArgumentTypeError("must be a decimal number") from error


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser without initializing external services."""

    parser = argparse.ArgumentParser(prog="refund-agent")
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    check = subparsers.add_parser("check", help="evaluate an order without issuing a refund")
    check.add_argument("--order-id", required=True)
    check.add_argument("--orders-file", type=Path, default=Path("orders.json"))
    check.add_argument("--max-age-days", type=int, default=30)
    check.add_argument("--max-amount", type=_decimal, default=Decimal("500.00"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Evaluate a stored order and print a machine-readable decision."""

    args = build_parser().parse_args(argv)
    try:
        order = OrderService(args.orders_file).load_order(args.order_id)
        result = RefundPolicy(
            max_age_days=args.max_age_days,
            max_amount=args.max_amount,
        ).evaluate(order)
    except (OSError, ValueError, OrderNotFoundError) as error:
        build_parser().error(str(error))

    print(
        json.dumps(
            {
                "decision": result.decision,
                "explanation": result.explanation,
                "order_id": order.order_id,
            },
            sort_keys=True,
        )
    )
    return 0
