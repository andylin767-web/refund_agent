"""JSON-backed service for loading and updating orders."""

import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from typing import cast

from refund_agent.order import Order, OrderStatus


class OrderNotFoundError(LookupError):
    """Raised when an order ID is not present in the store."""


class OrderService:
    """Load and update orders stored in a JSON file."""

    def __init__(self, orders_path: Path) -> None:
        self._orders_path = orders_path

    def load_orders(self) -> dict[str, Order]:
        """Load and validate every order, preserving file order."""

        document = self._load_document()
        orders: dict[str, Order] = {}
        for order_id, record in document.items():
            if not isinstance(record, Mapping):
                raise ValueError(f"order {order_id!r} must be a JSON object")
            orders[order_id] = Order.from_record(order_id, record)
        return orders

    def load_order(self, order_id: str) -> Order:
        """Load one order by ID."""

        try:
            return self.load_orders()[order_id]
        except KeyError as error:
            raise OrderNotFoundError(f"order {order_id!r} was not found") from error

    def update_order(
        self,
        order_id: str,
        *,
        delivered_days_ago: int | None = None,
        status: OrderStatus | None = None,
        price: Decimal | None = None,
    ) -> Order:
        """Update selected fields and atomically persist the order file."""

        if delivered_days_ago is None and status is None and price is None:
            raise ValueError("at least one order field must be provided")

        orders = self.load_orders()
        try:
            current = orders[order_id]
        except KeyError as error:
            raise OrderNotFoundError(f"order {order_id!r} was not found") from error

        updated = replace(
            current,
            delivered_days_ago=(
                current.delivered_days_ago if delivered_days_ago is None else delivered_days_ago
            ),
            status=current.status if status is None else status,
            price=current.price if price is None else price,
        )
        orders[order_id] = updated
        self._write_orders(orders)
        return updated

    def _load_document(self) -> dict[str, object]:
        with self._orders_path.open(encoding="utf-8") as orders_file:
            document: object = json.load(orders_file, parse_float=Decimal)
        if not isinstance(document, dict) or not all(isinstance(key, str) for key in document):
            raise ValueError("order store must contain a JSON object")
        return cast(dict[str, object], document)

    def _write_orders(self, orders: Mapping[str, Order]) -> None:
        payload = {order_id: order.to_record() for order_id, order in orders.items()}
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                dir=self._orders_path.parent,
                prefix=f".{self._orders_path.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
                json.dump(payload, temporary_file, indent=2)
                temporary_file.write("\n")
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
            os.replace(temporary_path, self._orders_path)
        except BaseException:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            raise
