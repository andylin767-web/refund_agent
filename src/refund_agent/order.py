"""Order model used by the JSON-backed order service."""

from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from enum import StrEnum


class OrderStatus(StrEnum):
    """Statuses currently supported by the sample order store."""

    DELIVERED = "delivered"
    REFUNDED = "refunded"


@dataclass(frozen=True, slots=True)
class Order:
    """A validated order loaded from the order store."""

    order_id: str
    delivered_days_ago: int
    status: OrderStatus
    price: Decimal

    def __post_init__(self) -> None:
        if not self.order_id.strip():
            raise ValueError("order_id must not be empty")
        if (
            not isinstance(self.delivered_days_ago, int)
            or isinstance(self.delivered_days_ago, bool)
            or self.delivered_days_ago < 0
        ):
            raise ValueError("delivered_days_ago must be a non-negative integer")
        if not isinstance(self.status, OrderStatus):
            raise ValueError("status must be an OrderStatus")
        if (
            not isinstance(self.price, Decimal)
            or not self.price.is_finite()
            or self.price <= Decimal("0")
        ):
            raise ValueError("price must be greater than zero")

    @classmethod
    def from_record(cls, order_id: str, record: Mapping[str, object]) -> "Order":
        """Create an order from one JSON object."""

        try:
            delivered_days_ago = record["delivered_days_ago"]
            status = record["status"]
            price = record["price"]
        except KeyError as error:
            raise ValueError(f"order {order_id!r} is missing field {error.args[0]!r}") from error

        if not isinstance(delivered_days_ago, int) or isinstance(delivered_days_ago, bool):
            raise ValueError("delivered_days_ago must be a non-negative integer")
        if not isinstance(status, str):
            raise ValueError("status must be a string")
        if isinstance(price, bool) or not isinstance(price, (Decimal, int, float, str)):
            raise ValueError("price must be a number")

        try:
            parsed_price = Decimal(str(price))
        except InvalidOperation as error:
            raise ValueError("price must be a number") from error

        try:
            parsed_status = OrderStatus(status)
        except ValueError as error:
            raise ValueError(f"unsupported order status: {status!r}") from error

        return cls(order_id, delivered_days_ago, parsed_status, parsed_price)

    def to_record(self) -> dict[str, object]:
        """Convert the order to its JSON-compatible representation."""

        return {
            "delivered_days_ago": self.delivered_days_ago,
            "status": self.status.value,
            "price": float(self.price),
        }
