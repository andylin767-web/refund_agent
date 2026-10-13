from decimal import Decimal

import pytest

from refund_agent.order import Order, OrderStatus


def test_order_round_trips_json_record() -> None:
    order = Order.from_record(
        "123",
        {"delivered_days_ago": 10, "status": "delivered", "price": Decimal("25.00")},
    )

    assert order == Order("123", 10, OrderStatus.DELIVERED, Decimal("25.00"))
    assert order.to_record() == {
        "delivered_days_ago": 10,
        "status": "delivered",
        "price": 25.0,
    }


@pytest.mark.parametrize(
    ("record", "message"),
    [
        ({"status": "delivered", "price": 10}, "missing field"),
        (
            {"delivered_days_ago": "1", "status": "delivered", "price": 10},
            "non-negative integer",
        ),
        ({"delivered_days_ago": 1, "status": 2, "price": 10}, "status must be a string"),
        ({"delivered_days_ago": 1, "status": "pending", "price": 10}, "unsupported"),
        ({"delivered_days_ago": 1, "status": "delivered", "price": []}, "number"),
    ],
)
def test_order_rejects_invalid_record(record: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        Order.from_record("123", record)


@pytest.mark.parametrize(
    "order",
    [
        Order("123", 1, OrderStatus.DELIVERED, Decimal("1")),
    ],
)
def test_valid_order(order: Order) -> None:
    assert order.order_id == "123"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"order_id": ""}, "order_id"),
        ({"delivered_days_ago": -1}, "non-negative integer"),
        ({"delivered_days_ago": "1"}, "non-negative integer"),
        ({"status": "delivered"}, "OrderStatus"),
        ({"price": Decimal("0")}, "greater than zero"),
        ({"price": Decimal("NaN")}, "greater than zero"),
    ],
)
def test_order_rejects_invalid_values(kwargs: dict[str, object], message: str) -> None:
    values: dict[str, object] = {
        "order_id": "123",
        "delivered_days_ago": 1,
        "status": OrderStatus.DELIVERED,
        "price": Decimal("10"),
    }
    values.update(kwargs)
    with pytest.raises(ValueError, match=message):
        Order(**values)  # type: ignore[arg-type]
