import json
from decimal import Decimal
from pathlib import Path

import pytest

from refund_agent.order import OrderStatus
from refund_agent.order_service import OrderNotFoundError, OrderService


@pytest.fixture
def orders_path(tmp_path: Path) -> Path:
    path = tmp_path / "orders.json"
    path.write_text(
        json.dumps(
            {
                "123": {
                    "delivered_days_ago": 10,
                    "status": "delivered",
                    "price": 25.00,
                },
                "789": {
                    "delivered_days_ago": 5,
                    "status": "refunded",
                    "price": 15.00,
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def test_load_order(orders_path: Path) -> None:
    order = OrderService(orders_path).load_order("123")

    assert order.order_id == "123"
    assert order.status is OrderStatus.DELIVERED
    assert order.price == Decimal("25.0")


def test_load_orders_preserves_order(orders_path: Path) -> None:
    orders = OrderService(orders_path).load_orders()

    assert list(orders) == ["123", "789"]


def test_update_order_persists_selected_fields(orders_path: Path) -> None:
    service = OrderService(orders_path)

    updated = service.update_order("123", status=OrderStatus.REFUNDED, price=Decimal("20.00"))

    assert updated.status is OrderStatus.REFUNDED
    assert updated.price == Decimal("20.00")
    assert updated.delivered_days_ago == 10
    assert service.load_order("123") == updated
    assert service.load_order("789").status is OrderStatus.REFUNDED
    assert not list(orders_path.parent.glob(".orders.json.*.tmp"))


def test_missing_order_raises(orders_path: Path) -> None:
    service = OrderService(orders_path)

    with pytest.raises(OrderNotFoundError, match="missing"):
        service.load_order("missing")
    with pytest.raises(OrderNotFoundError, match="missing"):
        service.update_order("missing", price=Decimal("10"))


def test_update_requires_a_field(orders_path: Path) -> None:
    with pytest.raises(ValueError, match="at least one"):
        OrderService(orders_path).update_order("123")


def test_load_rejects_non_object_root(tmp_path: Path) -> None:
    path = tmp_path / "orders.json"
    path.write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="JSON object"):
        OrderService(path).load_orders()


def test_load_rejects_non_object_order(tmp_path: Path) -> None:
    path = tmp_path / "orders.json"
    path.write_text('{"123": []}', encoding="utf-8")

    with pytest.raises(ValueError, match="must be a JSON object"):
        OrderService(path).load_orders()
