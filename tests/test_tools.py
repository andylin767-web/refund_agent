import json
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import ValidationError

from refund_agent.order import OrderStatus
from refund_agent.tools import (
    GetOrderInput,
    RefundOutcome,
    get_order,
    issue_refund,
)


@pytest.fixture
def orders_path(tmp_path: Path) -> Path:
    path = tmp_path / "orders.json"
    path.write_text(
        json.dumps(
            {
                "eligible": {
                    "delivered_days_ago": 10,
                    "status": "delivered",
                    "price": 25.00,
                },
                "old": {
                    "delivered_days_ago": 45,
                    "status": "delivered",
                    "price": 40.00,
                },
                "done": {
                    "delivered_days_ago": 5,
                    "status": "refunded",
                    "price": 15.00,
                },
                "expensive": {
                    "delivered_days_ago": 5,
                    "status": "delivered",
                    "price": 501.00,
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def test_get_order_returns_validated_output(orders_path: Path) -> None:
    result = get_order("eligible", orders_path=orders_path)

    assert result.order_id == "eligible"
    assert result.status is OrderStatus.DELIVERED
    assert result.price == Decimal("25.0")


@pytest.mark.parametrize("order_id", ["", "   ", 123])
def test_get_order_validates_input(order_id: object, orders_path: Path) -> None:
    with pytest.raises(ValidationError):
        get_order(order_id, orders_path=orders_path)  # type: ignore[arg-type]


def test_input_schema_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError, match="Extra inputs"):
        GetOrderInput.model_validate({"order_id": "123", "unsafe": True})


def test_issue_refund_updates_eligible_order(orders_path: Path) -> None:
    result = issue_refund("eligible", orders_path=orders_path)

    assert result.outcome is RefundOutcome.REFUNDED
    assert result.order.status is OrderStatus.REFUNDED
    assert get_order("eligible", orders_path=orders_path).status is OrderStatus.REFUNDED
    assert "no payment provider" in result.message


def test_issue_refund_denies_old_order_without_update(orders_path: Path) -> None:
    result = issue_refund("old", orders_path=orders_path)

    assert result.outcome is RefundOutcome.DENIED
    assert result.order.status is OrderStatus.DELIVERED
    assert get_order("old", orders_path=orders_path).status is OrderStatus.DELIVERED


def test_issue_refund_uses_policy_amount_limit(orders_path: Path) -> None:
    result = issue_refund("expensive", orders_path=orders_path)

    assert result.outcome is RefundOutcome.DENIED
    assert result.message == "amount exceeds the automatic refund limit"
    assert get_order("expensive", orders_path=orders_path).status is OrderStatus.DELIVERED


def test_issue_refund_is_idempotent(orders_path: Path) -> None:
    result = issue_refund("done", orders_path=orders_path)

    assert result.outcome is RefundOutcome.ALREADY_REFUNDED
    assert result.order.status is OrderStatus.REFUNDED
