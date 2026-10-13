from decimal import Decimal

import pytest

from refund_agent.models import RefundDecision
from refund_agent.order import Order, OrderStatus
from refund_agent.policy import RefundPolicy


def order(
    *,
    delivered_days_ago: int = 10,
    status: OrderStatus = OrderStatus.DELIVERED,
    price: str = "10.00",
) -> Order:
    return Order("order-123", delivered_days_ago, status, Decimal(price))


@pytest.mark.parametrize(
    ("candidate", "expected"),
    [
        (order(status=OrderStatus.REFUNDED), "order has already been refunded"),
        (order(delivered_days_ago=31), "refund window has expired"),
        (order(price="501.00"), "amount exceeds the automatic refund limit"),
    ],
)
def test_policy_denies_ineligible_orders(candidate: Order, expected: str) -> None:
    result = RefundPolicy().evaluate(candidate)

    assert result.decision is RefundDecision.DENIED
    assert result.explanation == expected


def test_policy_approves_order_within_limits() -> None:
    result = RefundPolicy().evaluate(order())

    assert result.decision is RefundDecision.APPROVED
    assert result.explanation == "order satisfies the refund policy"
    assert result.provider_reference is None


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_age_days": -1}, "max_age_days must not be negative"),
        ({"max_amount": Decimal("0")}, "max_amount must be greater than zero"),
    ],
)
def test_policy_rejects_invalid_configuration(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        RefundPolicy(**kwargs)  # type: ignore[arg-type]
