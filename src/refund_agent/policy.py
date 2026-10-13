"""Deterministic refund eligibility policy."""

from dataclasses import dataclass
from decimal import Decimal

from refund_agent.models import RefundDecision, RefundResult
from refund_agent.order import Order, OrderStatus


@dataclass(frozen=True, slots=True)
class RefundPolicy:
    """Evaluate order value objects using status, age, and amount limits."""

    max_age_days: int = 30
    max_amount: Decimal = Decimal("500.00")

    def __post_init__(self) -> None:
        if self.max_age_days < 0:
            raise ValueError("max_age_days must not be negative")
        if self.max_amount <= Decimal("0"):
            raise ValueError("max_amount must be greater than zero")

    def evaluate(self, order: Order) -> RefundResult:
        """Return an eligibility decision without causing side effects."""

        if order.status is OrderStatus.REFUNDED:
            return RefundResult(RefundDecision.DENIED, "order has already been refunded")
        if order.delivered_days_ago > self.max_age_days:
            return RefundResult(RefundDecision.DENIED, "refund window has expired")
        if order.price > self.max_amount:
            return RefundResult(RefundDecision.DENIED, "amount exceeds the automatic refund limit")

        return RefundResult(RefundDecision.APPROVED, "order satisfies the refund policy")
