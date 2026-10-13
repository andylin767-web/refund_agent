from decimal import Decimal

from refund_agent.models import RefundDecision
from refund_agent.order import Order, OrderStatus
from refund_agent.policy import RefundPolicy
from refund_agent.service import RefundService


class RecordingGateway:
    def __init__(self) -> None:
        self.orders: list[Order] = []

    def issue_refund(self, order: Order) -> str:
        self.orders.append(order)
        return "refund-456"


def test_service_issues_approved_refund() -> None:
    gateway = RecordingGateway()
    candidate = Order("order-123", 10, OrderStatus.DELIVERED, Decimal("10"))

    result = RefundService(RefundPolicy(), gateway).process(candidate)

    assert result.decision is RefundDecision.APPROVED
    assert result.provider_reference == "refund-456"
    assert gateway.orders == [candidate]


def test_service_does_not_call_gateway_when_denied() -> None:
    gateway = RecordingGateway()
    candidate = Order("order-123", 10, OrderStatus.DELIVERED, Decimal("501"))

    result = RefundService(RefundPolicy(), gateway).process(candidate)

    assert result.decision is RefundDecision.DENIED
    assert gateway.orders == []
