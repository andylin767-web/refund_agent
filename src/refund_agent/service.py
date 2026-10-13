"""Refund orchestration and external gateway boundary."""

from typing import Protocol

from refund_agent.models import RefundDecision, RefundResult
from refund_agent.order import Order
from refund_agent.policy import RefundPolicy


class RefundGateway(Protocol):
    """Interface implemented by a payment-provider adapter."""

    def issue_refund(self, order: Order) -> str:
        """Issue a refund and return the provider's reference."""
        ...


class RefundService:
    """Evaluate an order and issue approved refunds through a gateway."""

    def __init__(self, policy: RefundPolicy, gateway: RefundGateway) -> None:
        self._policy = policy
        self._gateway = gateway

    def process(self, order: Order) -> RefundResult:
        """Process one order, avoiding the gateway when policy denies it."""

        decision = self._policy.evaluate(order)
        if decision.decision is RefundDecision.DENIED:
            return decision

        reference = self._gateway.issue_refund(order)
        return RefundResult(
            RefundDecision.APPROVED,
            decision.explanation,
            provider_reference=reference,
        )
