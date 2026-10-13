"""Pydantic-validated tools exposed to a refund agent."""

from decimal import Decimal
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from refund_agent.models import RefundDecision
from refund_agent.order import Order, OrderStatus
from refund_agent.order_service import OrderService
from refund_agent.policy import RefundPolicy

DEFAULT_REFUND_WINDOW_DAYS = 30


class ToolInput(BaseModel):
    """Shared validation behavior for tool inputs."""

    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)


class GetOrderInput(ToolInput):
    """Input schema for ``get_order``."""

    order_id: str = Field(min_length=1, description="Unique order identifier")


class IssueRefundInput(ToolInput):
    """Input schema for ``issue_refund``."""

    order_id: str = Field(min_length=1, description="Unique order identifier")


class OrderOutput(BaseModel):
    """Structured order data returned by a tool."""

    model_config = ConfigDict(frozen=True)

    order_id: str
    delivered_days_ago: int
    status: OrderStatus
    price: Decimal

    @classmethod
    def from_order(cls, order: Order) -> "OrderOutput":
        """Build a validated output from the domain model."""

        return cls(
            order_id=order.order_id,
            delivered_days_ago=order.delivered_days_ago,
            status=order.status,
            price=order.price,
        )


class RefundOutcome(StrEnum):
    """Possible outcomes reported by the refund tool or agent."""

    REFUNDED = "refunded"
    ALREADY_REFUNDED = "already_refunded"
    DENIED = "denied"
    NEEDS_ORDER_ID = "needs_order_id"
    ORDER_NOT_FOUND = "order_not_found"


class RefundOutput(BaseModel):
    """Structured result returned by ``issue_refund``."""

    model_config = ConfigDict(frozen=True)

    order: OrderOutput
    outcome: RefundOutcome
    message: str


def get_order(order_id: str, *, orders_path: Path = Path("orders.json")) -> OrderOutput:
    """Load one order after validating the tool arguments with Pydantic."""

    tool_input = GetOrderInput.model_validate({"order_id": order_id})
    order = OrderService(orders_path).load_order(tool_input.order_id)
    return OrderOutput.from_order(order)


def issue_refund(
    order_id: str,
    *,
    orders_path: Path = Path("orders.json"),
    refund_window_days: int = DEFAULT_REFUND_WINDOW_DAYS,
) -> RefundOutput:
    """Record an eligible mock refund without contacting a payment provider.

    The update is idempotent: issuing a refund for an already-refunded order
    returns a successful no-op result.
    """

    tool_input = IssueRefundInput.model_validate({"order_id": order_id})
    service = OrderService(orders_path)
    order = service.load_order(tool_input.order_id)

    if order.status is OrderStatus.REFUNDED:
        return RefundOutput(
            order=OrderOutput.from_order(order),
            outcome=RefundOutcome.ALREADY_REFUNDED,
            message="order was already refunded; no change was made",
        )

    decision = RefundPolicy(max_age_days=refund_window_days).evaluate(order)
    if decision.decision is RefundDecision.DENIED:
        return RefundOutput(
            order=OrderOutput.from_order(order),
            outcome=RefundOutcome.DENIED,
            message=decision.explanation,
        )

    updated = service.update_order(order.order_id, status=OrderStatus.REFUNDED)
    return RefundOutput(
        order=OrderOutput.from_order(updated),
        outcome=RefundOutcome.REFUNDED,
        message="mock refund recorded; no payment provider was contacted",
    )
