"""Domain models for refund requests and outcomes."""

from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from enum import StrEnum


class RefundDecision(StrEnum):
    """Possible outcomes of an eligibility check."""

    APPROVED = "approved"
    DENIED = "denied"


@dataclass(frozen=True, slots=True)
class RefundRequest:
    """Information required to evaluate a refund request."""

    order_id: str
    amount: Decimal
    purchased_on: date
    reason: str

    def __post_init__(self) -> None:
        if not self.order_id.strip():
            raise ValueError("order_id must not be empty")
        if self.amount <= Decimal("0"):
            raise ValueError("amount must be greater than zero")
        if not self.reason.strip():
            raise ValueError("reason must not be empty")


@dataclass(frozen=True, slots=True)
class RefundResult:
    """A policy decision and optional provider reference."""

    decision: RefundDecision
    explanation: str
    provider_reference: str | None = None
