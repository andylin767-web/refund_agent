from datetime import date
from decimal import Decimal

import pytest

from refund_agent.models import RefundRequest


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("order_id", " ", "order_id must not be empty"),
        ("amount", Decimal("0"), "amount must be greater than zero"),
        ("reason", "", "reason must not be empty"),
    ],
)
def test_request_rejects_invalid_values(field: str, value: object, message: str) -> None:
    values: dict[str, object] = {
        "order_id": "order-123",
        "amount": Decimal("10.00"),
        "purchased_on": date(2026, 9, 1),
        "reason": "damaged",
    }
    values[field] = value

    with pytest.raises(ValueError, match=message):
        RefundRequest(**values)  # type: ignore[arg-type]
