from refund_agent import REFUND_AGENT_SYSTEM_MESSAGE


def test_system_message_defines_tools_and_safe_refund_workflow() -> None:
    normalized_message = " ".join(REFUND_AGENT_SYSTEM_MESSAGE.split())

    assert "get_order(order_id)" in REFUND_AGENT_SYSTEM_MESSAGE
    assert "issue_refund(order_id)" in REFUND_AGENT_SYSTEM_MESSAGE
    assert "explicitly requests the refund" in REFUND_AGENT_SYSTEM_MESSAGE
    assert "if the order status is refunded, do not call issue_refund" in normalized_message
    assert "Return already_refunded immediately" in REFUND_AGENT_SYSTEM_MESSAGE
    assert "does not contact a payment provider" in normalized_message
    assert "Never guess an order ID" in REFUND_AGENT_SYSTEM_MESSAGE


def test_system_message_is_ready_for_api_use() -> None:
    assert REFUND_AGENT_SYSTEM_MESSAGE.startswith("You are Refund Agent")
    assert REFUND_AGENT_SYSTEM_MESSAGE.strip() == REFUND_AGENT_SYSTEM_MESSAGE
