import json
from pathlib import Path

import pytest

from refund_agent.cli import _decimal, build_parser, main


@pytest.fixture
def orders_path(tmp_path: Path) -> Path:
    path = tmp_path / "orders.json"
    path.write_text(
        json.dumps(
            {
                "order-123": {
                    "delivered_days_ago": 1,
                    "status": "delivered",
                    "price": 24.99,
                }
            }
        ),
        encoding="utf-8",
    )
    return path


def test_cli_prints_decision(orders_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(
        [
            "check",
            "--order-id",
            "order-123",
            "--orders-file",
            str(orders_path),
        ]
    )

    assert exit_code == 0
    assert json.loads(capsys.readouterr().out) == {
        "decision": "approved",
        "explanation": "order satisfies the refund policy",
        "order_id": "order-123",
    }


def test_parser_supports_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit, match="0"):
        build_parser().parse_args(["--version"])
    assert capsys.readouterr().out.strip() == "0.1.0"


def test_decimal_converter_rejects_invalid_value() -> None:
    with pytest.raises(Exception, match="must"):
        _decimal("abc")


def test_cli_reports_missing_order(orders_path: Path) -> None:
    with pytest.raises(SystemExit, match="2"):
        main(
            [
                "check",
                "--order-id",
                "missing",
                "--orders-file",
                str(orders_path),
            ]
        )
