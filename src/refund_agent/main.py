"""Interactive terminal entry point for the refund agent."""

import asyncio

from refund_agent.agent import RefundAgent
from refund_agent.config import load_model_id

EXIT_COMMANDS = frozenset({"exit", "quit"})


def read_user_message() -> str:
    """Read and normalize one user message from standard input."""

    return input("You: ").strip()


async def run_chat(agent: RefundAgent) -> None:
    """Process terminal messages until the user exits the session."""

    while True:
        try:
            user_message = read_user_message()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            return

        if user_message.casefold() in EXIT_COMMANDS:
            print("Goodbye.")
            return
        if not user_message:
            print("Please enter a message, or type 'quit' or 'exit'.")
            continue

        result = await agent.run_agent(user_message)
        if result.status is None:
            print(f"Refund Agent: {result.message}")
        else:
            print(f"Refund Agent [{result.status.value}]: {result.message}")


def main() -> None:
    """Load configuration and start the interactive refund-agent loop."""

    agent = RefundAgent(model=load_model_id())
    asyncio.run(run_chat(agent))


if __name__ == "__main__":
    main()
