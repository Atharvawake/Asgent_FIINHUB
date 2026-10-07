from langchain_core.messages import HumanMessage
from app.agents.chat_agent import chat_app
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="langchain_google_genai")


def format_content(content) -> str:
    """Extract clean string text from str or list-of-dicts content blocks."""
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(block.get("text", ""))
            elif isinstance(block, str):
                text_parts.append(block)
        return "".join(text_parts)
    return str(content)


def run_chat():
    print("=" * 65)
    print("🤖 Agentic Chat Model Initialized (Gemini + Finnhub + Memory)")
    print("Type 'exit' or 'quit' to end the session.")
    print("=" * 65)

    config = {"configurable": {"thread_id": "session_1"}}
    state = {"messages": [], "summary": ""}

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            print("\nEnding session. Goodbye!")
            break

        if not user_input:
            continue

        state["messages"].append(HumanMessage(content=user_input))

        print("\nAI: ", end="", flush=True)

        # Collect streamed content tokens to maintain full state history
        accumulated_response = ""

        # Stream messages directly from the LangGraph execution
        for msg, metadata in chat_app.stream(
            state, config=config, stream_mode="messages"
        ):
            # Verify the chunk contains content from the AI model
            if msg.content:
                clean_chunk = format_content(msg.content)
                print(clean_chunk, end="", flush=True)
                accumulated_response += clean_chunk

        print()  # Add newline after response stream finishes

        # Update the state history for subsequent turns
        state = chat_app.get_state(config).values


if __name__ == "__main__":
    run_chat()