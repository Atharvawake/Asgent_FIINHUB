from langchain_core.messages import HumanMessage, RemoveMessage
from app.config import get_llm


def format_content(content) -> str:
    """Extract clean text string from str or list-of-dicts content blocks."""
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


llm = get_llm()


def summarize_conversation(state):
    """
    Summarizes older messages when state message length exceeds threshold,
    and removes old individual message nodes to save context window tokens.
    """
    summary = state.get("summary", "")
    messages = state["messages"]

    if summary:
        summary_prompt = (
            f"This is summary of the conversation so far: {summary}\n\n"
            "Extend the summary by incorporating key info from new messages above:"
        )
    else:
        summary_prompt = "Create a concise summary of the conversation above:"

    # Keep the last 2 messages intact, summarize older ones
    messages_to_summarize = messages[:-2]

    if not messages_to_summarize:
        return {}

    response = llm.invoke(messages_to_summarize + [HumanMessage(content=summary_prompt)])
    
    # Format and clean response content before storing in graph state
    clean_summary = format_content(response.content)

    # Generate deletion commands for pruned message nodes
    delete_messages = [
        RemoveMessage(id=m.id) for m in messages_to_summarize if hasattr(m, "id") and m.id
    ]

    return {
        "summary": clean_summary,
        "messages": delete_messages,
    }