from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from app.config import get_llm
from app.tools.finance_tools import finance_tools
from app.memory.summary_memory import summarize_conversation

# Define application state schema
class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    summary: str

# Bind tools to Gemini model
llm = get_llm().bind_tools(finance_tools)

def call_model(state: State):
    """Execution node for LLM inference."""
    summary = state.get("summary", "")
    messages = list(state["messages"])
    
    # If summary exists, inject it as initial System Context
    if summary:
        system_prompt = SystemMessage(content=f"Context Summary of prior conversation: {summary}")
        messages = [system_prompt] + messages
        
    response = llm.invoke(messages)
    return {"messages": [response]}

def should_summarize(state: State):
    """Conditional router to trigger summarization if message count exceeds limit."""
    if len(state["messages"]) > 6:
        return "summarize"
    return END

# Build Graph Workflow
workflow = StateGraph(State)

# Add Nodes
workflow.add_node("agent", call_model)
workflow.add_node("tools", ToolNode(finance_tools))
workflow.add_node("summarize", summarize_conversation)

# Add Edges
workflow.add_edge(START, "agent")

# Conditional Router: Agent -> Tool or Agent -> Summary/End
workflow.add_conditional_edges("agent", tools_condition)
workflow.add_edge("tools", "agent")
workflow.add_conditional_edges("agent", should_summarize, ["summarize", END])
workflow.add_edge("summarize", END)

# Compile executable app graph
chat_app = workflow.compile()