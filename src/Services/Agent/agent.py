import os
import sqlite3
from pathlib import Path

from dotenv import load_dotenv
import certifi

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver
from src.Services.Agent.tools import tools

load_dotenv()
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
Path("data").mkdir(exist_ok=True)

DEFAULT_MODEL = os.getenv("GOOGLE_MODEL", "gemini-3.5-flash")
ALLOWED_MODELS = {
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.1-flash-lite",
}

SYSTEM_PROMPT = """
You are a helpful Agentic AI assistant named AstraGPT.

TOOL SELECTION (follow this order strictly):

1. TIME-SENSITIVE questions → ALWAYS use tavily_search
   Keywords: latest, news, current, today, now, recent, update, price, weather, who is, what is happening, version, release, 2024, 2025, 2026
   Example: "What's the latest news about AI?" → use tavily_search

2. Questions about UPLOADED DOCUMENTS → use the document content provided in the message
   Keywords: the document, this file, this PDF, uploaded, the report, the file says, in the document
   Example: "What does the document say about revenue?" → use the provided document content

3. MATH calculations → use calculator
   Example: "What is 15 * 37?" → use calculator

4. GENERAL knowledge questions → answer directly from your training data
   Example: "What is Python?" → answer directly

5. REMEMBER/RECALL → use remember_this or recall_memory

CRITICAL RULES:
- NEVER use search_uploaded_documents for time-sensitive questions. Use tavily_search instead.
- NEVER use search_uploaded_documents for general knowledge questions. Answer directly.
- The document content in the message is provided as REFERENCE. It does NOT mean you must use it for every question.
- If a question can be answered from your training data, answer directly without calling any tool.
- Only call search_uploaded_documents when the question SPECIFICALLY asks about the uploaded document.
"""



def get_model(user_model: str | None) -> str:
    if user_model in ALLOWED_MODELS:
        return user_model
    return DEFAULT_MODEL



def build_agent(model_name: str | None = None):
    selected_model = get_model(model_name)

    llm = ChatGoogleGenerativeAI(
        model=selected_model,
        temperature=0.2,
        streaming=True,
    )
    
    llm_with_tools = llm.bind_tools(tools)


    def chat_node(State: MessagesState):
        messages = [(SystemMessage(content=SYSTEM_PROMPT))] + State["messages"]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    tool_node = ToolNode(tools)
    workflow = StateGraph(MessagesState)

    workflow.add_node("chatbot", chat_node)
    workflow.add_node("tools", tool_node)

    workflow.add_edge(START, "chatbot")
    workflow.add_conditional_edges("chatbot", tools_condition)
    workflow.add_edge("tools", "chatbot")

    conn = sqlite3.connect(
        "data/langgraph_checkpoints.sqlite",
        check_same_thread=False
    )

    checkpointer = SqliteSaver(conn)

    return workflow.compile(checkpointer=checkpointer)


_AGENT_CACHE = {}


def get_agent(model_name: str | None = None):
    selected_model = get_model(model_name)

    if selected_model not in _AGENT_CACHE:
        _AGENT_CACHE[selected_model] = build_agent(selected_model)

    return _AGENT_CACHE[selected_model]
