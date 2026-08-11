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

SYSTEM_PROMPT = """You are AstraGPT, a helpful AI assistant.

TOOL RULES (strict order):
1. News/prices/weather/current events → tavily_search (NEVER search_uploaded_documents)
2. About uploaded document → use document content in message
3. Math → calculator
4. General knowledge → answer directly
5. Remember/recall → remember_this / recall_memory

Document content in message is REFERENCE ONLY. Do not use it for non-document questions."""



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
        max_retries=2,
        request_timeout=60,
        thinking_budget=0,
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
