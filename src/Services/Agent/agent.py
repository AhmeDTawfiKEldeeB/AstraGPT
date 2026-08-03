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
You are a helpful Agentic AI assistant named BappyGPT similar to ChatGPT.

You can:
1. Answer normal questions.
2. Use tools when needed.
3. Search uploaded documents using the RAG tool.
4. Search the web for latest/current information using Tavily Search.
5. Remember important user information using the memory tool.
6. Recall memory when useful.
7. Use calculator for math.

Rules:
- If the user asks about latest news, current events, recent updates, today's information, current prices, current people, current versions, new releases, or anything time-sensitive, use Tavily Search.
- If the user asks about an uploaded document, use search_uploaded_documents.
- If the user asks you to remember something, use remember_this.
- If the user asks about previous preferences or saved facts, use recall_memory.
- Use calculator for math questions.
- When using web search, summarize clearly and answer this question.
- Be clear, helpful, and concise.
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
