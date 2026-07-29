import os
import requests

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from sympy import sympify
from sympy.core.sympify import SympifyError

from src.infrastructure.sqlalchemy_database import (
    save_memory,
    search_memory,
)
from src.Services.Rag.rag_service import retrieve_context

load_dotenv()

CURRENT_THREAD_ID = "default"


def set_current_thread_id(thread_id: str):
    global CURRENT_THREAD_ID
    CURRENT_THREAD_ID = thread_id


tavily_tool = TavilySearch(
    max_results=5,
    topic="general",
    include_answer=True,
    include_images=True,
)


@tool
def get_weather(city: str) -> str:
    """
    Get the current weather for a city using the OpenWeatherMap API.
    """

    api_key = os.getenv("OPENWEATHER_API_KEY")

    response = requests.get(
        "https://api.openweathermap.org/data/2.5/weather",
        params={
            "q": city,
            "appid": api_key,
            "units": "metric",
        },
        timeout=10,
    )

    response.raise_for_status()
    data = response.json()

    return (
        f"Weather in {data['name']}, {data['sys']['country']}:\n"
        f"Condition: {data['weather'][0]['description'].title()}\n"
        f"Temperature: {data['main']['temp']}°C\n"
        f"Feels Like: {data['main']['feels_like']}°C\n"
        f"Humidity: {data['main']['humidity']}%\n"
        f"Pressure: {data['main']['pressure']} hPa\n"
        f"Wind Speed: {data['wind']['speed']} m/s"
    )


@tool
def calculator(expression: str) -> str:
    """
    Evaluate mathematical expressions.
    """

    try:
        return str(sympify(expression).evalf())

    except SympifyError:
        return "Invalid mathematical expression."

    except Exception as e:
        return f"Calculation error: {e}"


@tool
def search_uploaded_documents(query: str) -> str:
    """
    Search uploaded documents for relevant information.

    Use this tool when the user asks about uploaded PDFs,
    DOCX, TXT, Markdown files, notes, or other uploaded documents.
    """

    return retrieve_context(
        query=query,
        thread_id=CURRENT_THREAD_ID,
    )


@tool
def remember_this(memory: str) -> str:
    """
    Save an important user preference or fact into long-term memory.
    Use this when the user asks you to remember something.
    """

    return save_memory(
        thread_id=CURRENT_THREAD_ID,
        memory=memory,
    )


@tool
def recall_memory(query: str) -> str:
    """
    Recall saved long-term memories about the user or this conversation.
    """

    return search_memory(
        thread_id=CURRENT_THREAD_ID,
        query=query,
    )


tools = [
    tavily_tool,
    get_weather,
    calculator,
    search_uploaded_documents,
    remember_this,
    recall_memory,
]