import math
import requests
import os
from dotenv import load_dotenv
from sympy import sympify
from sympy.core.sympify import SympifyError
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from infrastructure.sqlalchemy_database import save_memory, search_memory
from rag import retrieve_from_rag

load_dotenv()
# tool for Searching the web using Tavily Search
tavily_tool = TavilySearch(
    max_results=5,
    topic="general",        # "general" or "news"
    include_answer=True,
    include_images=True,
)
#tool for get real-time weather information using OpenWeatherMap API
@tool
def get_weather(city: str) -> str:
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

# tool for calculating mathematical expressions
@tool
def calculator(expression: str) -> str:
    try:
        result = sympify(expression).evalf()
        return str(result)
    except SympifyError:
        return "Invalid mathematical expression."
    except Exception as e:
        return f"Calculation error: {e}"

     