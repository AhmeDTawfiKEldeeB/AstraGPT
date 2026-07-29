from src.Services.Agent.agent import get_agent
from langchain_core.messages import SystemMessage, HumanMessage

agent = get_agent("gemini-2.5-flash")

config = {
    "configurable":{
        "thread_id": "test_thread_id",
    }}


for message_chunk,merge in agent.stream(
    
    {'messages':[HumanMessage(content="Generate a blog about machine learning and AI in 2024, include the latest trends and technologies, and make it engaging for readers.")]},
    config=config,
    stream_mode='messages'):

    if message_chunk.content:
        print(message_chunk.content,end='',flush=True)
    
