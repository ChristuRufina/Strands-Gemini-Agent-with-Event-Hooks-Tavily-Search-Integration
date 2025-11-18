import asyncio
import os
from dotenv import load_dotenv
from strands import Agent
from strands.models.gemini import GeminiModel
from strands_tools.tavily import tavily_search
from strands.hooks import HookProvider, HookRegistry
from strands.hooks.events import (
    BeforeInvocationEvent, AfterInvocationEvent,
    BeforeModelCallEvent, AfterModelCallEvent,
    BeforeToolCallEvent, AfterToolCallEvent,
    MessageAddedEvent
)

# Load environment variables from .env if present
load_dotenv()

# Set API keys
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

def log_event(event):
    print("\n" + "="*80)
    print(f"🔹 EVENT: {type(event).__name__}")
    if hasattr(event, "tool_name"):
        print(f"Tool Name: {event.tool_name}")
    if hasattr(event, "args"):
        print(f"Args: {event.args}")
    if hasattr(event, "result"):
        print(f"Result: {event.result}")
    print(event)
    print("="*80)

class LoggingHooks(HookProvider):
    def register_hooks(self, registry: HookRegistry) -> None:
        registry.add_callback(BeforeInvocationEvent, log_event)
        registry.add_callback(AfterInvocationEvent, log_event)
        registry.add_callback(BeforeModelCallEvent, log_event)
        registry.add_callback(AfterModelCallEvent, log_event)
        registry.add_callback(BeforeToolCallEvent, log_event)
        registry.add_callback(AfterToolCallEvent, log_event)
        registry.add_callback(MessageAddedEvent, log_event)

model = GeminiModel(
    client_args={"api_key": GEMINI_API_KEY},
    model_id="gemini-2.5-flash",
    params={
        "temperature": 0.7,
        "max_output_tokens": 2048,
        "top_p": 0.9,
        "top_k": 40,
    },
)

agent = Agent(model=model, tools=[tavily_search], hooks=[LoggingHooks()])

async def main():
    query = "What are the latest advancements in artificial intelligence?"
    response = agent(f"Use tavily_search to answer: {query}")
    print("\n✅ FINAL BOT RESPONSE:\n", response)

if __name__ == "__main__":
    asyncio.run(main())
