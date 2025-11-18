# Strands-Gemini-Agent-with-Event-Hooks-Tavily-Search-Integration
A fully-functional Python agent built using the Strands Agentic Framework, integrated with Google Gemini 2.5 Flash and Tavily Search for real-time web intelligence. This project demonstrates how to use event hooks to monitor and log every stage of agent execution ,including model calls, tool calls, message updates, and invocation flow.

Key Features
🔹 Gemini 2.5 Flash as the primary LLM
🔹 Tavily Search tool integration for live web search
🔹 Complete event logging using Strands hook system
🔹 Logs all stages:
Before/After Invocation
Before/After Model Call
Before/After Tool Call
Message Added events
🔹 Clean async runner using asyncio
🔹 Environment variable support with .env
🔹 Ready-to-use boilerplate for building advanced agent workflows

📂 Ideal For
Developers exploring Agentic AI
Anyone learning Strands LLM agents
Debugging or teaching how agent event pipelines work
Building future multi-tool, multi-step AI workflows

Just add your GEMINI_API_KEY and TAVILY_API_KEY run:
python app.py
