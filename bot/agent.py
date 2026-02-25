import json
import logging
import os

from dotenv import load_dotenv
from mcp import ClientSession
from mcp.client.sse import sse_client
from openai import AsyncOpenAI

load_dotenv()

logger = logging.getLogger(__name__)

MCP_SERVER_URL = os.getenv("MCP_SERVER_URL")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL = os.getenv("OPENROUTER_MODEL")

openrouter = AsyncOpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
)

# Conversation history per user, keyed by phone number
conversation_histories: dict[str, list[dict]] = {}

SYSTEM_PROMPT = """You are a stock portfolio assistant. You have access to tools to:
- Get real-time and historical stock prices
- Read and analyze the user's trading history

Be concise and helpful. Format numbers clearly."""


def _mcp_tool_to_openai(tool) -> dict:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": tool.inputSchema,
        },
    }


async def run_agent(user_id: str, user_message: str) -> str:
    # Initialize history for new users
    if user_id not in conversation_histories:
        conversation_histories[user_id] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]

    history = conversation_histories[user_id]
    history.append({"role": "user", "content": user_message})

    async with sse_client(MCP_SERVER_URL) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # Fetch available tools from MCP server
            mcp_tools = await session.list_tools()
            openai_tools = [_mcp_tool_to_openai(t) for t in mcp_tools.tools]

            # Agent loop — keeps going until LLM gives a plain text response
            while True:
                response = await openrouter.chat.completions.create(
                    model=MODEL,
                    messages=history,
                    tools=openai_tools,
                    extra_body={"provider": {"ignore": ["Venice"]}},
                )

                message = response.choices[0].message

                # No tool calls → LLM is done, return the answer
                if not message.tool_calls:
                    reply = message.content or "Done."
                    logger.info(f"Final reply: {reply}")
                    history.append({"role": "assistant", "content": reply})
                    return reply

                # LLM wants to call tools — execute them and feed results back
                # Append as plain dict (not OpenAI object) to avoid serialization issues
                history.append({
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": tc.type,
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in message.tool_calls
                    ],
                })

                for tool_call in message.tool_calls:
                    tool_name = tool_call.function.name
                    tool_args = json.loads(tool_call.function.arguments)

                    logger.info(f"Tool call: {tool_name}({tool_args})")

                    result = await session.call_tool(tool_name, tool_args)

                    history.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": str(result.content),
                    })
