import os
import json
import asyncio
import queue
import threading
import logging
from typing import Generator

from dotenv import load_dotenv
from mcp import ClientSession
from mcp.client.sse import sse_client
from openai import OpenAI

load_dotenv()

logger = logging.getLogger(__name__)

MCP_SERVER_URL = os.getenv("MCP_SERVER_URL")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL")


def _tool_to_openai(tool) -> dict:
    """Convert an MCP tool definition to the OpenAI function-calling format."""
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": tool.inputSchema,
        },
    }


async def _run_agent_async(messages: list, chunk_queue: queue.Queue):
    """
    Async agentic loop.
    - Connects to the MCP server over SSE.
    - Calls OpenRouter with streaming, accumulating tool calls if any.
    - Executes tool calls on the MCP server and loops back until the LLM
      produces a plain text response.
    - Puts text chunks into chunk_queue as they arrive.
    - Puts None as a sentinel when done (or on error).

    NOTE: All intermediate messages (tool calls, tool results) are appended
    directly to the `messages` list. The final assistant text message is NOT
    appended here — the caller (Streamlit) handles that using the return value
    of st.write_stream().
    """
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
    )

    try:
        async with sse_client(MCP_SERVER_URL) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                tools_result = await session.list_tools()
                tools = [_tool_to_openai(t) for t in tools_result.tools]

                while True:
                    stream = client.chat.completions.create(
                        model=OPENROUTER_MODEL,
                        messages=messages,
                        tools=tools if tools else None,
                        stream=True,
                        extra_body={"provider": {"ignore": ["Venice"]}},
                    )

                    full_text = ""
                    tool_calls_acc: dict[int, dict] = {}

                    for chunk in stream:
                        if not chunk.choices:
                            continue
                        delta = chunk.choices[0].delta

                        # Stream text tokens to the UI
                        if delta.content:
                            full_text += delta.content
                            chunk_queue.put(delta.content)

                        # Accumulate tool call chunks (they arrive piece by piece)
                        if delta.tool_calls:
                            for tc in delta.tool_calls:
                                idx = tc.index
                                if idx not in tool_calls_acc:
                                    tool_calls_acc[idx] = {
                                        "id": "",
                                        "type": "function",
                                        "function": {"name": "", "arguments": ""},
                                    }
                                if tc.id:
                                    tool_calls_acc[idx]["id"] = tc.id
                                if tc.function and tc.function.name:
                                    tool_calls_acc[idx]["function"]["name"] += tc.function.name
                                if tc.function and tc.function.arguments:
                                    tool_calls_acc[idx]["function"]["arguments"] += tc.function.arguments

                    if tool_calls_acc:
                        # Append the assistant turn with tool_calls so the loop continues
                        tool_calls = list(tool_calls_acc.values())
                        messages.append({
                            "role": "assistant",
                            "content": full_text or None,
                            "tool_calls": tool_calls,
                        })

                        # Execute each tool and append results
                        for tc in tool_calls:
                            name = tc["function"]["name"]
                            try:
                                args = json.loads(tc["function"]["arguments"])
                            except json.JSONDecodeError:
                                args = {}

                            logger.info(f"Calling MCP tool '{name}' with args: {args}")
                            result = await session.call_tool(name, args)

                            result_text = "\n".join(
                                c.text for c in result.content if hasattr(c, "text")
                            )
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tc["id"],
                                "content": result_text,
                            })
                        # Loop back — LLM will now process the tool results
                    else:
                        # No tool calls: final response, exit the loop
                        break

    except Exception as e:
        logger.error(f"Agent error: {e}", exc_info=True)
        chunk_queue.put(f"\n\n[Error: {e}]")
    finally:
        chunk_queue.put(None)  # sentinel — tells the sync wrapper we're done


def stream_agent_response(messages: list) -> Generator[str, None, None]:
    """
    Sync generator compatible with st.write_stream().

    Runs the async agent in a background thread and yields text chunks
    as they arrive. The caller is responsible for appending the final
    assistant message to `messages` using the return value of st.write_stream().
    """
    chunk_queue: queue.Queue = queue.Queue()

    def _run_in_thread():
        asyncio.run(_run_agent_async(messages, chunk_queue))

    thread = threading.Thread(target=_run_in_thread, daemon=True)
    thread.start()

    while True:
        chunk = chunk_queue.get()
        if chunk is None:
            break
        yield chunk

    thread.join()
