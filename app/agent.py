import os
import asyncio
import logging
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from datetime import date
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig

from app.conversation_manager import db
from app.user_store import user_store
from app.profile_store import profile_store

load_dotenv()

model = ChatOpenAI(
    model=os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
    temperature=0.5,
    base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
    api_key=os.getenv("OPENROUTER_API_KEY"),
    extra_body={"provider": {"ignore": ["Venice"]}},
)
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8001/sse")
MCP_PROMTES=[
    {
        "name": "portfolio_activity_summary",
        "description": (
            "Use ONLY when the user explicitly asks for a full portfolio summary, "
            "performance report, or deep analysis of their trading history. "
            "Do NOT use for simple data lookups like listing holdings, or getting latest price. "
        )
    },
]


def make_prompt_tool(client, name: str, description: str):
    @tool(f"load_{name}", description=description)
    async def prompt_tool() -> str:
        messages = await client.get_prompt("stocks", name)
        return "\n".join(m.content for m in messages)
    return prompt_tool


@tool("retrieve_user_profile")
def retrieve_user_profile(query: str, config: RunnableConfig) -> str:
    """
    Retrieve relevant information from the user's investment profile.
    Call this when the user asks about their preferences, risk tolerance,
    markets they invest in, experience, or investment style.
    Also call this before giving investment advice to personalize the response.
    """
    user_id = config.get("configurable", {}).get("user_id", "")
    chunks = profile_store.retrieve(user_id, query)
    if not chunks:
        logger.info(f"[TOOL] retrieve_user_profile | query='{query}' → no results")
        return "No profile information found. Do not retry this tool — proceed without profile context."
    result = profile_store.format_for_prompt(chunks)
    logger.info(f"[TOOL] retrieve_user_profile | query='{query}' → {[c['category'] for c in chunks]}")
    return result


@tool("update_user_profile")
def update_user_profile(category: str, text: str, config: RunnableConfig) -> str:
    """
    Update one aspect of the user's investment profile.
    Use this when the user explicitly states a change in their preferences.
    category must be one of: risk_tolerance, markets, experience, horizon, style.
    text should be a clear, concise description of the user's preference.
    Example: category='markets', text='User is now interested in both US and Israeli stocks.'
    """
    user_id = config.get("configurable", {}).get("user_id", "")
    try:
        profile_store.upsert(user_id, category, text)
        logger.info(f"[TOOL] update_user_profile | [{category}] {text}")
        return f"Profile updated: [{category}] {text}"
    except ValueError as e:
        logger.warning(f"[TOOL] update_user_profile | error: {e}")
        return f"Error: {e}"




async def init_agent():
    """Initialize MCP client, load tools, build agent. Call once at startup."""
    db.init_db()
    user_store.init_db()
    profile_store.init_db()

    client = MultiServerMCPClient({
        "stocks": {"url": MCP_SERVER_URL, "transport": "sse"}
    })
    tools = await client.get_tools()
    tools += [make_prompt_tool(client, p['name'], p['description']) for p in MCP_PROMTES]
    tools += [retrieve_user_profile, update_user_profile]

    ## logging to see the tools:
    if not tools:
        logger.warning("No tools found. Check MCP server connection and configuration.")
    logger.info("------  List of available tools: -------------------")
    for t in tools:
        logger.info(f"Tool: {t.name}")
    logger.info("------------------------------------")

    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=f"You are a stock assistant. Today's date is {date.today()}. Always use tools to get real data.",
    )
    return agent


async def astream_response(agent, user_id: str, conversation_id: str, prompt: str):
    """
    Load history for (user_id, conversation_id), stream agent events, then save the turn.

    Yields (event_type, data) tuples:
      ("tool_start", tool_name)
      ("tool_end",   tool_name)
      ("token",      chunk_str)
    """
    messages = db.load_history(user_id, conversation_id)
    messages.append(HumanMessage(prompt))
    logger.info(f"------ NEW TURN -----")
    logger.info(f"[TURN] user: prompt: {prompt[:80]!r}")

    response_text = ""
    tool_call_count = 0

    async for event in agent.astream_events(
        {"messages": messages},
        version="v2",
        config={"recursion_limit": 50, "configurable": {"user_id": user_id}},
    ):
        kind = event["event"]
        if kind == "on_tool_start":
            tool_call_count += 1
            params = {k: v for k, v in event.get("data", {}).get("input", {}).items() if k != "runtime"}
            params_str = f" {params}" if params else ""
            logger.info(f"[TOOL CALL #{tool_call_count}] → {event['name']}{params_str}")
            yield "tool_start", event["name"]
        elif kind == "on_tool_end":
            output = event.get("data", {}).get("output", "")
            output_preview = str(output)[:30].replace("\n", " ")
            logger.info(f"[TOOL DONE] ← {event['name']} | {output_preview}")
            yield "tool_end", event["name"]
        elif kind == "on_chat_model_stream":
            chunk = event["data"]["chunk"].content
            if chunk:
                response_text += chunk
                yield "token", chunk

    logger.info(f"[TURN DONE] {tool_call_count} tool calls | response: {len(response_text)} chars")
    db.save_turn(user_id, conversation_id, prompt, response_text)


# # ── Dev runner ────────────────────────────────────────────────────────────────
# async def _dev_run():
#     agent = await init_agent()
#     messages = [HumanMessage("read my transactions and list the companies I have invested in, for the top 3 holding show me the last closing price")]

#     async for event_type, data in astream_response(agent, messages):
#         if event_type == "tool_start":
#             print(f"\n[TOOL] {data}")
#         elif event_type == "token":
#             print(data, end="", flush=True)
#     print()

# if __name__ == "__main__":
#     asyncio.run(_dev_run())






