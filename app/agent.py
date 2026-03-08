import os
import asyncio
from dotenv import load_dotenv
from langchain_core import messages
from langchain_core.tools import tool

from langchain_mcp_adapters import client
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

#from app.models import UserProfile
#from app.promts.promtes import strategy_template
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.messages import HumanMessage

load_dotenv()

model = ChatOpenAI(
    model=os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
    temperature=0.5,
    base_url="https://openrouter.ai/api/v1",
   
    api_key=os.getenv("OPENROUTER_API_KEY"),
)
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8001/sse")
MCP_PROMTES=[
    {
    "name": "portfolio_activity_summary",
    "description": "Guides the model to analyze and summarize the user's portfolio performance"
    },
    ]


def make_prompt_tool(client, name: str, description: str):
    @tool(f"load_{name}", description=description)
    async def prompt_tool() -> str:
        messages = await client.get_prompt("stocks", name)
        return "\n".join(m.content for m in messages)
    return prompt_tool




async def create_langchain_agent():
    client = MultiServerMCPClient({
        "stocks": {"url": MCP_SERVER_URL, "transport": "sse"}
        })
    tools = await client.get_tools()
    tools += [make_prompt_tool(client,p['name'],p['description']) for p in MCP_PROMTES]
    
        
    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt="You are a stock assistant. Always use tools to get real data.",
    )
    return agent


async def run_agent():

    
    ## connect to the MCP AND extract tools and promotes
    # client = MultiServerMCPClient({
    #     "stocks": {"url": MCP_SERVER_URL, "transport": "sse"}
    # })
    # tools = await client.get_tools()
    # tools += [make_prompt_tool(client,p['name'],p['description']) for p in MCP_PROMTES]

    agent=await create_langchain_agent()

    result = await agent.ainvoke({
        "messages": [HumanMessage("summerise my portfolio please?")]
    })


    print(f"\n  Final answer: {result['messages'][-1].content}")
if __name__ == "__main__":
    asyncio.run(run_agent())




# # ── User profile (simulates a filled form) ──────────────────────────────────
# # In production: loaded from Postgres by user ID.
# # In the Streamlit UI: populated from st.form() inputs.
# profile = UserProfile(
#     horizon="1-2 years",
#     risk_level="Moderate",
#     strategy="Growth",
#     markets="US (NYSE/NASDAQ)",
#     currency="USD",
#     sectors="AI & Machine Learning, SaaS, Energy",
#     max_allocation_per_stock=15,
#     min_positions=10,
#     max_positions=25,
#     rebalance_frequency="Quarterly",
#     benchmark="S&P 500",
#     esg_filter=False,
#     leverage_allowed=False,
#     custom_notes="",
# )

# # ── Build chain — profile is baked in, only {question} remains ──────────────
# chain = strategy_template.partial(**profile.model_dump()) | model | StrOutputParser()

# # ── Ask a question ───────────────────────────────────────────────────────────
# for chunk in chain.stream({"question": "What would be a good stock to buy right now?"}):
#     print(chunk, end="", flush=True)

# print()



