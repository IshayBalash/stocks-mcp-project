import os
import sys
from dotenv import load_dotenv
from typing import List
import logging
from mcp.server.fastmcp import FastMCP

# Add project root to Python path to find the clients module
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from clients.polygon_io import PolygonIo
from general.utils.argument_clases.stocks_clases import StockBase,StockInfoByDate
from general.constans import STOCK_TRADES_CSV_FILE_PATH





# Configure logging with timestamp and professional format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

load_dotenv()
polygon_client = PolygonIo(os.getenv("POLYGON_API_KEY"))
mcp = FastMCP("Stock Get Info", port=8001)

# ============================================================
# 🧩 Tool 1 — Get stock data for a specific date range
# ============================================================
@mcp.tool()
def get_stock_value(stock: StockInfoByDate) -> List[dict]:
    """
    Retrieve daily stock data from Polygon.io for a given symbol and date range.

    Args:
        stock (StockInfoByDate): A validated Pydantic model containing:
            - symbol (str): Stock ticker (e.g. 'AAPL')
            - from_date (date): Start date in YYYY-MM-DD format
            - to_date (date): End date in YYYY-MM-DD format

    Returns:
        List[dict]: A list of dictionaries representing the stock's daily information.
    """
    logging.info("TOOL 'GET_STOCK_VALUE' IS NOW IN USE")
    res = polygon_client.get_stock_daily_data(
        symbol=stock.symbol,
        from_date=stock.from_date,
        to_date=stock.to_date
    )
    return res


# ============================================================
# 🧩 Tool 2 — Read local user exchange data
# ============================================================
@mcp.tool()
def read_user_exchanges_data() -> str:
    """
    Reads the user stock exchange transaction data from a local CSV file.

    Returns:
        str: The raw contents of the CSV file.
             If the file is empty, returns a message:
             "no records were found in stock_trades file".
             If an error occurs, returns the exception message.
    """
    logging.info("TOOL 'READ_USER_EXCHANGES_DATA' IS NOW IN USE")
    try:
        with open(STOCK_TRADES_CSV_FILE_PATH, "r") as f:
            text = f.read()
            return text if text else "no records were found in stock_trades file"
    except Exception as e:
        return f"An error occurred: {e}"


# ============================================================
# 🧩 Tool 3 — Get last closing stock price
# ============================================================
@mcp.tool()
def get_last_closing_stock_price(stock: StockBase) -> List[float]:
    """
    Returns the last recorded closing price for a given stock symbol using Polygon.io.

    Args:
        stock (StockBase): A validated Pydantic model containing:
            - symbol (str): Stock ticker (e.g. 'AAPL')

    Returns:
        List[float]: The last closing price wrapped in a list (e.g. [220.15]).
    """
    logging.info("TOOL 'GET_LAST_CLOSING_STOCK_PRICE' IS NOW IN USE")
    res = polygon_client.get_stock_last_close_price(stock.symbol)
    return res


# ============================================================
# 🧩 Tool 4 — Generate best user stock exgae promtes
# ============================================================
@mcp.prompt()
def generate_portfolio_analysis_prompt () -> str:
    """
    Return a formatted prompt for summarizing the user's stock portfolio.
    """
    return """
You are a professional financial analysis assistant.

The user will provide a CSV file representing their recent portfolio actions.
Each row in the CSV contains:
ticker, date, action, stock_amount, closing_day_stock_price

Your goal:
1. Analyze the user's overall performance and trading behavior.
2. Summarize each ticker separately:
   - Total buys vs. sells
   - Average buy/sell prices
   - Estimated profit or loss
   - Current holding status (if any shares remain)
3. Identify which stock performed best and worst.
4. End with a short natural-language summary of the user’s trading strategy or risk level.

Format the response as:
- A short paragraph summary per ticker.
- A final paragraph summarizing the portfolio as a whole.
"""



# ============================================================
# 🧩 Tool 5 — Combine prompt + user data for LLM call
# ============================================================
@mcp.tool()
def get_user_genral_view_on_portfolio() -> str:
    """
    1. Constructs the portfolio analysis prompt from Tool 4.
    2. Reads the user'sx stock exchange data from Tool 2.
    3. Returns a full prompt ready to send to an LLM for analysis.
    """
    logging.info("TOOL 'GET_USER_GENERAL_VIEW_ON_PORTFOLIO' IS NOW IN USE")
    # Step 1: Get the prompt template
    prompt_template = generate_portfolio_analysis_prompt()
    
    # Step 2: Read CSV data
    portfolio_csv = read_user_exchanges_data()
    
    # Step 3: Combine prompt + data
    full_prompt = f"{prompt_template}\n\nHere is the user's portfolio data:\n{portfolio_csv}"
    
    return full_prompt




