import os
import sys
from pathlib import Path
from datetime import datetime, timezone
from dotenv import load_dotenv
from typing import List
import logging
from mcp.server.fastmcp import FastMCP

# Add project root to Python path to find the clients module
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from clients.polygon_io import PolygonIo
from clients.finnhub_client import FinnhubClient
from general.utils.argument_clases.stocks_clases import StockBase, StockInfoByDate
from general.constans import STOCK_TRADES_CSV_FILE_PATH


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_prompt(name: str) -> str:
    return (Path(__file__).parent / "promts" / f"{name}.md").read_text()


load_dotenv()
polygon_client = PolygonIo(os.getenv("POLYGON_API_KEY"))
finnhub_client = FinnhubClient(os.getenv("FINNHUB_API_KEY"))
mcp = FastMCP("Stock Get Info")


# ============================================================
# Tool 1 — Read the user's personal trade history (CSV)
# ============================================================
@mcp.tool()
def read_user_exchanges_data() -> str:
    """
    Read the user's personal stock trade history from a local CSV file.

    Use this whenever the user asks about their own portfolio, holdings, transactions,
    or performance — before doing any portfolio analysis or calculation.

    The CSV columns are: ticker, date, action (buy/sell), stock_amount, closing_day_stock_price.

    Returns:
        str: Raw CSV content, or a message if the file is empty or unreadable.
    """
    logger.info("[TOOL] read_user_exchanges_data | reading CSV")
    try:
        with open(STOCK_TRADES_CSV_FILE_PATH, "r") as f:
            text = f.read()
            lines = len(text.strip().splitlines()) - 1 if text.strip() else 0
            logger.info(f"[TOOL] read_user_exchanges_data | {lines} trade records found")
            return text if text else "no records were found in stock_trades file"
    except Exception as e:
        logger.warning(f"[TOOL] read_user_exchanges_data | error: {e}")
        return f"An error occurred: {e}"


# ============================================================
# Prompt — Portfolio activity summary
# ============================================================
@mcp.prompt()
def portfolio_activity_summary() -> str:
    """Guides the model to analyze and summarize the user's portfolio performance."""
    return load_prompt("portfolio_activity_summary")


# ============================================================
# Tool 2 — Historical daily price data for a date range (Polygon)
# ============================================================
@mcp.tool()
def get_stock_value(stock: StockInfoByDate) -> List[dict]:
    """
    Retrieve historical daily price data (open, high, low, close, volume) for a stock symbol over a date range.

    Use this when the user asks about price history, trends, performance over time,
    or needs data to compare across multiple dates (e.g. "how did AAPL do last month?").
    Do NOT use this just to get the current price — use get_last_closing_stock_price instead.

    Args:
        stock (StockInfoByDate):
            - symbol (str): Stock ticker (e.g. 'AAPL')
            - from_date (date): Start date in YYYY-MM-DD format
            - to_date (date): End date in YYYY-MM-DD format

    Returns:
        List[dict]: One entry per trading day, each with:
            date, open, high, low, close, volume

    IMPORTANT: Call this tool ONCE per stock with a single wide date range.
    Do NOT call it multiple times with smaller ranges for the same stock.
    """
    logger.info(f"[TOOL] get_stock_value | {stock.symbol} {stock.from_date} → {stock.to_date}")
    try:
        raw = polygon_client.get_stock_daily_data(
            symbol=stock.symbol,
            from_date=stock.from_date,
            to_date=stock.to_date
        )
        res = [
            {
                "date": datetime.fromtimestamp(r["timestamp"] / 1000, tz=timezone.utc).strftime("%Y-%m-%d"),
                "close": r.get("close"),
                "open": r.get("open"),
                "high": r.get("high"),
                "low": r.get("low"),
                "volume": r.get("volume"),
            }
            for r in raw
        ]
        logger.info(f"[TOOL] get_stock_value | {stock.symbol} → {len(res)} records")
        return res
    except Exception as e:
        logger.warning(f"[TOOL] get_stock_value | {stock.symbol} error: {e}")
        return f"Error fetching data for {stock.symbol}: {e}"


# ============================================================
# Tool 3 — Last closing price for a symbol (Polygon)
# ============================================================
@mcp.tool()
def get_last_closing_stock_price(stock: StockBase) -> List[float]:
    """
    Get the most recent closing price for a stock symbol.

    Use this for quick, single-price lookups (e.g. "what is TSLA trading at?").
    For price history over a date range, use get_stock_value instead.

    Args:
        stock (StockBase):
            - symbol (str): Stock ticker (e.g. 'AAPL')

    Returns:
        List[float]: The last closing price, e.g. [220.15].
    """
    logger.info(f"[TOOL] get_last_closing_stock_price | {stock.symbol}")
    try:
        res = polygon_client.get_stock_last_close_price(stock.symbol)
        logger.info(f"[TOOL] get_last_closing_stock_price | {stock.symbol} → {res}")
        return res
    except Exception as e:
        logger.warning(f"[TOOL] get_last_closing_stock_price | {stock.symbol} error: {e}")
        return f"Error fetching price for {stock.symbol}: {e}"


# ============================================================
# Tool 4 — Company profile (Finnhub)
# ============================================================
@mcp.tool()
def get_company_profile(stock: StockBase) -> dict:
    """
    Get company profile information for a stock symbol.

    Use this when the user asks what a company does, what sector/industry it's in,
    its market cap, exchange, website, or country of origin.

    Args:
        stock (StockBase):
            - symbol (str): Stock ticker (e.g. 'AAPL')

    Returns:
        dict: Company name, exchange, industry, market cap, country, website.
    """
    logger.info(f"[TOOL] get_company_profile | {stock.symbol}")
    try:
        result = finnhub_client.get_company_profile(stock.symbol)
        result.pop("logo", None)
        logger.info(f"[TOOL] get_company_profile | {stock.symbol} → ok")
        return result
    except Exception as e:
        logger.warning(f"[TOOL] get_company_profile | {stock.symbol} error: {e}")
        return {"error": str(e)}


# ============================================================
# Tool 5 — Recent company news (Finnhub)
# ============================================================
@mcp.tool()
def get_company_news(stock: StockInfoByDate) -> list:
    """
    Get recent news articles for a company over a date range.

    Use this when the user asks about recent news, events, sentiment, or anything
    that may have affected a stock recently (e.g. "what's been happening with NVDA?").

    Args:
        stock (StockInfoByDate):
            - symbol (str): Stock ticker (e.g. 'AAPL')
            - from_date (date): Start date in YYYY-MM-DD format
            - to_date (date): End date in YYYY-MM-DD format

    Returns:
        list[dict]: News articles, each with: headline, summary, source, url, datetime.
    """
    logger.info(f"[TOOL] get_company_news | {stock.symbol} {stock.from_date} → {stock.to_date}")
    try:
        result = finnhub_client.get_company_news(
            stock.symbol,
            str(stock.from_date),
            str(stock.to_date),
        )
        logger.info(f"[TOOL] get_company_news | {stock.symbol} → {len(result)} articles")
        return result
    except Exception as e:
        logger.warning(f"[TOOL] get_company_news | {stock.symbol} error: {e}")
        return [{"error": str(e)}]


# ============================================================
# Tool 6 — Key financial metrics (Finnhub)
# ============================================================
@mcp.tool()
def get_basic_financials(stock: StockBase) -> dict:
    """
    Get fundamental financial metrics for a stock symbol.

    Use this when the user asks about valuation (P/E, EPS), risk (beta),
    dividend yield, 52-week high/low, or other key financial ratios.
    Do NOT use this for price history — use get_stock_value for that.

    Args:
        stock (StockBase):
            - symbol (str): Stock ticker (e.g. 'AAPL')

    Returns:
        dict: Metrics including P/E ratio, EPS, 52-week high/low, beta, dividend yield, and more.
    """
    logger.info(f"[TOOL] get_basic_financials | {stock.symbol}")
    try:
        result = finnhub_client.get_basic_financials(stock.symbol)
        logger.info(f"[TOOL] get_basic_financials | {stock.symbol} → {len(result)} metrics")
        return result
    except Exception as e:
        logger.warning(f"[TOOL] get_basic_financials | {stock.symbol} error: {e}")
        return {"error": str(e)}
