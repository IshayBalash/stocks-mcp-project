# User Stocks Project

A comprehensive stock portfolio analysis tool built with Python that provides MCP (Model Context Protocol) server capabilities for stock data retrieval and portfolio analysis using the Polygon.io API. 




## Overview

This project offers a powerful MCP server that enables:
- Real-time stock data retrieval from Polygon.io
- Portfolio analysis based on user trading history
- Stock price tracking and historical data analysis
- Automated portfolio performance reporting

## Features

### 🔧 MCP Tools Available

1. **`get_stock_value`** - Retrieve daily stock data for any symbol within a specified date range
2. **`read_user_exchanges_data`** - Read and analyze your local stock trading CSV file
3. **`get_last_closing_stock_price`** - Get the most recent closing price for any stock
4. **`get_user_genral_view_on_portfolio`** - Generate comprehensive portfolio analysis prompts

### 📊 Portfolio Analysis

The system automatically analyzes your trading history to provide:
- Performance summary per ticker
- Buy vs. sell analysis
- Profit/loss calculations
- Trading strategy insights
- Risk assessment

## Project Structure

```
user_stocks_project/
├── src/
│   └── mcpserver/
│       ├── __init__.py
│       ├── __main__.py
│       └── server.py          # Main MCP server implementation
├── clients/
│   └── polygon_io.py          # Polygon.io API client
├── general/
│   ├── constans.py           # Configuration constants
│   └── utils/
│       ├── argument_clases/
│       │   └── stocls_clases.py  # Pydantic models
│       └── fake_user_dat_func.py
├── my_stock_trades.csv       # Example trading history file
├── .env                      # Environment variables (required)
├── pyproject.toml           # Project configuration
└── README.md
```

## Requirements

### Prerequisites




- Python 3.11 or higher
- A valid Polygon.io API key
- Stock trading history in the required CSV format

### Dependencies

The project uses the following main dependencies:
- `mcp[cli]>=1.18.0` - Model Context Protocol framework
- `polygon-api-client>=1.15.4` - Polygon.io API client
- `pandas>=2.3.3` - Data manipulation
- `numpy>=2.3.4` - Numerical computing
- `python-dotenv>=0.9.9` - Environment variable management

## Configuration Requirements

### 1. Polygon.io API Key

You **must** have a valid Polygon.io API key configured in your `.env` file:

```env
POLYGON_API_KEY=your_polygon_api_key_here
```

To get a Polygon.io API key:
1. Visit [polygon.io](https://polygon.io/)
2. Sign up for an account
3. Navigate to your dashboard
4. Copy your API key

### 2. Stock Trades CSV File

You **must** have a CSV file with your stock trading history. The file path is configured in [`general/constans.py`](general/constans.py):

```python
STOCK_TRADES_CSV_FILE_PATH = "/path/to/your/stock_trades.csv"
```

**Required CSV Format**:
- `ticker name`: Full company name
- `ticker symbol`: Stock ticker symbol (e.g., AAPL, MSFT)
- `action`: Either "BUY" or "SELL"
- `date`: Transaction date in DD/MM/YYYY format
- `stock_amount`: Number of shares traded
- `price_per_stock_at_transection_time`: Price per share at transaction time

**Important**: Update the [`STOCK_TRADES_CSV_FILE_PATH`](general/constans.py:1) in [`general/constans.py`](general/constans.py) to point to your actual CSV file location.


## Logging

All operations are logged with timestamps and detailed information:
- API calls and responses
- File operations
- Error conditions
