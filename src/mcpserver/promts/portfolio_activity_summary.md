---
name: Summary of portfolio activity
description: Fetch current stock prices, weekly performance, and key metrics for a portfolio of stocks. Use this skill whenever the user asks about stock prices, portfolio performance, or activity summary.
---

# Portfolio Activity Summary

Produces a full summary of the user's portfolio: current prices, weekly trend, and key insights.

## Step 1 — Load portfolio

Use the tool `read_user_exchanges_data` to get the user's holdings (list of stock symbols).

## Step 2 — Fetch data per stock

For each stock symbol, make TWO tool calls:

1. **`get_last_closing_stock_price`** — get the latest closing price
2. **`get_stock_value`** — get daily data for the past 7 days (from_date = today minus 7 days, to_date = today)

Use today's actual date for all date calculations.

## Step 3 — Compute per-stock metrics

From the weekly data, calculate:
- **Last close price** (from `get_last_closing_stock_price`)
- **Weekly change %** = (last close - price 7 days ago) / price 7 days ago × 100
- **Weekly high / low**
- **Average daily volume** over the week
- **Trend**: UP if price is higher than 7 days ago, DOWN if lower, FLAT if within ±0.5%

## Step 4 — Present results

Format the output as a markdown table so it renders cleanly:

**Example:**

```
📊 Portfolio Summary — [Today's Date]

| Stock | Last Price | Weekly Change | Weekly High | Weekly Low | Trend |
|-------|-----------|---------------|-------------|------------|-------|
| AAPL  | $211.45   | +3.2%         | $214.00     | $205.10    | ⬆ UP  |
| NVDA  | $134.83   | -0.5%         | $138.20     | $131.00    | ⬇ DOWN|
| MSFT  | $453.13   | +0.0%         | $455.00     | $448.50    | ➡ FLAT|

🔥 Big Movers (weekly change > 3%):
  ⬆ AAPL +3.2%

📉 Biggest Losers:
  ⬇ NVDA -0.5%

💡 Key Metrics:
  • Best performer this week: [symbol] ([change]%)
  • Worst performer this week: [symbol] ([change]%)
```

## Rules
- Always use today's actual date — do not guess or use a hardcoded date.
- Use `get_last_closing_stock_price` for the current price, NOT `get_stock_value` with a single date.
- If weekly data is unavailable for a stock (e.g. newly listed), note it and skip the weekly columns.
- All prices in USD for US stocks.
