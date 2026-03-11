---
name: Summary of portfolio activity
description: Fetch current stock prices, daily changes, and key metrics for a portfolio of stocks In NYSE market.  Use this skill whenever the user asks about stock prices, portfolio per
---

# Stock Price Fetcher

Fetches current stock price data for the user's portfolio

## How to Fetch portfolio 
Use the MCP tool you have:read_user_exchanges_data to get the user last recorded stocks portfolio.

## How to Fetch Prices

Use the MCP tool you have: **get_stock_value** to fetch current stock prices. 


### Step 2: Extract key data per stock

For each stock, extract:
- **Current price** (in local currency: USD for US, ILS/ILA for TLV) if available
- **Daily change** (absolute and %) if available
- **Previous close** if available
- **Day range** (high/low) if available
- **Volume** if available

### Step 3: Present results

Format the output as a clean summary:

**Example output format:**

```
📊 Portfolio Snapshot — [Date, Time]

🇺🇸 US MARKET
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Stock            Price      Change     Volume
GOOG             $300.10    -0.27%     12.3M
TEVA             $18.45     +1.23%     8.1M
...

🇮🇱 TEL AVIV (TLV)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Stock            Price      Change     Volume
POLI.TA          ₪7,578     +0.57%     2.1M
AZRG.TA          ₪32,450    -0.12%     150K
...

🔥 BIG MOVERS (>3% change):
  ⬆️ JNUG +5.2%
  ⬇️ NXSN.TA -3.8%

and so on...
```

- US market hours: Mon-Fri, 09:30-16:00 ET.
- If fetching outside market hours, you'll get the last closing price.
- Always note the timestamp of the data so the user knows how fresh it is.
