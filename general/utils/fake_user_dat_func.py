import numpy as np
import pandas as pd
import os 
from dotenv import load_dotenv
import time
load_dotenv()


import logging

stock_tickers = [
'MSFT',

'AAPL',

'NVDA',

'JPM',

'MA',

'PG',

'JNJ',

'XOM',

'LEU',

'U',

]

from clients.polygon_io import PolygonIo


def generate_trades(
    n_trades=10,
    start_date="2025-01-01",
    seed=123,
    output_path="fake_stock_trades.csv"
):
    """
    Generate a synthetic dataset of stock buy/sell trades for ML training.
    
    Parameters:
        n_tickers (int): Number of unique fake stock tickers to simulate.
        n_trades (int): Number of trade rows to create.
        start_date (str): Start date for random trade dates.
        seed (int): Random seed for reproducibility.
        output_path (str): Where to save the generated CSV.
    """
    np.random.seed(seed)
    

    # Generate random business days
    dates = pd.bdate_range(start=pd.to_datetime(start_date), periods=n_trades * 10)
    
    # Build random trades
    trades = []
    logging.info(os.getenv('POLYGON_API_KEY'))
    client=PolygonIo(os.getenv('POLYGON_API_KEY'))
    for i in range(n_trades):
        logging.info(f"loop index: {i}")
        ticker = np.random.choice(stock_tickers)
        action = np.random.choice(["BUY", "SELL"])
        logging.info(f"action: {action}")
        date = np.random.choice(dates)
        date_pd = pd.Timestamp(date)  # Convert to pandas Timestamp
        from_date = date_pd.strftime('%Y-%m-%d')

        actal_price = client.get_stock_daily_data(
            symbol=ticker,
            from_date=from_date,
            to_date=from_date
        )

        # Only add trade if API returned valid price data
        if actal_price and len(actal_price) > 0:
            closing_price = actal_price[0]['close']  # Extract single float value
            amount = np.random.randint(10, 500)  # shares traded
            trades.append((ticker, date, action, amount, closing_price))
    
    # Create DataFrame
    df = pd.DataFrame(trades, columns=["ticker", "date", "action", "stock_amount", "closing_day_stock_price"])
    
    # Calculate required initial positions per ticker based on chronological order
    net_positions = []
    for ticker in df['ticker'].unique():
        ticker_data = df[df['ticker'] == ticker].sort_values('date').copy()

        # Calculate running balance (cumulative buys - cumulative sells)
        ticker_data['signed_amount'] = ticker_data.apply(
            lambda row: row['stock_amount'] if row['action'] == 'BUY' else -row['stock_amount'],
            axis=1
        )
        ticker_data['running_balance'] = ticker_data['signed_amount'].cumsum()

        # Find minimum running balance
        min_balance = ticker_data['running_balance'].min()

        # If minimum balance is negative, we need a synthetic BUY at the start
        if min_balance < 0:
            # Use earliest trade date minus 1 week
            reference_date = ticker_data['date'].min() - pd.Timedelta(weeks=1)

            # Fetch actual stock price for the synthetic BUY record
            reference_date_str = reference_date.strftime('%Y-%m-%d')
            price_data = client.get_stock_daily_data(
                symbol=ticker,
                from_date=reference_date_str,
                to_date=reference_date_str
            )

            # Extract closing price if available
            closing_price = None
            if price_data and len(price_data) > 0:
                closing_price = price_data[0]['close']

            net_positions.append({
                'ticker': ticker,
                'date': reference_date,
                'action': 'BUY',
                'stock_amount': abs(min_balance),
                'closing_day_stock_price': closing_price
            })

    # Add synthetic BUY records to DataFrame (to ensure no SELL without prior BUY)
    if net_positions:
        net_df = pd.DataFrame(net_positions)
        df = pd.concat([df, net_df], ignore_index=True)
    df = df.sort_values(by="date").reset_index(drop=True)
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"✅ Synthetic trade log saved to: {output_path}")
    print(df.head(10))  # show preview
    return df


