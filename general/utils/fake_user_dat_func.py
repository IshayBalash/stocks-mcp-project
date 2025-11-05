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

from clients.polygon_io import PolyginIo


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
    client=PolyginIo(os.getenv('POLYGON_API_KEY'))
    for i in range(n_trades):
        logging.info(f"loop index: {i}")
        ticker = np.random.choice(stock_tickers)
        action = np.random.choice(["BUY", "SELL"])
        logging.info(f"action: {action}")
        date = np.random.choice(dates)
        date_pd = pd.Timestamp(date)  # Convert to pandas Timestamp
        from_date = date_pd.strftime('%Y-%m-%d')
        to_date = (date_pd + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        actal_price =client.get_stock_daily_data(
                symbol=ticker,
                from_date=from_date,
                to_date=to_date
        )
        time.sleep(20)
        closing_price = [item['close'] for item in actal_price]
        amount = np.random.randint(10, 500)               # shares traded
        trades.append((ticker, date, action, amount, closing_price))
    
    # Create DataFrame
    df = pd.DataFrame(trades, columns=["ticker", "date", "action", "stock_amount", "closing_day_stock_price"])
    # Calculate net positions per ticker
    net_positions = []
    
    for ticker in df['ticker'].unique():
        ticker_data = df[df['ticker'] == ticker]
        
        buy_total = ticker_data[ticker_data['action'] == 'BUY']['stock_amount'].sum()
        sell_total = ticker_data[ticker_data['action'] == 'SELL']['stock_amount'].sum()
        net_position = buy_total - sell_total
        
        # Only add positive net positions (remove negative cases)
        if net_position > 0:
            # Find first SELL date for this ticker
            sell_dates = ticker_data[ticker_data['action'] == 'SELL']['date']
            
            if len(sell_dates) > 0:
                # Use first SELL date minus 1 week
                reference_date = sell_dates.min() - pd.Timedelta(weeks=1)
            else:
                # Use earliest record minus 1 week
                reference_date = ticker_data['date'].min() - pd.Timedelta(weeks=1)
            
            net_positions.append({
                'ticker': ticker,
                'date': reference_date,
                'action': 'BUY',  # Changed from "first buy" to "BUY"
                'stock_amount': net_position,
                'closing_day_stock_price': None
            })

    # Add net position rows to DataFrame (only positive ones)
    if net_positions:
        net_df = pd.DataFrame(net_positions)
        df = pd.concat([df, net_df], ignore_index=True)






    
    df = df.sort_values(by="date").reset_index(drop=True)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"✅ Synthetic trade log saved to: {output_path}")
    print(df.head(10))  # show preview
    return df


