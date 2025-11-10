from typing import List, Dict, Any, Optional
from polygon import RESTClient
import logging
from collections import deque
from datetime import datetime, timedelta
import time

#### something something like rate limiting can be implemented here ###


class PolygonIo:
    """Client for interacting with Polygon.io API for stock data."""
    
    def __init__(self, api_key: str, max_calls: int = 5, time_window: int = 60) -> None:
        """
        Initialize the Polygon.io client.
        
        Args:
            api_key: Your Polygon.io API key
            
        Raises:
            ValueError: If api_key is empty or None
            ConnectionError: If unable to connect to Polygon.io API
        """
        if not api_key:
            raise ValueError("API key cannot be empty")
            
        try:
            self.client = RESTClient(api_key)
        except Exception as e:
            raise ConnectionError(f"Failed to initialize Polygon.io client: {e}") from e
        self.max_calls=max_calls
        self.counter_call=0
        self.time_window=time_window
        self.call_timestamps =None 

    def _wait_if_needed(self):
        current_ts = datetime.now().isoformat(timespec='seconds')
        
        if (self.counter_call==0) or (self.call_timestamps is None):
            self.counter_call=1
            self.call_timestamps=current_ts
            return

        # Convert strings to datetime
        fmt = "%Y-%m-%dT%H:%M:%S"
        current_dt = datetime.strptime(current_ts, fmt)
        last_call_dt = datetime.strptime(self.call_timestamps, fmt)
        diff_seconds = (current_dt - last_call_dt).total_seconds()
        if diff_seconds>self.time_window:
            self.call_timestamps=current_ts
            self.counter_call=1
        
        else:
            if self.counter_call<5:
                self.counter_call+=1
            else:
                logging.info(f"reached max API calls,sleeping for {(self.time_window-diff_seconds)+5}")
                time.sleep((self.time_window-diff_seconds)+5)
                self.counter_call=1
                self.call_timestamps=datetime.now().isoformat(timespec='seconds')
        return
                
    def get_stock_daily_data(
        self, 
        symbol: str, 
        from_date: str, 
        to_date: str
    ) -> List[Dict[str, Any]]:
        """
        Get daily stock data for a symbol within a date range.
        
        Args:
            symbol: Stock ticker symbol (e.g., 'AAPL')
            from_date: Start date in YYYY-MM-DD format
            to_date: End date in YYYY-MM-DD format
            
        Returns:
            List of dictionaries containing daily stock data
            
        Raises:
            ValueError: If parameters are invalid
            RuntimeError: If API request fails
        """
        if not symbol:
            raise ValueError("Symbol cannot be empty")

        self._wait_if_needed() 
        logging.info(f"Fetching daily data for {symbol} from {from_date} to {to_date}")
        
        try:
            aggs = self.client.get_aggs(
                ticker=symbol,
                multiplier=1,
                timespan='day',
                from_=from_date,
                to=to_date
            )
            return [day.__dict__ for day in aggs]
        except Exception as e:
            raise RuntimeError(f"Failed to fetch daily data for {symbol}: {e}") from e
    
    def get_stock_last_close_price(self, symbol: str) -> List[float]:
        """
        Get the last closing price for a stock symbol.
        
        Args:
            symbol: Stock ticker symbol (e.g., 'AAPL')
            
        Returns:
            List containing the closing price(s)
            
        Raises:
            ValueError: If symbol is invalid
            RuntimeError: If API request fails
        """
        if not symbol:
            raise ValueError("Symbol cannot be empty")
        self._wait_if_needed()
        try:
            aggs = self.client.get_previous_close_agg(ticker=symbol)
            return [agg.close for agg in aggs]
        except Exception as e:
            raise RuntimeError(f"Failed to fetch last close price for {symbol}: {e}") from e
