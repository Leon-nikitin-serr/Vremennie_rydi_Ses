
"""
Data service for loading and processing stock data
"""

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from config import config
import logging

logger = logging.getLogger(__name__)


class DataService:
    """Service for working with stock data"""

    @staticmethod
    def load_stock_data(ticker: str) -> pd.DataFrame:
        """
        Load historical stock data

        Args:
            ticker: Stock ticker (e.g., AAPL)

        Returns:
            DataFrame with closing prices, or None if error
        """
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=config.HISTORY_DAYS)

            data = yf.download(ticker, start=start_date, end=end_date, progress=False)

            if data.empty:
                logger.error(f"No data found for ticker {ticker}")
                return None

            # Extract only closing prices
            df = data[['Close']].copy()
            df.columns = ['price']

            logger.info(f"Loaded {len(df)} records for {ticker}")
            return df

        except Exception as e:
            logger.error(f"Error loading data for {ticker}: {e}")
            return None

    @staticmethod
    def validate_ticker(ticker: str) -> bool:
        """Validate stock ticker"""
        if not ticker or len(ticker) > 10:
            return False
        return ticker.replace('.', '').replace('-', '').isalnum()
