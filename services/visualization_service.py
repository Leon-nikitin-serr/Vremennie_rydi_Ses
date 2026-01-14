# -*- coding: utf-8 -*-

"""
Service for data and prediction visualization
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List
from config import config


class VisualizationService:
    """Service for creating charts"""

    @staticmethod
    def plot_prediction(
            ticker: str,
            historical: pd.DataFrame,
            predictions: np.ndarray,
            buy_days: List[int],
            sell_days: List[int]
    ) -> str:
        """
        Create prediction chart

        Args:
            ticker: Company ticker
            historical: Historical data
            predictions: Predicted prices
            buy_days: Days to buy
            sell_days: Days to sell

        Returns:
            Path to saved file
        """
        plt.figure(figsize=config.FIGURE_SIZE)

        # Historical data
        plt.plot(
            historical.index,
            historical['price'],
            label='Historical data',
            linewidth=2,
            color='#2E86AB'
        )

        # Prediction
        future_dates = pd.date_range(
            start=historical.index[-1] + timedelta(days=1),
            periods=len(predictions)
        )
        plt.plot(
            future_dates,
            predictions,
            label='Prediction',
            linewidth=2,
            linestyle='--',
            color='#F77F00'
        )

        # Buy signals
        if buy_days:
            plt.scatter(
                [future_dates[i] for i in buy_days],
                [predictions[i] for i in buy_days],
                color='#06A77D',
                s=150,
                marker='^',
                label='Buy',
                zorder=5,
                edgecolors='black',
                linewidths=1
            )

        # Sell signals
        if sell_days:
            plt.scatter(
                [future_dates[i] for i in sell_days],
                [predictions[i] for i in sell_days],
                color='#D62828',
                s=150,
                marker='v',
                label='Sell',
                zorder=5,
                edgecolors='black',
                linewidths=1
            )

        plt.xlabel('Date', fontsize=12, fontweight='bold')
        plt.ylabel('Price ($)', fontsize=12, fontweight='bold')
        plt.title(
            f'{ticker} Stock Price Forecast for {config.FORECAST_DAYS} days',
            fontsize=14,
            fontweight='bold'
        )
        plt.legend(fontsize=10, loc='best')
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()

        filename = f'{ticker}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        plt.savefig(filename, dpi=config.DPI, bbox_inches='tight')
        plt.close()

        return filename