# -*- coding: utf-8 -*-

"""
Utilities for trading signals detection
"""

import numpy as np
from scipy.signal import argrelextrema
from typing import List, Tuple


class TradingSignals:
    """Class for trading signals detection"""

    @staticmethod
    def find_extrema(predictions: np.ndarray, order: int = 5) -> Tuple[List[int], List[int]]:
        """
        Find local minima and maxima

        Args:
            predictions: Array of predicted prices
            order: Order for extremum detection

        Returns:
            Tuple (buy days, sell days)
        """
        local_min = argrelextrema(predictions, np.less, order=order)[0]
        local_max = argrelextrema(predictions, np.greater, order=order)[0]

        return local_min.tolist(), local_max.tolist()

    @staticmethod
    def calculate_profit(
            predictions: np.ndarray,
            investment: float,
            buy_days: List[int],
            sell_days: List[int]
    ) -> Tuple[float, str]:
        """
        Calculate potential profit

        Args:
            predictions: Array of predicted prices
            investment: Investment amount
            buy_days: Days to buy
            sell_days: Days to sell

        Returns:
            Tuple (total profit, strategy description)
        """
        if not buy_days or not sell_days:
            return 0.0, "Not enough signals for strategy calculation"

        strategy = []
        total_profit = 0

        for buy_day in buy_days:
            # Find next sell day after buy
            sell_candidates = [d for d in sell_days if d > buy_day]

            if sell_candidates:
                sell_day = sell_candidates[0]
                buy_price = predictions[buy_day]
                sell_price = predictions[sell_day]

                # Calculate profit from one trade
                shares = investment / buy_price
                profit = shares * (sell_price - buy_price)
                total_profit += profit

                strategy.append(
                    f"📅 Day {buy_day+1}: Buy at ${buy_price:.2f}\n"
                    f"📅 Day {sell_day+1}: Sell at ${sell_price:.2f}\n"
                    f"💵 Trade profit: ${profit:.2f}"
                )

        if not strategy:
            return 0.0, "No profitable points for buying and selling"

        strategy_text = "\n\n".join(strategy)
        return total_profit, strategy_text