# -*- coding: utf-8 -*-

"""
Base class for all models
"""

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd


class BaseModel(ABC):
    """Base class for forecasting models"""

    def __init__(self, name: str):
        self.name = name
        self.model = None
        self.trained = False

    @abstractmethod
    def train(self, data: pd.DataFrame, train_size: float) -> float:
        """
        Train the model

        Args:
            data: DataFrame with historical data
            train_size: Training set size (0-1)

        Returns:
            RMSE on test set
        """
        pass

    @abstractmethod
    def predict(self, steps: int) -> np.ndarray:
        """
        Predict future values

        Args:
            steps: Number of steps to predict

        Returns:
            Array of predicted values
        """
        pass

    def get_name(self) -> str:
        """Get model name"""
        return self.name

    def is_trained(self) -> bool:
        """Check if model is trained"""
        return self.trained