# -*- coding: utf-8 -*-

"""
Service for stock price prediction
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from models.random_forest import RandomForestModel
from models.arima_model import ARIMAModel
from models.lstm_model import LSTMModel
from models.base_model import BaseModel
from config import config
import logging

logger = logging.getLogger(__name__)


class PredictionService:
    """Service for model training and prediction"""

    def __init__(self):
        self.models = {
            'Random Forest': RandomForestModel(),
            'ARIMA': ARIMAModel(),
            'LSTM': LSTMModel()
        }
        self.best_model_name: Optional[str] = None
        self.best_rmse: float = float('inf')
        self.results: Dict[str, float] = {}

    def train_all_models(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Train all models

        Args:
            data: DataFrame with historical data

        Returns:
            Dictionary {model_name: RMSE}
        """
        results = {}

        for name, model in self.models.items():
            logger.info(f"Training model {name}...")
            try:
                rmse = model.train(data, train_size=config.TRAIN_SIZE)
                results[name] = rmse
                logger.info(f"{name}: RMSE = {rmse:.2f}")
            except Exception as e:
                logger.error(f"Error training {name}: {e}")
                results[name] = float('inf')

        # Select best model
        self.results = results
        self.best_model_name = min(results.keys(), key=lambda k: results[k])
        self.best_rmse = results[self.best_model_name]

        logger.info(f"Best model: {self.best_model_name} (RMSE={self.best_rmse:.2f})")

        return results

    def get_best_model(self) -> BaseModel:
        """Get the best model"""
        if self.best_model_name is None:
            raise ValueError("Models are not trained")
        return self.models[self.best_model_name]

    def predict(self, steps: int = 30) -> np.ndarray:
        """
        Predict future prices

        Args:
            steps: Number of days for prediction

        Returns:
            Array of predicted prices
        """
        best_model = self.get_best_model()
        predictions = best_model.predict(steps)
        return predictions

    def get_results_summary(self) -> Dict[str, any]:
        """Get results summary"""
        return {
            'best_model': self.best_model_name,
            'best_rmse': self.best_rmse,
            'all_results': self.results
        }