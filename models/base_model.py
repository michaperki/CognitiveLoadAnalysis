import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
from sklearn.base import BaseEstimator
from abc import ABC, abstractmethod


class BaseModel(ABC):
    """
    Abstract base class for cognitive load prediction models
    that defines a common interface.
    """
    
    def __init__(self, id_column: str = 'pilot_id',
                 target_column: str = 'avg_tlx_quantile',
                 seed: int = 42,
                 output_dir: str = None):
        """
        Initialize the BaseModel object.
        
        Args:
            id_column: Column name for subject identifier
            target_column: Column name for the target variable
            seed: Random seed for reproducibility
            output_dir: Directory to save output files
        """
        self.id_column = id_column
        self.target_column = target_column
        self.seed = seed
        self.output_dir = output_dir
        
        # Model attribute
        self.model = None
        
        # Performance metrics
        self.metrics = {}
        
        # Logger
        self.logger = logging.getLogger(__name__)
        
    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> None:
        """
        Train the model on the provided data.
        
        Args:
            X: Feature data
            y: Target data
            **kwargs: Additional arguments for model training
        """
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions with the trained model.
        
        Args:
            X: Feature data
            
        Returns:
            Array of predictions
        """
        pass
    
    @abstractmethod
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Evaluate the model on the provided data.
        
        Args:
            X: Feature data
            y: Target data
            
        Returns:
            Dictionary of evaluation metrics
        """
        pass
    
    @abstractmethod
    def get_feature_importance(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Get feature importance for the model.
        
        Args:
            X: Feature data
            y: Target data
            
        Returns:
            DataFrame with feature importances
        """
        pass
    
    def save_model(self, path: str) -> None:
        """
        Save the model to disk.
        
        Args:
            path: Path to save the model
        """
        if self.model is None:
            self.logger.error("No model to save. Train a model first.")
            return
        
        try:
            import joblib
            joblib.dump(self.model, path)
            self.logger.info(f"Model saved to {path}")
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
    
    def load_model(self, path: str) -> None:
        """
        Load a model from disk.
        
        Args:
            path: Path to the saved model
        """
        try:
            import joblib
            self.model = joblib.load(path)
            self.logger.info(f"Model loaded from {path}")
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            
    def check_is_fitted(self) -> bool:
        """
        Check if the model is fitted.
        
        Returns:
            Boolean indicating if the model is fitted
        """
        if self.model is None:
            self.logger.warning("Model is not trained yet")
            return False
        return True
