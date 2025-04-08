import logging
from typing import Dict, List, Any, Optional

# Import model implementations
from .base_model import BaseModel
from .gradient_boosting import GradientBoostingModel


class ModelRegistry:
    """
    Registry for cognitive load prediction models that provides
    a unified interface for model creation and management.
    """
    
    def __init__(self):
        """Initialize the ModelRegistry."""
        self.registry = {}
        self.logger = logging.getLogger(__name__)
        
        # Register all available models
        self._register_models()
    
    def _register_models(self) -> None:
        """Register all available model types."""
        # Register Gradient Boosting Regressor
        self.registry['gb'] = GradientBoostingModel
        
        # Add more model types here as they are implemented
        # self.registry['rf'] = RandomForestModel
        # self.registry['xgb'] = XGBoostModel
        # self.registry['lgb'] = LightGBMModel
        
        self.logger.info(f"Registered {len(self.registry)} model types: {list(self.registry.keys())}")
    
    def get_model(self, model_type: str, id_column: str = 'pilot_id',
                target_column: str = 'avg_tlx_quantile',
                seed: int = 42,
                output_dir: str = None,
                params: Dict[str, Any] = None) -> Optional[BaseModel]:
        """
        Get a model instance of the specified type.
        
        Args:
            model_type: Model type identifier (e.g., 'gb', 'rf', 'xgb')
            id_column: Column name for subject identifier
            target_column: Column name for the target variable
            seed: Random seed for reproducibility
            output_dir: Directory to save output files
            params: Model-specific parameters
            
        Returns:
            Model instance or None if model_type is not found
        """
        if model_type not in self.registry:
            self.logger.error(f"Unknown model type: {model_type}")
            self.logger.info(f"Available model types: {list(self.registry.keys())}")
            return None
        
        try:
            model_class = self.registry[model_type]
            return model_class(
                id_column=id_column,
                target_column=target_column,
                seed=seed,
                output_dir=output_dir,
                params=params
            )
        except Exception as e:
            self.logger.error(f"Error creating model instance of type {model_type}: {str(e)}")
            return None
    
    def get_available_models(self) -> List[str]:
        """
        Get list of available model types.
        
        Returns:
            List of model type identifiers
        """
        return list(self.registry.keys())
