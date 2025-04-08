import os
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

from .base_model import BaseModel


class GradientBoostingModel(BaseModel):
    """
    Implementation of Gradient Boosting Regressor for cognitive load prediction.
    """
    
    def __init__(self, id_column: str = 'pilot_id',
                 target_column: str = 'avg_tlx_quantile',
                 seed: int = 42,
                 output_dir: str = None,
                 params: Dict[str, Any] = None):
        """
        Initialize the GradientBoostingModel.
        
        Args:
            id_column: Column name for subject identifier
            target_column: Column name for the target variable
            seed: Random seed for reproducibility
            output_dir: Directory to save output files
            params: Parameters for the gradient boosting model
        """
        super().__init__(id_column, target_column, seed, output_dir)
        
        # Default parameters
        self.default_params = {
            'n_estimators': 100,
            'learning_rate': 0.05,
            'max_depth': 5,
            'min_samples_split': 20,
            'min_samples_leaf': 10,
            'subsample': 0.8,
            'random_state': seed
        }
        
        # Set model parameters
        if params:
            self.params = {**self.default_params, **params}
        else:
            self.params = self.default_params
            
        # Initialize the model
        self.scaler = StandardScaler()
        self.estimator = GradientBoostingRegressor(**self.params)
        self.model = Pipeline([('scaler', self.scaler), ('gb', self.estimator)])
        
        # Initialize feature names
        self.feature_names = None
    
    def train(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> None:
        """
        Train the gradient boosting model.
        
        Args:
            X: Feature data
            y: Target data
            **kwargs: Additional arguments for model training
        """
        self.logger.info("Training Gradient Boosting model...")
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Update parameters if provided
        if 'params' in kwargs:
            new_params = kwargs['params']
            self.params = {**self.params, **new_params}
            self.estimator.set_params(**self.params)
        
        # Get sample weights if provided
        sample_weight = kwargs.get('sample_weight', None)
        
        # Train the model
        try:
            self.model.fit(X, y, gb__sample_weight=sample_weight)
            
            # Log training information
            self.logger.info(f"Model trained with {len(X)} samples and {X.shape[1]} features")
            self.logger.info(f"Parameters: {self.params}")
            
            # Calculate training metrics
            y_pred = self.model.predict(X)
            train_mse = mean_squared_error(y, y_pred)
            train_rmse = np.sqrt(train_mse)
            train_r2 = r2_score(y, y_pred)
            
            self.metrics['train_mse'] = train_mse
            self.metrics['train_rmse'] = train_rmse
            self.metrics['train_r2'] = train_r2
            
            self.logger.info(f"Training metrics - RMSE: {train_rmse:.4f}, R²: {train_r2:.4f}")
            
        except Exception as e:
            self.logger.error(f"Error training model: {str(e)}")
            raise
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions with the trained model.
        
        Args:
            X: Feature data
            
        Returns:
            Array of predictions
        """
        if not self.check_is_fitted():
            return np.zeros(len(X))
        
        try:
            return self.model.predict(X)
        except Exception as e:
            self.logger.error(f"Error making predictions: {str(e)}")
            return np.zeros(len(X))
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Evaluate the model on the provided data.
        
        Args:
            X: Feature data
            y: Target data
            
        Returns:
            Dictionary of evaluation metrics
        """
        if not self.check_is_fitted():
            return {}
        
        try:
            # Make predictions
            y_pred = self.predict(X)
            
            # Calculate metrics
            mse = mean_squared_error(y, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y, y_pred)
            r2 = r2_score(y, y_pred)
            
            # Calculate residuals
            residuals = y - y_pred
            residuals_mean = np.mean(residuals)
            residuals_std = np.std(residuals)
            
            # Store metrics
            metrics = {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'residuals_mean': residuals_mean,
                'residuals_std': residuals_std,
                'explained_variance': self.estimator.explained_variance_
            }
            
            # Log evaluation metrics
            self.logger.info(f"Evaluation metrics - RMSE: {rmse:.4f}, R²: {r2:.4f}, MAE: {mae:.4f}")
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error evaluating model: {str(e)}")
            return {}
    
    def get_feature_importance(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        """
        Get feature importance for the model.
        
        Args:
            X: Feature data
            y: Target data (optional, used for permutation importance)
            
        Returns:
            DataFrame with feature importances
        """
        if not self.check_is_fitted():
            return pd.DataFrame()
        
        try:
            # Get feature importances from the model
            importances = self.estimator.feature_importances_
            
            # Create DataFrame
            if self.feature_names is None:
                self.feature_names = [f"feature_{i}" for i in range(len(importances))]
                
            # Create importance DataFrame
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            # Calculate permutation importance if y is provided
            if y is not None:
                perm_importance = permutation_importance(self.model, X, y, n_repeats=10, random_state=self.seed)
                perm_imp = perm_importance.importances_mean
                
                # Add permutation importance to DataFrame
                perm_df = pd.DataFrame({
                    'feature': self.feature_names,
                    'permutation_importance': perm_imp
                }).sort_values('permutation_importance', ascending=False)
                
                # Merge with original importance DataFrame
                importance_df = importance_df.merge(perm_df, on='feature')
            
            # Visualize feature importance if output directory is provided
            if self.output_dir:
                self._visualize_feature_importance(importance_df)
            
            return importance_df
            
        except Exception as e:
            self.logger.error(f"Error calculating feature importance: {str(e)}")
            return pd.DataFrame()
    
    def _visualize_feature_importance(self, importance_df: pd.DataFrame) -> None:
        """
        Visualize feature importance.
        
        Args:
            importance_df: DataFrame with feature importance
        """
        if not self.output_dir:
            return
            
        # Create visualization directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Get top features
        top_n = min(20, len(importance_df))
        plt.figure(figsize=(12, 8))
        
        # Create horizontal bar chart
        features = importance_df['feature'].values[:top_n]
        importances = importance_df['importance'].values[:top_n]
        
        # Sort features for better visualization (ascending order for horizontal bar chart)
        idx = np.argsort(importances)
        features = features[idx]
        importances = importances[idx]
        
        # Create color gradient
        colors = plt.cm.viridis(np.linspace(0, 1, len(features)))
        
        # Create horizontal bars
        bars = plt.barh(range(len(features)), importances, align='center', color=colors)
        
        # Add feature names as y-tick labels
        plt.yticks(range(len(features)), features)
        
        # Add labels and title
        plt.xlabel('Importance', fontsize=12)
        plt.title('Gradient Boosting Feature Importance', fontsize=14)
        
        # Add importance values at the end of each bar
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width, bar.get_y() + bar.get_height()/2, 
                   f' {width:.4f}', va='center')
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'gb_feature_importance.png'), dpi=300)
        plt.close()
        
        # Create permutation importance plot if available
        if 'permutation_importance' in importance_df.columns:
            plt.figure(figsize=(12, 8))
            
            # Get top features by permutation importance
            importance_df = importance_df.sort_values('permutation_importance', ascending=False)
            features = importance_df['feature'].values[:top_n]
            importances = importance_df['permutation_importance'].values[:top_n]
            
            # Sort features for better visualization (ascending order for horizontal bar chart)
            idx = np.argsort(importances)
            features = features[idx]
            importances = importances[idx]
            
            # Create horizontal bars
            bars = plt.barh(range(len(features)), importances, align='center', color=colors)
            
            # Add feature names as y-tick labels
            plt.yticks(range(len(features)), features)
            
            # Add labels and title
            plt.xlabel('Permutation Importance', fontsize=12)
            plt.title('Gradient Boosting Permutation Feature Importance', fontsize=14)
            
            # Save figure
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'gb_permutation_importance.png'), dpi=300)
            plt.close()
