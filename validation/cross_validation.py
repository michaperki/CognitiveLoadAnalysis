import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
from tqdm import tqdm
import os


class CrossValidator:
    """
    Performs cross-validation for cognitive load prediction models 
    with support for subject independence.
    """
    
    def __init__(self, id_column: str = 'pilot_id',
                 target_column: str = 'avg_tlx_quantile',
                 seed: int = 42,
                 output_dir: str = None):
        """
        Initialize the CrossValidator.
        
        Args:
            id_column: Column name for subject identifier
            target_column: Column name for the target variable
            seed: Random seed for reproducibility
            output_dir: Directory to save validation results
        """
        self.id_column = id_column
        self.target_column = target_column
        self.seed = seed
        self.output_dir = output_dir
        
        # Validation results
        self.results = {}
        
        # Logger
        self.logger = logging.getLogger(__name__)
        
        # Set random seed
        np.random.seed(self.seed)
    
    def perform_group_cv(self, data: pd.DataFrame, features: List[str], 
                       model_fn: Callable,
                       n_splits: int = 5,
                       stratify: bool = False,
                       stratify_col: str = None) -> Dict[str, Any]:
        """
        Perform group-based cross-validation that respects subject independence.
        
        Args:
            data: Input DataFrame
            features: List of feature columns
            model_fn: Function that returns a trained model given X_train, y_train
            n_splits: Number of CV splits
            stratify: Whether to use stratified groups
            stratify_col: Column to use for stratification
            
        Returns:
            Dictionary with cross-validation results
        """
        self.logger.info(f"Performing {n_splits}-fold cross-validation with group independence...")
        
        # Check if ID column exists
        if self.id_column not in data.columns:
            self.logger.error(f"ID column '{self.id_column}' not found in data")
            raise ValueError(f"ID column '{self.id_column}' not found in data")
        
        # Check if target column exists
        if self.target_column not in data.columns:
            self.logger.error(f"Target column '{self.target_column}' not found in data")
            raise ValueError(f"Target column '{self.target_column}' not found in data")
        
        # Prepare data
        X = data[features]
        y = data[self.target_column]
        groups = data[self.id_column]
        
        # Choose appropriate CV splitter
        if stratify and stratify_col is not None and stratify_col in data.columns:
            self.logger.info(f"Using StratifiedGroupKFold with '{stratify_col}' for stratification")
            cv_splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=self.seed)
            stratify_values = data[stratify_col]
        else:
            self.logger.info("Using GroupKFold for subject independence")
            cv_splitter = GroupKFold(n_splits=n_splits)
            stratify_values = None
        
        # Initialize arrays to store results
        fold_metrics = {
            'train_r2': [],
            'test_r2': [],
            'train_rmse': [],
            'test_rmse': [],
            'train_mae': [],
            'test_mae': [],
            'test_predictions': [],
            'test_actuals': []
        }
        
        # Perform cross-validation
        for i, (train_idx, test_idx) in enumerate(tqdm(cv_splitter.split(X, stratify_values, groups))):
            self.logger.info(f"Fold {i+1}/{n_splits}")
            
            # Split data
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Train model
            model = model_fn(X_train, y_train)
            
            # Make predictions
            try:
                train_pred = model.predict(X_train)
                test_pred = model.predict(X_test)
                
                # Calculate metrics
                train_r2 = r2_score(y_train, train_pred)
                test_r2 = r2_score(y_test, test_pred)
                
                train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
                test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
                
                train_mae = mean_absolute_error(y_train, train_pred)
                test_mae = mean_absolute_error(y_test, test_pred)
                
                # Store metrics
                fold_metrics['train_r2'].append(train_r2)
                fold_metrics['test_r2'].append(test_r2)
                fold_metrics['train_rmse'].append(train_rmse)
                fold_metrics['test_rmse'].append(test_rmse)
                fold_metrics['train_mae'].append(train_mae)
                fold_metrics['test_mae'].append(test_mae)
                
                # Store predictions and actuals
                fold_metrics['test_predictions'].append(test_pred)
                fold_metrics['test_actuals'].append(y_test.values)
                
                self.logger.info(f"  Train R²: {train_r2:.4f}, RMSE: {train_rmse:.4f}")
                self.logger.info(f"  Test R²: {test_r2:.4f}, RMSE: {test_rmse:.4f}")
                
            except Exception as e:
                self.logger.error(f"Error in fold {i+1}: {str(e)}")
        
        # Calculate mean and std of metrics
        mean_train_r2 = np.mean(fold_metrics['train_r2'])
        std_train_r2 = np.std(fold_metrics['train_r2'])
        
        mean_test_r2 = np.mean(fold_metrics['test_r2'])
        std_test_r2 = np.std(fold_metrics['test_r2'])
        
        mean_train_rmse = np.mean(fold_metrics['train_rmse'])
        std_train_rmse = np.std(fold_metrics['train_rmse'])
        
        mean_test_rmse = np.mean(fold_metrics['test_rmse'])
        std_test_rmse = np.std(fold_metrics['test_rmse'])
        
        # Log overall results
        self.logger.info("\nCross-validation results:")
        self.logger.info(f"  Mean train R²: {mean_train_r2:.4f} ± {std_train_r2:.4f}")
        self.logger.info(f"  Mean test R²: {mean_test_r2:.4f} ± {std_test_r2:.4f}")
        self.logger.info(f"  Mean train RMSE: {mean_train_rmse:.4f} ± {std_train_rmse:.4f}")
        self.logger.info(f"  Mean test RMSE: {mean_test_rmse:.4f} ± {std_test_rmse:.4f}")
        
        # Calculate overfitting gap
        r2_gap = mean_train_r2 - mean_test_r2
        rmse_gap = mean_test_rmse - mean_train_rmse
        
        self.logger.info(f"  R² gap (train - test): {r2_gap:.4f}")
        self.logger.info(f"  RMSE gap (test - train): {rmse_gap:.4f}")
        
        if r2_gap > 0.2:
            self.logger.warning("  Potential overfitting detected based on R² gap")
        
        # Create summary results
        cv_results = {
            'fold_metrics': fold_metrics,
            'mean_train_r2': mean_train_r2,
            'std_train_r2': std_train_r2,
            'mean_test_r2': mean_test_r2,
            'std_test_r2': std_test_r2,
            'mean_train_rmse': mean_train_rmse,
            'std_train_rmse': std_train_rmse,
            'mean_test_rmse': mean_test_rmse,
            'std_test_rmse': std_test_rmse,
            'r2_gap': r2_gap,
            'rmse_gap': rmse_gap,
            'n_splits': n_splits,
            'features': features
        }
        
        # Create visualizations
        if self.output_dir:
            self._visualize_cv_results(cv_results)
        
        # Store results
        self.results['group_cv'] = cv_results
        
        return cv_results
    
    def _visualize_cv_results(self, cv_results: Dict[str, Any]) -> None:
        """
        Visualize cross-validation results.
        
        Args:
            cv_results: Dictionary with CV results
        """
        if not self.output_dir:
            return
            
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create performance plot
        plt.figure(figsize=(12, 6))
        
        # Plot R² for each fold
        plt.subplot(1, 2, 1)
