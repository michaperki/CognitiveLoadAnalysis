
import os
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import ElasticNet, Ridge, Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.base import clone, BaseEstimator
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
import xgboost as xgb
import lightgbm as lgb
import joblib
import optuna
from tqdm import tqdm


class ModelTrainer:
    """
    Handles model training and evaluation for cognitive load prediction
    with improved validation and overfitting detection capabilities.
    """
    
    def __init__(self, id_column: str = 'pilot_id',
                 target_column: str = 'avg_tlx_quantile',
                 seed: int = 42,
                 output_dir: str = None,
                 model_config: Dict[str, Any] = None):
        """
        Initialize the ModelTrainer object.
        
        Args:
            id_column: Column name for subject identifier
            target_column: Column name for the target variable
            seed: Random seed for reproducibility
            output_dir: Directory to save output files
            model_config: Configuration for models
        """
        self.id_column = id_column
        self.target_column = target_column
        self.seed = seed
        self.output_dir = output_dir
        self.model_config = model_config or {}
        
        # Model containers
        self.models = {}
        self.results = {}
        self.best_model = None
        
        # Logger
        self.logger = logging.getLogger(__name__)
        
        # Set random seed
        np.random.seed(self.seed)
    
    def get_model_instance(self, model_type: str, params: Dict[str, Any] = None) -> BaseEstimator:
        """
        Get a model instance of the specified type.
        
        Args:
            model_type: Model type ('rf', 'gb', 'xgb', 'lgb', 'nn', 'elastic', 'svr')
            params: Model parameters
            
        Returns:
            Model instance
        """
        default_params = {'random_state': self.seed}
        
        if params:
            all_params = {**default_params, **params}
        else:
            all_params = default_params
        
        if model_type == 'rf':
            return RandomForestRegressor(**all_params)
        elif model_type == 'gb':
            return GradientBoostingRegressor(**all_params)
        elif model_type == 'xgb':
            return xgb.XGBRegressor(**all_params)
        elif model_type == 'lgb':
            # Add 'verbose': -1 to suppress LightGBM output
            lgb_params = {**all_params, 'verbose': -1}
            return lgb.LGBMRegressor(**lgb_params)
        elif model_type == 'nn':
            # Add default parameters for MLP
            mlp_params = {
                'hidden_layer_sizes': (100, 50),
                'max_iter': 1000,
                **all_params
            }
            return MLPRegressor(**mlp_params)
        elif model_type == 'elastic':
            return ElasticNet(**all_params)
        elif model_type == 'svr':
            # Remove random_state for SVR as it doesn't support it
            svr_params = all_params.copy()
            svr_params.pop('random_state', None)
            return SVR(**svr_params)
        elif model_type == 'ridge':
            return Ridge(**all_params)
        elif model_type == 'lasso':
            return Lasso(**all_params)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def optimize_hyperparameters(self, data: pd.DataFrame, feature_set: List[str], 
                               model_type: str = 'gb', 
                               n_trials: int = 20) -> Dict[str, Any]:
        """
        Optimize hyperparameters for a specific model and feature set.
        
        Args:
            data: DataFrame with features and target
            feature_set: List of features to use
            model_type: Model type ('rf', 'gb', 'xgb', 'lgb', 'nn', 'elastic', 'svr')
            n_trials: Number of optimization trials
            
        Returns:
            Dictionary with optimized hyperparameters
        """
        self.logger.info(f"\nOptimizing hyperparameters for {model_type} model with {n_trials} trials...")
        
        # Prepare data
        X = data[feature_set]
        y = data[self.target_column]
        
        # Create train/test split for hyperparameter tuning using GroupKFold for subject independence
        group_cv = GroupKFold(n_splits=5)
        groups = data[self.id_column].values
        
        # Define objective function based on cross-validation
        def objective(trial):
            if model_type == 'rf':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                    'max_depth': trial.suggest_int('max_depth', 3, 25),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 30),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
                    'max_features': trial.suggest_float('max_features', 0.1, 1.0),
                    'random_state': self.seed
                }
                model = RandomForestRegressor(**params)
            elif model_type == 'gb':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'max_depth': trial.suggest_int('max_depth', 2, 10),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 30),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'random_state': self.seed
                }
                model = GradientBoostingRegressor(**params)
            elif model_type == 'xgb':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'gamma': trial.suggest_float('gamma', 0, 5),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                    'random_state': self.seed
                }
                model = xgb.XGBRegressor(**params)
            elif model_type == 'lgb':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'num_leaves': trial.suggest_int('num_leaves', 20, 150),
                    'max_depth': trial.suggest_int('max_depth', 3, 15),
                    'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 5, 50),
                    'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
                    'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
                    'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                    'random_state': self.seed,
                    'verbose': -1
                }
                model = lgb.LGBMRegressor(**params)
            elif model_type == 'nn':
                params = {
                    'hidden_layer_sizes': (
                        trial.suggest_int('hidden_layer_1', 20, 200),
                        trial.suggest_int('hidden_layer_2', 10, 100),
                    ),
                    'activation': trial.suggest_categorical('activation', ['relu', 'tanh']),
                    'solver': trial.suggest_categorical('solver', ['adam', 'sgd']),
                    'alpha': trial.suggest_float('alpha', 0.0001, 0.01, log=True),
                    'learning_rate_init': trial.suggest_float('learning_rate_init', 0.001, 0.01, log=True),
                    'max_iter': 1000,
                    'random_state': self.seed
                }
                model = MLPRegressor(**params)
            elif model_type == 'elastic':
                params = {
                    'alpha': trial.suggest_float('alpha', 0.0001, 1.0, log=True),
                    'l1_ratio': trial.suggest_float('l1_ratio', 0.1, 0.9),
                    'max_iter': 2000,
                    'random_state': self.seed
                }
                model = ElasticNet(**params)
            elif model_type == 'svr':
                params = {
                    'C': trial.suggest_float('C', 0.1, 100.0, log=True),
                    'epsilon': trial.suggest_float('epsilon', 0.01, 1.0),
                    'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']),
                    'kernel': trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly'])
                }
                model = SVR(**params)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            # Cross-validation evaluation
            cv_scores = []
            for train_idx, val_idx in group_cv.split(X, y, groups):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_val_scaled = scaler.transform(X_val)
                
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_val_scaled)
                score = r2_score(y_val, y_pred)
                cv_scores.append(score)
            
            return np.mean(cv_scores)
        
        try:
            study = optuna.create_study(direction='maximize', 
                                        sampler=optuna.samplers.TPESampler(seed=self.seed))
            study.optimize(objective, n_trials=n_trials, timeout=3600)  # 1 hour timeout
            
            best_params = study.best_params
            best_value = study.best_value
            
            self.logger.info(f"Best CV R² score: {best_value:.4f}")
            self.logger.info(f"Best hyperparameters: {best_params}")
            
            if self.output_dir:
                params_path = os.path.join(self.output_dir, f'{model_type}_best_params.txt')
                with open(params_path, 'w') as f:
                    f.write(f"Best CV R² score: {best_value:.4f}\n")
                    for param, value in best_params.items():
                        f.write(f"{param}: {value}\n")
                    
            return {
                'params': best_params,
                'performance': best_value
            }
        except Exception as e:
            self.logger.error(f"Error in hyperparameter optimization: {str(e)}")
            raise
    
    def train_single_model(self, X_train: pd.DataFrame, y_train: pd.Series,
                             model_type: str = 'gb', params: Dict[str, Any] = None) -> BaseEstimator:
        """
        Train a single model with specified parameters.
        
        Args:
            X_train: Training features
            y_train: Training target
            model_type: Model type
            params: Model parameters
            
        Returns:
            Trained model (wrapped in a pipeline)
        """
        model = self.get_model_instance(model_type, params)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        self.logger.info(f"Training {model_type} model with {len(X_train)} samples and {X_train.shape[1]} features")
        model.fit(X_train_scaled, y_train)
        
        return Pipeline([('scaler', scaler), ('model', model)])
    
    def train_models(self, train_data: pd.DataFrame, test_data: pd.DataFrame,
                     feature_sets: Dict[str, List[str]]) -> Dict[str, Dict[str, Any]]:
        """
        Train multiple models on different feature sets.
        
        Args:
            train_data: Training DataFrame
            test_data: Testing DataFrame
            feature_sets: Dictionary mapping set names to feature lists
            
        Returns:
            Dictionary with model results
        """
        self.logger.info("\nTraining models on different feature sets...")
        
        self.models = {}
        self.results = {}
        
        model_types = self.model_config.get('model_types', ['rf', 'gb', 'xgb', 'lgb', 'elastic', 'svr'])
        use_optimized = self.model_config.get('optimize_hyperparameters', False)
        y_train = train_data[self.target_column]
        y_test = test_data[self.target_column]
        
        for set_name, features in feature_sets.items():
            self.logger.info(f"\nTraining models with feature set '{set_name}' ({len(features)} features)")
            
            X_train = train_data[features]
            X_test = test_data[features]
            
            for model_type in model_types:
                if model_type == 'nn' and len(features) > 100:
                    self.logger.info(f"Skipping {model_type} for large feature set")
                    continue
                
                self.logger.info(f"Training {model_type} model...")
                
                if use_optimized and self.model_config.get(f'{model_type}_params'):
                    params = self.model_config.get(f'{model_type}_params')
                    self.logger.info(f"Using optimized parameters: {params}")
                else:
                    if model_type == 'rf':
                        params = {'n_estimators': 100, 'random_state': self.seed}
                    elif model_type == 'gb':
                        params = {'n_estimators': 100, 'learning_rate': 0.05, 'max_depth': 5, 'random_state': self.seed}
                    elif model_type == 'xgb':
                        params = {'n_estimators': 100, 'learning_rate': 0.05, 'max_depth': 5, 'random_state': self.seed}
                    elif model_type == 'lgb':
                        params = {'n_estimators': 100, 'learning_rate': 0.05, 'num_leaves': 31, 'random_state': self.seed, 'verbose': -1}
                    elif model_type == 'nn':
                        params = {'hidden_layer_sizes': (100, 50), 'max_iter': 1000, 'random_state': self.seed}
                    elif model_type == 'elastic':
                        params = {'alpha': 0.1, 'l1_ratio': 0.5, 'random_state': self.seed}
                    elif model_type == 'svr':
                        params = {'C': 1.0, 'epsilon': 0.1, 'kernel': 'rbf'}
                    else:
                        params = {'random_state': self.seed}
                
                try:
                    model = self.train_single_model(X_train, y_train, model_type, params)
                    y_pred = model.predict(X_test)
                    
                    mse = mean_squared_error(y_test, y_pred)
                    rmse = np.sqrt(mse)
                    mae = mean_absolute_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    
                    residuals = y_test - y_pred
                    residuals_rmse = np.sqrt(np.mean(residuals**2))
                    residuals_mean = np.mean(residuals)
                    residuals_std = np.std(residuals)
                    
                    model_key = f"{set_name}_{model_type}"
                    
                    self.models[model_key] = model
                    self.results[model_key] = {
                        'model': model,
                        'features': features,
                        'mse': mse,
                        'rmse': rmse,
                        'mae': mae,
                        'r2': r2,
                        'residuals_rmse': residuals_rmse,
                        'residuals_mean': residuals_mean,
                        'residuals_std': residuals_std,
                        'predictions': y_pred,
                        'ground_truth': y_test,
                        'residuals': residuals
                    }
                    
                    self.logger.info(f"  {model_type.upper()} Results: RMSE = {rmse:.4f}, R² = {r2:.4f}, MAE = {mae:.4f}")
                    
                    y_train_pred = model.predict(X_train)
                    train_r2 = r2_score(y_train, y_train_pred)
                    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
                    
                    self.results[model_key]['train_r2'] = train_r2
                    self.results[model_key]['train_rmse'] = train_rmse
                    
                    r2_gap = train_r2 - r2
                    rmse_gap = rmse - train_rmse
                    
                    self.results[model_key]['r2_gap'] = r2_gap
                    self.results[model_key]['rmse_gap'] = rmse_gap
                    
                    if r2_gap > 0.2:
                        self.logger.warning(f"  Potential overfitting detected: Train R² = {train_r2:.4f}, Test R² = {r2:.4f}, Gap = {r2_gap:.4f}")
                    
                    if self.output_dir:
                        self._visualize_predictions(y_test, y_pred, model_key)
                        self._visualize_residuals(residuals, model_key)
                    
                except Exception as e:
                    self.logger.error(f"Error training {model_type} model: {str(e)}")
        
        self._identify_best_model()
        
        if self.output_dir and self.model_config.get('save_models', True):
            for model_key, model in self.models.items():
                model_path = os.path.join(self.output_dir, 'models', f"{model_key}.joblib")
                joblib.dump(model, model_path)
            if self.best_model:
                best_model_path = os.path.join(self.output_dir, 'best_model.joblib')
                joblib.dump(self.best_model, best_model_path)
        
        return self.results
    
    def _identify_best_model(self) -> None:
        """
        Identify the best performing model based on R² score and overfitting gap.
        """
        if not self.results:
            self.logger.warning("No model results available")
            return
        
        sorted_models = sorted(self.results.items(), key=lambda x: x[1]['r2'], reverse=True)
        
        filtered_models = []
        for model_key, result in sorted_models:
            if result.get('r2_gap', 0) < 0.2:
                filtered_models.append((model_key, result))
            else:
                self.logger.warning(f"Model {model_key} excluded from best model selection due to overfitting")
        
        if not filtered_models:
            filtered_models = sorted_models
            self.logger.warning("All models show signs of overfitting, using best performing model anyway")
        
        best_key, best_result = filtered_models[0]
        
        self.best_model = {
            'key': best_key,
            'model': self.models[best_key],
            'features': best_result['features'],
            'r2': best_result['r2'],
            'rmse': best_result['rmse'],
            'mae': best_result['mae'],
            'train_r2': best_result.get('train_r2', 0),
            'r2_gap': best_result.get('r2_gap', 0)
        }
        
        self.logger.info(f"\nBest model: {best_key}")
        self.logger.info(f"  R² = {best_result['r2']:.4f}")
        self.logger.info(f"  RMSE = {best_result['rmse']:.4f}")
        self.logger.info(f"  MAE = {best_result['mae']:.4f}")
        self.logger.info(f"  Train R² = {best_result.get('train_r2', 0):.4f}")
        self.logger.info(f"  R² Gap = {best_result.get('r2_gap', 0):.4f}")
    
    def _visualize_predictions(self, y_true: np.ndarray, y_pred: np.ndarray, model_key: str) -> None:
        """
        Create a visualization of predicted vs. actual values.
        
        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            model_key: Model identifier
        """
        if not self.output_dir:
            return
            
        plt.figure(figsize=(10, 8))
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        plt.hexbin(y_true, y_pred, gridsize=30, cmap='viridis', alpha=0.8)
        plt.colorbar(label='Count')
        
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        
        plt.text(0.05, 0.95, f'R² = {r2:.4f}\nRMSE = {rmse:.4f}\nMAE = {mae:.4f}',
                 transform=plt.gca().transAxes, fontsize=12,
                 verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.2))
        
        plt.xlabel('Actual', fontsize=12)
        plt.ylabel('Predicted', fontsize=12)
        plt.title(f'Actual vs. Predicted Cognitive Load: {model_key}', fontsize=14)
        plt.axis('equal')
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, f'actual_vs_pred_{model_key}.png')
        plt.savefig(save_path, dpi=300)
        plt.close()
    
    def _visualize_residuals(self, residuals: np.ndarray, model_key: str) -> None:
        """
        Create visualizations of model residuals.
        
        Args:
            residuals: Residual values (y_true - y_pred)
            model_key: Model identifier
        """
        if not self.output_dir:
            return
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        sns.histplot(residuals, kde=True, ax=ax1, color='steelblue')
        ax1.axvline(x=0, color='r', linestyle='--', alpha=0.8)
        ax1.set_xlabel('Residual Value', fontsize=12)
        ax1.set_ylabel('Count', fontsize=12)
        ax1.set_title('Residuals Distribution', fontsize=14)
        mean_val = np.mean(residuals)
        std_val = np.std(residuals)
        ax1.text(0.95, 0.95, f'Mean = {mean_val:.4f}\nStd = {std_val:.4f}',
                 transform=ax1.transAxes, fontsize=10,
                 horizontalalignment='right',
                 verticalalignment='top', 
                 bbox=dict(boxstyle='round', alpha=0.2))
        
        from scipy import stats
        stats.probplot(residuals, plot=ax2)
        ax2.set_title('Residuals Q-Q Plot', fontsize=14)
        
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, f'residuals_{model_key}.png')
        plt.savefig(save_path, dpi=300)
        plt.close()
    
    def analyze_feature_importance(self, data: pd.DataFrame, model_key: Optional[str] = None) -> pd.DataFrame:
        """
        Get feature importance for the given model.
        
        Args:
            data: DataFrame with features and target
            model_key: Key of the model to analyze, or None for the best model
            
        Returns:
            DataFrame with feature importances
        """
        self.logger.info("\nAnalyzing feature importance...")
        
        if model_key is None:
            if self.best_model is None:
                self.logger.error("No best model identified")
                return pd.DataFrame()
            model_key = self.best_model['key']
        
        if model_key not in self.models:
            self.logger.error(f"Model {model_key} not found")
            return pd.DataFrame()
        
        model = self.models[model_key]
        features = self.results[model_key]['features']
        
        if hasattr(model, 'named_steps') and 'model' in model.named_steps:
            actual_model = model.named_steps['model']
        else:
            actual_model = model
        
        X = data[features]
        y = data[self.target_column]
        
        if hasattr(model, 'named_steps') and 'scaler' in model.named_steps:
            scaler = model.named_steps['scaler']
            X_scaled = scaler.transform(X)
        else:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
        
        importance_methods = []
        
        if hasattr(actual_model, 'feature_importances_'):
            importance = actual_model.feature_importances_
            importance_df = pd.DataFrame({'feature': features, 'importance': importance}).sort_values('importance', ascending=False)
            importance_methods.append(('built_in', importance_df))
            self.logger.info("Top 10 features by built-in importance:")
            for i, (_, row) in enumerate(importance_df.head(10).iterrows()):
                self.logger.info(f"  {i+1}. {row['feature']}: {row['importance']:.4f}")
        
        try:
            self.logger.info("Calculating permutation importance...")
            perm = permutation_importance(model, X, y, n_repeats=10, random_state=self.seed)
            perm_importance = perm.importances_mean
            perm_df = pd.DataFrame({'feature': features, 'importance': perm_importance}).sort_values('importance', ascending=False)
            importance_methods.append(('permutation', perm_df))
            self.logger.info("Top 10 features by permutation importance:")
            for i, (_, row) in enumerate(perm_df.head(10).iterrows()):
                self.logger.info(f"  {i+1}. {row['feature']}: {row['importance']:.4f}")
        except Exception as e:
            self.logger.warning(f"Error calculating permutation importance: {str(e)}")
        
        if self.output_dir:
            for method_name, imp_df in importance_methods:
                self._visualize_importance(imp_df, method_name, model_key)
            self._visualize_feature_correlation(data, features, model_key)
        
        if importance_methods:
            # Return the primary method's importance (if combined is not used here)
            return importance_methods[0][1]
        else:
            return pd.DataFrame()
    
    def _visualize_importance(self, importance_df: pd.DataFrame, method: str, model_key: str) -> None:
        """
        Create visualization of feature importance.
        
        Args:
            importance_df: DataFrame with feature importance
            method: Method used for importance calculation
            model_key: Model identifier
        """
        if not self.output_dir:
            return
            
        top_n = min(20, len(importance_df))
        top_df = importance_df.sort_values("importance", ascending=False).head(top_n)
        
        plt.figure(figsize=(12, 10))
        features = top_df['feature'].values[::-1]
        values = top_df['importance'].values[::-1]
        colors = plt.cm.viridis(np.linspace(0, 1, len(features)))
        bars = plt.barh(range(len(features)), values, align='center', color=colors)
        plt.yticks(range(len(features)), features)
        plt.xlabel('Importance', fontsize=12)
        plt.title(f'Feature Importance ({method}): {model_key}', fontsize=14)
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width, bar.get_y() + bar.get_height()/2, f' {width:.4f}', va='center')
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, f'feature_importance_{method}_{model_key}.png')
        plt.savefig(save_path, dpi=300)
        plt.close()
    
    def _visualize_feature_correlation(self, data: pd.DataFrame, features: List[str], model_key: str) -> None:
        """
        Create a correlation heatmap for the selected features.
        
        Args:
            data: DataFrame with features
            features: List of feature names
            model_key: Model identifier
        """
        if not self.output_dir:
            return
            
        if len(features) > 20:
            top_features = features[:20]
        else:
            top_features = features
        
        columns = top_features + [self.target_column]
        corr_matrix = data[columns].corr()
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', vmin=-1, vmax=1, center=0, linewidths=0.5)
        plt.title(f'Feature Correlation Matrix: {model_key}', fontsize=14)
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, f'feature_correlation_{model_key}.png')
        plt.savefig(save_path, dpi=300)
        plt.close()
    
    def analyze_results_by_category(self, data: pd.DataFrame, 
                                    model: Optional[Dict[str, Any]] = None) -> Dict[str, Dict[str, float]]:
        """
        Analyze model performance by pilot category.
        
        Args:
            data: DataFrame with pilot category information
            model: Model information dictionary or None for best model
            
        Returns:
            Dictionary with performance metrics by category
        """
        if model is None:
            if self.best_model is None:
                self.logger.error("No best model identified")
                return {}
            model = self.best_model
        
        if 'pilot_category' not in data.columns:
            self.logger.warning("Pilot category information not available")
            return {}
            
        self.logger.info("\nAnalyzing results by pilot category...")
        model_obj = model['model']
        features = model['features']
        
        X = data[features]
        y = data[self.target_column]
        
        y_pred = model_obj.predict(X)
        
        result_data = data.copy()
        result_data['predicted'] = y_pred
        result_data['error'] = result_data['predicted'] - result_data[self.target_column]
        result_data['abs_error'] = np.abs(result_data['error'])
        result_data['squared_error'] = result_data['error'] ** 2
        
        category_metrics = {}
        for category in result_data['pilot_category'].unique():
            cat_data = result_data[result_data['pilot_category'] == category]
            metrics = {
                'count': len(cat_data),
                'rmse': np.sqrt(cat_data['squared_error'].mean()),
                'mae': cat_data['abs_error'].mean(),
                'r2': r2_score(cat_data[self.target_column], cat_data['predicted']),
                'mean_error': cat_data['error'].mean(),
                'max_error': cat_data['abs_error'].max()
            }
            category_metrics[category] = metrics
            self.logger.info(f"\nPerformance for {category} pilots:")
            self.logger.info(f"  Count: {metrics['count']}")
            self.logger.info(f"  RMSE: {metrics['rmse']:.4f}")
            self.logger.info(f"  MAE: {metrics['mae']:.4f}")
            self.logger.info(f"  R²: {metrics['r2']:.4f}")
            self.logger.info(f"  Mean Error: {metrics['mean_error']:.4f}")
            self.logger.info(f"  Max Error: {metrics['max_error']:.4f}")
        
        if self.output_dir:
            self._visualize_category_performance(category_metrics)
            self._visualize_category_predictions(result_data)
        
        return category_metrics
    
    def _visualize_category_performance(self, category_metrics: Dict[str, Dict[str, float]]) -> None:
        """
        Create visualization of model performance by category.
        
        Args:
            category_metrics: Dictionary with metrics by category
        """
        if not self.output_dir:
            return
            
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Model Performance by Pilot Category', fontsize=16)
        
        categories = list(category_metrics.keys())
        colors = plt.cm.viridis(np.linspace(0, 1, len(categories)))
        
        rmse_values = [category_metrics[cat]['rmse'] for cat in categories]
        axes[0].bar(categories, rmse_values, color=colors)
        axes[0].set_title('RMSE by Category')
        axes[0].set_ylabel('RMSE (lower is better)')
        
        mae_values = [category_metrics[cat]['mae'] for cat in categories]
        axes[1].bar(categories, mae_values, color=colors)
        axes[1].set_title('MAE by Category')
        axes[1].set_ylabel('MAE (lower is better)')
        
        r2_values = [category_metrics[cat]['r2'] for cat in categories]
        axes[2].bar(categories, r2_values, color=colors)
        axes[2].set_title('R² by Category')
        axes[2].set_ylabel('R² (higher is better)')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        save_path = os.path.join(self.output_dir, 'performance_by_category.png')
        plt.savefig(save_path, dpi=300)
        plt.close()
    
    def _visualize_category_predictions(self, result_data: pd.DataFrame) -> None:
        """
        Create visualizations of predictions by pilot category.
        
        Args:
            result_data: DataFrame with predictions and categories
        """
        if not self.output_dir:
            return
            
        plt.figure(figsize=(12, 8))
        categories = result_data['pilot_category'].unique()
        colors = plt.cm.viridis(np.linspace(0, 1, len(categories)))
        
        for i, category in enumerate(categories):
            cat_data = result_data[result_data['pilot_category'] == category]
            plt.scatter(cat_data[self.target_column], cat_data['predicted'], 
                        alpha=0.7, label=category, color=colors[i])
                
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        plt.xlabel('Actual Cognitive Load')
        plt.ylabel('Predicted Cognitive Load')
        plt.title('Predictions vs Actual by Pilot Category')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, 'predictions_by_category.png')
        plt.savefig(save_path, dpi=300)
        plt.close()
        
        plt.figure(figsize=(12, 8))
        for i, category in enumerate(categories):
            cat_data = result_data[result_data['pilot_category'] == category]
            sns.kdeplot(cat_data['error'], label=category, color=colors[i])
        plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
        plt.xlabel('Prediction Error')
        plt.ylabel('Density')
        plt.title('Error Distribution by Pilot Category')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, 'error_distribution.png')
        plt.savefig(save_path, dpi=300)
        plt.close()
    
    def analyze_turbulence_performance(self, data: pd.DataFrame, 
                                       model: Optional[Dict[str, Any]] = None) -> Dict[str, Dict[str, float]]:
        """
        Analyze model performance by turbulence level.
        
        Args:
            data: DataFrame with turbulence information
            model: Model information dictionary or None for best model
            
        Returns:
            Dictionary with performance metrics by turbulence level
        """
        if model is None:
            if self.best_model is None:
                self.logger.error("No best model identified")
                return {}
            model = self.best_model
        
        if 'turbulence' not in data.columns:
            self.logger.warning("Turbulence information not available")
            return {}
            
        self.logger.info("\nAnalyzing results by turbulence level...")
        model_obj = model['model']
        features = model['features']
        
        X = data[features]
        y = data[self.target_column]
        
        y_pred = model_obj.predict(X)
        result_data = data.copy()
        result_data['predicted'] = y_pred
        result_data['error'] = result_data['predicted'] - result_data[self.target_column]
        result_data['abs_error'] = np.abs(result_data['error'])
        result_data['squared_error'] = result_data['error'] ** 2
        
        turb_metrics = {}
        for level in sorted(result_data['turbulence'].unique()):
            turb_data = result_data[result_data['turbulence'] == level]
            metrics = {
                'count': len(turb_data),
                'rmse': np.sqrt(turb_data['squared_error'].mean()),
                'mae': turb_data['abs_error'].mean(),
                'r2': r2_score(turb_data[self.target_column], turb_data['predicted']),
                'mean_error': turb_data['error'].mean(),
                'max_error': turb_data['abs_error'].max()
            }
            turb_metrics[float(level)] = metrics
            self.logger.info(f"\nPerformance for turbulence level {level}:")
            self.logger.info(f"  Count: {metrics['count']}")
            self.logger.info(f"  RMSE: {metrics['rmse']:.4f}")
            self.logger.info(f"  MAE: {metrics['mae']:.4f}")
            self.logger.info(f"  R²: {metrics['r2']:.4f}")
            self.logger.info(f"  Mean Error: {metrics['mean_error']:.4f}")
            self.logger.info(f"  Max Error: {metrics['max_error']:.4f}")
        
        if self.output_dir:
            self._visualize_turbulence_performance(turb_metrics)
            self._visualize_turbulence_predictions(result_data)
        
        return turb_metrics
    
    def _visualize_turbulence_performance(self, turb_metrics: Dict[float, Dict[str, float]]) -> None:
        """
        Create visualization of model performance by turbulence level.
        
        Args:
            turb_metrics: Dictionary with metrics by turbulence level
        """
        if not self.output_dir:
            return
            
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Model Performance by Turbulence Level', fontsize=16)
        
        levels = sorted(list(turb_metrics.keys()))
        colors = plt.cm.viridis(np.linspace(0, 1, len(levels)))
        
        rmse_values = [turb_metrics[level]['rmse'] for level in levels]
        axes[0].bar(levels, rmse_values, color=colors)
        axes[0].set_title('RMSE by Turbulence Level')
        axes[0].set_xlabel('Turbulence Level')
        axes[0].set_ylabel('RMSE (lower is better)')
        
        mae_values = [turb_metrics[level]['mae'] for level in levels]
        axes[1].bar(levels, mae_values, color=colors)
        axes[1].set_title('MAE by Turbulence Level')
        axes[1].set_xlabel('Turbulence Level')
        axes[1].set_ylabel('MAE (lower is better)')
        
        r2_values = [turb_metrics[level]['r2'] for level in levels]
        axes[2].bar(levels, r2_values, color=colors)
        axes[2].set_title('R² by Turbulence Level')
        axes[2].set_xlabel('Turbulence Level')
        axes[2].set_ylabel('R² (higher is better)')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        save_path = os.path.join(self.output_dir, 'performance_by_turbulence.png')
        plt.savefig(save_path, dpi=300)
        plt.close()
    
    def _visualize_turbulence_predictions(self, result_data: pd.DataFrame) -> None:
        """
        Create visualizations of predictions by turbulence level.
        
        Args:
            result_data: DataFrame with predictions and turbulence levels
        """
        if not self.output_dir:
            return
            
        plt.figure(figsize=(12, 8))
        levels = sorted(result_data['turbulence'].unique())
        colors = plt.cm.viridis(np.linspace(0, 1, len(levels)))
        
        for i, level in enumerate(levels):
            level_data = result_data[result_data['turbulence'] == level]
            plt.scatter(level_data[self.target_column], level_data['predicted'], 
                        alpha=0.7, label=f'Level {level}', color=colors[i])
                
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        plt.xlabel('Actual Cognitive Load')
        plt.ylabel('Predicted Cognitive Load')
        plt.title('Predictions vs Actual by Turbulence Level')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, 'predictions_by_turbulence.png')
        plt.savefig(save_path, dpi=300)
        plt.close()
        
        plt.figure(figsize=(12, 8))
        sns.boxplot(x='turbulence', y='error', data=result_data)
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        plt.xlabel('Turbulence Level')
        plt.ylabel('Prediction Error')
        plt.title('Error Distribution by Turbulence Level')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, 'error_by_turbulence.png')
        plt.savefig(save_path, dpi=300)
        plt.close()
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the best trained model.
        
        Args:
            data: DataFrame with features
            
        Returns:
            Array of predictions
        """
        if self.best_model is None:
            self.logger.error("No best model available. Train models first.")
            raise ValueError("No best model available. Train models first.")
        
        model = self.best_model['model']
        features = self.best_model['features']
        
        missing_features = [f for f in features if f not in data.columns]
        if missing_features:
            self.logger.error(f"Missing required features: {missing_features}")
            raise ValueError(f"Missing required features: {missing_features}")
        
        X = data[features]
        predictions = model.predict(X)
        return predictions

