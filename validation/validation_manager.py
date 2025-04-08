import os
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import learning_curve, train_test_split
from sklearn.utils import resample
import joblib
from tqdm import tqdm


class ValidationManager:
    """
    Manages advanced validation strategies for cognitive load prediction models
    to detect and prevent overfitting.
    """
    
    def __init__(self, id_column: str = 'pilot_id',
                 target_column: str = 'avg_tlx_quantile',
                 seed: int = 42,
                 output_dir: str = None,
                 config: Dict[str, Any] = None):
        """
        Initialize the ValidationManager.
        
        Args:
            id_column: Column name for subject identifier
            target_column: Column name for the target variable
            seed: Random seed for reproducibility
            output_dir: Directory to save validation results
            config: Dictionary with validation configuration
        """
        self.id_column = id_column
        self.target_column = target_column
        self.seed = seed
        self.output_dir = output_dir
        self.config = config or {
            'group_kfold_splits': 5,
            'leave_one_pilot_out': True,
            'generate_learning_curves': True,
            'permutation_test': True,
            'significance_testing': True,
            'alpha': 0.05
        }
        
        # Results containers
        self.results = {
            'lopo': {},  # Leave-one-pilot-out results
            'learning_curves': {},
            'permutation': {},
            'significance': {}
        }
        
        # Logger
        self.logger = logging.getLogger(__name__)
        
        # Set random seed
        np.random.seed(self.seed)
    
    def leave_one_pilot_out_validation(self, data: pd.DataFrame,
                                     feature_engineer: Any,
                                     feature_selector: Any,
                                     model_trainer: Any) -> Dict[str, Any]:
        """
        Perform leave-one-pilot-out validation to evaluate generalization to new pilots.
        
        Args:
            data: Input DataFrame
            feature_engineer: Feature engineering object
            feature_selector: Feature selection object
            model_trainer: Model training object
            
        Returns:
            Dictionary with validation results
        """
        self.logger.info("\nPerforming leave-one-pilot-out validation...")
        
        # Check if ID column exists
        if self.id_column not in data.columns:
            self.logger.error(f"ID column '{self.id_column}' not found in data")
            raise ValueError(f"ID column '{self.id_column}' not found in data")
        
        # Check if target column exists
        if self.target_column not in data.columns:
            self.logger.error(f"Target column '{self.target_column}' not found in data")
            raise ValueError(f"Target column '{self.target_column}' not found in data")
        
        # Get unique pilots
        pilots = data[self.id_column].unique()
        n_pilots = len(pilots)
        
        self.logger.info(f"Running leave-one-pilot-out validation for {n_pilots} pilots")
        
        # Store results for each pilot
        pilot_results = {}
        
        # Use tqdm for progress tracking
        for i, test_pilot in enumerate(tqdm(pilots, desc="Leave-one-pilot-out validation")):
            self.logger.info(f"\nValidation for pilot {test_pilot} ({i+1}/{n_pilots})")
            
            # Split data
            train_data = data[data[self.id_column] != test_pilot].copy().reset_index(drop=True)
            test_data = data[data[self.id_column] == test_pilot].copy().reset_index(drop=True)
            
            # Engineer features
            train_engineered = feature_engineer.engineer_features(train_data)
            test_engineered = feature_engineer.engineer_features(test_data)
            
            # Select features
            selected_features = feature_selector.select_features(train_engineered)
            
            # Train model on best feature set
            # Here we use the 'optimal' feature set with default hyperparameters
            feature_sets = feature_selector.create_feature_sets(train_engineered, selected_features)
            optimal_features = feature_sets.get('optimal', selected_features[:min(20, len(selected_features))])
            
            # Train and evaluate model
            model_key = 'gb'  # Use gradient boosting as default
            model_params = {
                'n_estimators': 100,
                'learning_rate': 0.05,
                'max_depth': 5,
                'random_state': self.seed
            }
            
            # Train model
            X_train = train_engineered[optimal_features]
            y_train = train_engineered[self.target_column]
            
            X_test = test_engineered[optimal_features]
            y_test = test_engineered[self.target_column]
            
            # Train model
            try:
                model = model_trainer.train_single_model(X_train, y_train, model_key, model_params)
                
                # Make predictions
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                # Store results
                pilot_results[test_pilot] = {
                    'mse': mse,
                    'rmse': rmse,
                    'mae': mae,
                    'r2': r2,
                    'n_train_samples': len(train_data),
                    'n_test_samples': len(test_data),
                    'n_features': len(optimal_features)
                }
                
                self.logger.info(f"  Results for pilot {test_pilot}: RMSE = {rmse:.4f}, R² = {r2:.4f}")
                
                # Store predictions
                if 'pilot_category' in test_data.columns:
                    category = test_data['pilot_category'].iloc[0]
                    pilot_results[test_pilot]['category'] = category
            except Exception as e:
                self.logger.error(f"Error in validation for pilot {test_pilot}: {str(e)}")
        
        # Calculate overall statistics
        r2_values = [res['r2'] for res in pilot_results.values()]
        rmse_values = [res['rmse'] for res in pilot_results.values()]
        
        mean_r2 = np.mean(r2_values)
        std_r2 = np.std(r2_values)
        mean_rmse = np.mean(rmse_values)
        std_rmse = np.std(rmse_values)
        
        self.logger.info(f"\nOverall leave-one-pilot-out validation results:")
        self.logger.info(f"Mean R²: {mean_r2:.4f} ± {std_r2:.4f}")
        self.logger.info(f"Mean RMSE: {mean_rmse:.4f} ± {std_rmse:.4f}")
        
        # Check for pilots with negative R²
        negative_r2 = [pilot for pilot, res in pilot_results.items() if res['r2'] < 0]
        if negative_r2:
            self.logger.warning(f"Found {len(negative_r2)} pilots with negative R² (poor generalization): {negative_r2}")
        
        # Create aggregated results
        lopo_results = {
            'pilot_results': pilot_results,
            'mean_r2': mean_r2,
            'std_r2': std_r2,
            'mean_rmse': mean_rmse,
            'std_rmse': std_rmse,
            'negative_r2_pilots': negative_r2
        }
        
        # Add by-category statistics if available
        if any('category' in res for res in pilot_results.values()):
            category_stats = {}
            for category in ['minimal_exp', 'commercial', 'air_force']:
                cat_pilots = [pilot for pilot, res in pilot_results.items() 
                             if res.get('category') == category]
                
                if cat_pilots:
                    cat_r2 = [pilot_results[pilot]['r2'] for pilot in cat_pilots]
                    cat_rmse = [pilot_results[pilot]['rmse'] for pilot in cat_pilots]
                    
                    category_stats[category] = {
                        'n_pilots': len(cat_pilots),
                        'mean_r2': np.mean(cat_r2),
                        'std_r2': np.std(cat_r2),
                        'mean_rmse': np.mean(cat_rmse),
                        'std_rmse': np.std(cat_rmse)
                    }
                    
                    self.logger.info(f"  {category} pilots: Mean R² = {np.mean(cat_r2):.4f} ± {np.std(cat_r2):.4f}")
            
            lopo_results['category_stats'] = category_stats
        
        # Create visualizations
        if self.output_dir:
            self._visualize_lopo_results(lopo_results)
        
        # Store results
        self.results['lopo'] = lopo_results
        
        return lopo_results
    
    def generate_learning_curves(self, data: pd.DataFrame,
                              feature_engineer: Any,
                              feature_selector: Any,
                              model_trainer: Any) -> Dict[str, Any]:
        """
        Generate learning curves to analyze how model performance changes with training set size.
        
        Args:
            data: Input DataFrame
            feature_engineer: Feature engineering object
            feature_selector: Feature selection object
            model_trainer: Model training object
            
        Returns:
            Dictionary with learning curve results
        """
        self.logger.info("\nGenerating learning curves...")
        
        # Engineer features
        engineered_data = feature_engineer.engineer_features(data)
        
        # Select features
        selected_features = feature_selector.select_features(engineered_data)
        
        # Create feature sets
        feature_sets = feature_selector.create_feature_sets(engineered_data, selected_features)
        optimal_features = feature_sets.get('optimal', selected_features[:min(20, len(selected_features))])
        
        # Prepare data
        X = engineered_data[optimal_features]
        y = engineered_data[self.target_column]
        
        # Define train sizes
        train_sizes = np.linspace(0.1, 1.0, 10)
        
        # Define model
        model = model_trainer.get_model_instance('gb', {
            'n_estimators': 100,
            'learning_rate': 0.05,
            'max_depth': 5,
            'random_state': self.seed
        })
        
        # Generate learning curves
        self.logger.info("Calculating learning curves - this may take some time...")
        try:
            # Get groups for subject independence
            groups = engineered_data[self.id_column]
            
            # Create custom learning curve function that respects subject independence
            train_sizes_abs, train_scores, test_scores = self._custom_learning_curve(
                model, X, y, groups=groups, train_sizes=train_sizes, cv=5, 
                scoring='r2', random_state=self.seed
            )
            
            # Store learning curve data
            learning_curve_results = {
                'train_sizes_abs': train_sizes_abs,
                'train_scores': train_scores,
                'test_scores': test_scores,
                'train_sizes': train_sizes,
                'features': optimal_features
            }
            
            # Calculate means and standard deviations
            train_scores_mean = np.mean(train_scores, axis=1)
            train_scores_std = np.std(train_scores, axis=1)
            test_scores_mean = np.mean(test_scores, axis=1)
            test_scores_std = np.std(test_scores, axis=1)
            
            # Look for signs of overfitting
            # 1. Large gap between train and test scores
            gap = train_scores_mean - test_scores_mean
            max_gap = np.max(gap)
            
            # 2. Decreasing test scores with increasing train size
            test_slope = np.polyfit(train_sizes, test_scores_mean, 1)[0]
            
            # 3. High variance in test scores
            test_variance = np.mean(test_scores_std)
            
            # Analyze learning curve
            overfitting_score = 0
            
            if max_gap > 0.2:  # Significant gap between train and test performance
                overfitting_score += max_gap * 5
                self.logger.warning(f"Large gap detected between train and test performance: {max_gap:.4f}")
            
            if test_slope < -0.05:  # Test performance decreases with more data
                overfitting_score -= test_slope * 10
                self.logger.warning(f"Test performance decreases with more training data (slope: {test_slope:.4f})")
            
            if test_variance > 0.1:  # High variance in test scores
                overfitting_score += test_variance * 5
                self.logger.warning(f"High variance in test scores: {test_variance:.4f}")
            
            self.logger.info(f"Overfitting risk score: {overfitting_score:.4f}")
            
            learning_curve_results['overfitting_analysis'] = {
                'max_gap': max_gap,
                'test_slope': test_slope,
                'test_variance': test_variance,
                'overfitting_score': overfitting_score
            }
            
            # Create visualization
            if self.output_dir:
                self._visualize_learning_curves(train_sizes, train_scores_mean, train_scores_std,
                                             test_scores_mean, test_scores_std)
            
            # Store results
            self.results['learning_curves'] = learning_curve_results
            
            return learning_curve_results
            
        except Exception as e:
            self.logger.error(f"Error generating learning curves: {str(e)}")
            return {}
    
    def _custom_learning_curve(self, estimator, X, y, groups, train_sizes, cv, scoring, random_state):
        """
        Custom implementation of learning curve that respects subject independence.
        
        Args:
            estimator: The estimator to use for learning curve
            X: Feature matrix
            y: Target values
            groups: Group labels for subject independence
            train_sizes: Array of training set sizes
            cv: Number of CV folds
            scoring: Scoring metric
            random_state: Random seed
            
        Returns:
            Tuple of (train_sizes_abs, train_scores, test_scores)
        """
        from sklearn.model_selection import GroupKFold
        from sklearn.metrics import check_scoring
        
        # Create GroupKFold
        cv_splitter = GroupKFold(n_splits=cv)
        
        # Create scoring function
        scorer = check_scoring(estimator, scoring=scoring)
        
        # Initialize arrays
        train_sizes_abs = np.zeros(len(train_sizes), dtype=int)
        train_scores = np.zeros((len(train_sizes), cv))
        test_scores = np.zeros((len(train_sizes), cv))
        
        # Create folds
        splits = list(cv_splitter.split(X, y, groups))
        
        # Iterate over each fold
        for i, (train_idx, test_idx) in enumerate(splits):
            # Split data
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Get groups for the training set
            train_groups = groups.iloc[train_idx]
            
            # Get unique subjects in the training set
            unique_subjects = train_groups.unique()
            n_unique_subjects = len(unique_subjects)
            
            # Calculate absolute train sizes based on number of subjects
            n_max_samples = len(X_train)
            
            # Iterate over train sizes
            for j, frac in enumerate(train_sizes):
                # Calculate the number of subjects to include
                n_subjects = max(1, int(frac * n_unique_subjects))
                
                # Select random subjects
                np.random.seed(random_state + i)
                selected_subjects = np.random.choice(unique_subjects, n_subjects, replace=False)
                
                # Select samples from these subjects
                subject_mask = train_groups.isin(selected_subjects)
                X_train_sub = X_train[subject_mask]
                y_train_sub = y_train[subject_mask]
                
                # Store actual number of samples
                train_sizes_abs[j] = len(X_train_sub)
                
                # Train model
                estimator.fit(X_train_sub, y_train_sub)
                
                # Score on train and test sets
                train_scores[j, i] = scorer(estimator, X_train_sub, y_train_sub)
                test_scores[j, i] = scorer(estimator, X_test, y_test)
        
        return train_sizes_abs, train_scores, test_scores
    
    def perform_permutation_test(self, data: pd.DataFrame, model: Any,
                              features: List[str], n_permutations: int = 100) -> Dict[str, Any]:
        """
        Perform permutation test to check if the model performs better than chance.
        
        Args:
            data: DataFrame with features and target
            model: Trained model
            features: List of features used by the model
            n_permutations: Number of permutations to perform
            
        Returns:
            Dictionary with permutation test results
        """
        self.logger.info(f"\nPerforming permutation test with {n_permutations} permutations...")
        
        # Prepare data
        X = data[features]
        y = data[self.target_column]
        
        # Get true model performance
        y_pred = model.predict(X)
        true_r2 = r2_score(y, y_pred)
        true_rmse = np.sqrt(mean_squared_error(y, y_pred))
        
        self.logger.info(f"True model performance: R² = {true_r2:.4f}, RMSE = {true_rmse:.4f}")
        
        # Perform permutation test
        permutation_r2 = []
        permutation_rmse = []
        
        for i in tqdm(range(n_permutations), desc="Permutation test"):
            # Permute target
            y_permuted = y.sample(frac=1.0, random_state=self.seed + i).reset_index(drop=True)
            
            # Recalculate metrics
            perm_r2 = r2_score(y_permuted, y_pred)
            perm_rmse = np.sqrt(mean_squared_error(y_permuted, y_pred))
            
            permutation_r2.append(perm_r2)
            permutation_rmse.append(perm_rmse)
        
        # Calculate p-values
        p_value_r2 = np.mean([p >= true_r2 for p in permutation_r2])
        p_value_rmse = np.mean([p <= true_rmse for p in permutation_rmse])
        
        self.logger.info(f"Permutation test results:")
        self.logger.info(f"  R² p-value: {p_value_r2:.4f}")
        self.logger.info(f"  RMSE p-value: {p_value_rmse:.4f}")
        
        # Determine if results are statistically significant
        alpha = self.config.get('alpha', 0.05)
        significant_r2 = p_value_r2 < alpha
        significant_rmse = p_value_rmse < alpha
        
        self.logger.info(f"  R² is {'statistically significant' if significant_r2 else 'not statistically significant'}")
        self.logger.info(f"  RMSE is {'statistically significant' if significant_rmse else 'not statistically significant'}")
        
        # Create results dictionary
        permutation_results = {
            'true_r2': true_r2,
            'true_rmse': true_rmse,
            'permutation_r2': permutation_r2,
            'permutation_rmse': permutation_rmse,
            'p_value_r2': p_value_r2,
            'p_value_rmse': p_value_rmse,
            'significant_r2': significant_r2,
            'significant_rmse': significant_rmse,
            'n_permutations': n_permutations,
            'alpha': alpha
        }
        
        # Create visualization
        if self.output_dir:
            self._visualize_permutation_test(permutation_results)
        
        # Store results
        self.results['permutation'] = permutation_results
        
        return permutation_results
    
    def compare_models_statistically(self, data: pd.DataFrame, 
                                 models: Dict[str, Any], 
                                 features: Dict[str, List[str]]) -> Dict[str, Any]:
        """
        Compare models statistically using bootstrapped confidence intervals.
        
        Args:
            data: DataFrame with features and target
            models: Dictionary mapping model names to model objects
            features: Dictionary mapping model names to feature lists
            
        Returns:
            Dictionary with statistical comparison results
        """
        self.logger.info("\nComparing models statistically...")
        
        # Check if we have at least two models
        if len(models) < 2:
            self.logger.warning("Need at least two models for statistical comparison")
            return {}
        
        # Prepare target
        y = data[self.target_column]
        
        # Calculate predictions and errors for each model
        predictions = {}
        errors = {}
        metrics = {}
        
        for model_name, model in models.items():
            # Get features for this model
            model_features = features.get(model_name)
            
            if model_features is None:
                self.logger.warning(f"No features defined for model {model_name}")
                continue
            
            # Get predictions
            X = data[model_features]
            
            try:
                pred = model.predict(X)
                predictions[model_name] = pred
                
                # Calculate errors
                error = y - pred
                errors[model_name] = error
                
                # Calculate metrics
                metrics[model_name] = {
                    'r2': r2_score(y, pred),
                    'rmse': np.sqrt(mean_squared_error(y, pred)),
                    'mae': mean_absolute_error(y, pred)
                }
            except Exception as e:
                self.logger.error(f"Error getting predictions for model {model_name}: {str(e)}")
        
        # Bootstrap confidence intervals
        n_bootstrap = 1000
        alpha = self.config.get('alpha', 0.05)
        
        bootstrap_results = {}
        
        for model_name in predictions.keys():
            bootstrap_r2 = []
            bootstrap_rmse = []
            bootstrap_mae = []
            
            for i in range(n_bootstrap):
                # Bootstrap resample
                idx = resample(np.arange(len(y)), replace=True, random_state=self.seed + i, n_samples=len(y))
                
                # Calculate metrics on bootstrap sample
                bs_y = y.iloc[idx]
                bs_pred = predictions[model_name][idx]
                
                bootstrap_r2.append(r2_score(bs_y, bs_pred))
                bootstrap_rmse.append(np.sqrt(mean_squared_error(bs_y, bs_pred)))
                bootstrap_mae.append(mean_absolute_error(bs_y, bs_pred))
            
            # Calculate confidence intervals
            ci_r2 = np.percentile(bootstrap_r2, [alpha/2*100, (1-alpha/2)*100])
            ci_rmse = np.percentile(bootstrap_rmse, [alpha/2*100, (1-alpha/2)*100])
            ci_mae = np.percentile(bootstrap_mae, [alpha/2*100, (1-alpha/2)*100])
            
            bootstrap_results[model_name] = {
                'bootstrap_r2': bootstrap_r2,
                'bootstrap_rmse': bootstrap_rmse,
                'bootstrap_mae': bootstrap_mae,
                'ci_r2': ci_r2,
                'ci_rmse': ci_rmse,
                'ci_mae': ci_mae
            }
            
            self.logger.info(f"Model {model_name}:")
            self.logger.info(f"  R²: {metrics[model_name]['r2']:.4f} 95% CI: [{ci_r2[0]:.4f}, {ci_r2[1]:.4f}]")
            self.logger.info(f"  RMSE: {metrics[model_name]['rmse']:.4f} 95% CI: [{ci_rmse[0]:.4f}, {ci_rmse[1]:.4f}]")
        
        # Compare all pairs of models
        model_names = list(predictions.keys())
        comparison_results = {}
        
        for i, model1 in enumerate(model_names):
            for j, model2 in enumerate(model_names):
                if i >= j:  # Only compare each pair once
                    continue
                
                # Calculate difference in R² for each bootstrap sample
                diff_r2 = np.array(bootstrap_results[model1]['bootstrap_r2']) - np.array(bootstrap_results[model2]['bootstrap_r2'])
                ci_diff_r2 = np.percentile(diff_r2, [alpha/2*100, (1-alpha/2)*100])
                
                # Calculate difference in RMSE for each bootstrap sample
                diff_rmse = np.array(bootstrap_results[model1]['bootstrap_rmse']) - np.array(bootstrap_results[model2]['bootstrap_rmse'])
                ci_diff_rmse = np.percentile(diff_rmse, [alpha/2*100, (1-alpha/2)*100])
                
                # Determine if difference is significant
                sig_r2 = not (ci_diff_r2[0] <= 0 <= ci_diff_r2[1])
                sig_rmse = not (ci_diff_rmse[0] <= 0 <= ci_diff_rmse[1])
                
                # Store comparison results
                comparison_key = f"{model1} vs {model2}"
                comparison_results[comparison_key] = {
                    'diff_r2': metrics[model1]['r2'] - metrics[model2]['r2'],
                    'ci_diff_r2': ci_diff_r2,
                    'significant_r2': sig_r2,
                    'diff_rmse': metrics[model1]['rmse'] - metrics[model2]['rmse'],
                    'ci_diff_rmse': ci_diff_rmse,
                    'significant_rmse': sig_rmse
                }
                
                self.logger.info(f"Comparison: {model1} vs {model2}")
                self.logger.info(f"  R² difference: {metrics[model1]['r2'] - metrics[model2]['r2']:.4f} 95% CI: [{ci_diff_r2[0]:.4f}, {ci_diff_r2[1]:.4f}]")
                self.logger.info(f"  RMSE difference: {metrics[model1]['rmse'] - metrics[model2]['rmse']:.4f} 95% CI: [{ci_diff_rmse[0]:.4f}, {ci_diff_rmse[1]:.4f}]")
                self.logger.info(f"  R² difference is {'significant' if sig_r2 else 'not significant'}")
                self.logger.info(f"  RMSE difference is {'significant' if sig_rmse else 'not significant'}")
        
        # Create overall results
        significance_results = {
            'metrics': metrics,
            'bootstrap_results': bootstrap_results,
            'comparison_results': comparison_results,
            'alpha': alpha,
            'n_bootstrap': n_bootstrap
        }
        
        # Create visualization
        if self.output_dir:
            self._visualize_model_comparison(significance_results)
        
        # Store results
        self.results['significance'] = significance_results
        
        return significance_results
    
    def _visualize_lopo_results(self, lopo_results: Dict[str, Any]) -> None:
        """
        Visualize leave-one-pilot-out validation results.
        
        Args:
            lopo_results: Dictionary with LOPO results
        """
        if not self.output_dir:
            return
            
        # Create first visualization: R² for each pilot
        plt.figure(figsize=(12, 8))
        
        pilot_results = lopo_results['pilot_results']
        r2_values = []
        pilot_ids = []
        categories = []
        
        for pilot, result in pilot_results.items():
            r2_values.append(result['r2'])
            pilot_ids.append(pilot)
            categories.append(result.get('category', 'unknown'))
        
        # Sort by R² value
        sorted_idx = np.argsort(r2_values)
        r2_values = [r2_values[i] for i in sorted_idx]
        pilot_ids = [pilot_ids[i] for i in sorted_idx]
        categories = [categories[i] for i in sorted_idx]
        
        # Create color map based on category if available
        if 'unknown' not in categories:
            category_map = {'minimal_exp': 'red', 'commercial': 'green', 'air_force': 'blue'}
            colors = [category_map.get(cat, 'gray') for cat in categories]
            
            # Create legend handles
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Minimal Exp'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Commercial'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Air Force')
            ]
            
            plt.legend(handles=legend_elements, loc='best')
        else:
            colors = 'steelblue'
        
        # Create bar chart
        bars = plt.bar(range(len(pilot_ids)), r2_values, color=colors)
        
        # Add r² = 0 line
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.7)
        
        # Add labels
        plt.xlabel('Pilot', fontsize=12)
        plt.ylabel('R² Score', fontsize=12)
        plt.title('Leave-One-Pilot-Out Validation: R² Score by Pilot', fontsize=14)
        
        # Add pilot ids as tick labels if not too many
        if len(pilot_ids) <= 20:
            plt.xticks(range(len(pilot_ids)), pilot_ids, rotation=45, ha='right')
        else:
            plt.xticks([])
        
        # Add overall mean
        plt.axhline(y=lopo_results['mean_r2'], color='k', linestyle='-', alpha=0.7)
        plt.text(len(pilot_ids)-1, lopo_results['mean_r2'], f' Mean R²: {lopo_results["mean_r2"]:.4f}', 
                va='center', ha='right', backgroundcolor='white', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'lopo_r2_by_pilot.png'), dpi=300)
        plt.close()
        
        # Create second visualization: R² by category if available
        if 'category_stats' in lopo_results:
            plt.figure(figsize=(10, 6))
            
            category_stats = lopo_results['category_stats']
            categories = list(category_stats.keys())
            mean_r2 = [category_stats[cat]['mean_r2'] for cat in categories]
            std_r2 = [category_stats[cat]['std_r2'] for cat in categories]
            
            # Create bar chart with error bars
            plt.bar(categories, mean_r2, yerr=std_r2, capsize=10, alpha=0.7)
            
            # Add labels
            plt.xlabel('Pilot Category', fontsize=12)
            plt.ylabel('Mean R² Score', fontsize=12)
            plt.title('Leave-One-Pilot-Out Validation: Mean R² by Pilot Category', fontsize=14)
            
            # Add r² = 0 line
            plt.axhline(y=0, color='r', linestyle='--', alpha=0.7)
            
            # Add text with number of pilots
            for i, cat in enumerate(categories):
                n_pilots = category_stats[cat]['n_pilots']
                plt.text(i, mean_r2[i] + std_r2[i] + 0.05, f'n={n_pilots}', ha='center')
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'lopo_r2_by_category.png'), dpi=300)
            plt.close()
    
    def _visualize_learning_curves(self, train_sizes, train_scores_mean, train_scores_std,
                                test_scores_mean, test_scores_std) -> None:
        """
        Visualize learning curves.
        
        Args:
            train_sizes: Array of training set sizes
            train_scores_mean: Array of mean training scores
            train_scores_std: Array of std training scores
            test_scores_mean: Array of mean test scores
            test_scores_std: Array of std test scores
        """
        if not self.output_dir:
            return
            
        plt.figure(figsize=(10, 6))
        
        # Plot learning curves
        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                        train_scores_mean + train_scores_std, alpha=0.1, color='b')
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                        test_scores_mean + test_scores_std, alpha=0.1, color='r')
        plt.plot(train_sizes, train_scores_mean, 'o-', color='b', label='Training score')
        plt.plot(train_sizes, test_scores_mean, 'o-', color='r', label='Cross-validation score')
        
        # Add labels
        plt.xlabel('Training Samples', fontsize=12)
        plt.ylabel('R² Score', fontsize=12)
        plt.title('Learning Curves (Gradient Boosting Regressor)', fontsize=14)
        plt.legend(loc='best')
        
        # Add gridlines
        plt.grid(True, alpha=0.3)
        
        # Add annotation for the gap
        gap = train_scores_mean[-1] - test_scores_mean[-1]
        plt.annotate(f'Gap: {gap:.4f}', xy=(train_sizes[-1], train_scores_mean[-1] - gap/2),
                   xytext=(train_sizes[-1] - 0.2, train_scores_mean[-1] - gap/2 + 0.05),
                   arrowprops=dict(arrowstyle='->'))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'learning_curves.png'), dpi=300)
        plt.close()
        
        # Create second visualization: train vs test curve
        plt.figure(figsize=(10, 6))
        
        # Plot train vs test curve
        plt.plot([0, 1], [0, 1], 'k--')
        plt.scatter(train_scores_mean, test_scores_mean, s=30, c=plt.cm.viridis(np.linspace(0, 1, len(train_sizes))))
        
        # Connect points
        plt.plot(train_scores_mean, test_scores_mean, '-')
        
        # Add color bar to show training size
        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=min(train_sizes), vmax=max(train_sizes)))
        sm.set_array([])
        cbar = plt.colorbar(sm)
        cbar.set_label('Training Size Fraction')
        
        # Add labels
        plt.xlabel('Training Score', fontsize=12)
        plt.ylabel('Cross-Validation Score', fontsize=12)
        plt.title('Training vs Cross-Validation Performance', fontsize=14)
        
        # Add gridlines
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'train_vs_test_curve.png'), dpi=300)
        plt.close()
    
    def _visualize_permutation_test(self, permutation_results: Dict[str, Any]) -> None:
        """
        Visualize permutation test results.
        
        Args:
            permutation_results: Dictionary with permutation test results
        """
        if not self.output_dir:
            return
            
        plt.figure(figsize=(12, 5))
        
        # Get results
        true_r2 = permutation_results['true_r2']
        permutation_r2 = permutation_results['permutation_r2']
        p_value_r2 = permutation_results['p_value_r2']
        
        # Plot R² distribution
        plt.subplot(1, 2, 1)
        plt.hist(permutation_r2, bins=30, alpha=0.7, color='skyblue')
        plt.axvline(x=true_r2, color='red', linestyle='--', linewidth=2)
        
        # Add labels
        plt.xlabel('R² Score', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title(f'Permutation Test: R² (p={p_value_r2:.4f})', fontsize=14)
        
        # Add text box with true R²
        plt.text(0.05, 0.95, f'True R²: {true_r2:.4f}', transform=plt.gca().transAxes,
                fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.1))
        
        # Plot RMSE distribution
        plt.subplot(1, 2, 2)
        true_rmse = permutation_results['true_rmse']
        permutation_rmse = permutation_results['permutation_rmse']
        p_value_rmse = permutation_results['p_value_rmse']
        
        plt.hist(permutation_rmse, bins=30, alpha=0.7, color='salmon')
        plt.axvline(x=true_rmse, color='red', linestyle='--', linewidth=2)
        
        # Add labels
        plt.xlabel('RMSE', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title(f'Permutation Test: RMSE (p={p_value_rmse:.4f})', fontsize=14)
        
        # Add text box with true RMSE
        plt.text(0.05, 0.95, f'True RMSE: {true_rmse:.4f}', transform=plt.gca().transAxes,
                fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.1))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'permutation_test.png'), dpi=300)
        plt.close()
    
    def _visualize_model_comparison(self, significance_results: Dict[str, Any]) -> None:
        """
        Visualize statistical model comparison.
        
        Args:
            significance_results: Dictionary with significance test results
        """
        if not self.output_dir:
            return
            
        # Get results
        metrics = significance_results['metrics']
        bootstrap_results = significance_results['bootstrap_results']
        
        # Create first visualization: R² comparison
        plt.figure(figsize=(12, 6))
        
        # Collect model names and R² values
        model_names = list(metrics.keys())
        r2_values = [metrics[model]['r2'] for model in model_names]
        lower_ci = [bootstrap_results[model]['ci_r2'][0] for model in model_names]
        upper_ci = [bootstrap_results[model]['ci_r2'][1] for model in model_names]
        
        # Calculate error bars
        yerr = np.vstack([r2_values - lower_ci, upper_ci - r2_values])
        
        # Sort by R² value
        sorted_idx = np.argsort(r2_values)
        model_names = [model_names[i] for i in sorted_idx]
        r2_values = [r2_values[i] for i in sorted_idx]
        yerr = yerr[:, sorted_idx]
        
        # Create bar chart
        plt.bar(model_names, r2_values, yerr=yerr, capsize=10, alpha=0.7)
        
        # Add labels
        plt.xlabel('Model', fontsize=12)
        plt.ylabel('R² Score with 95% CI', fontsize=12)
        plt.title('Model Performance Comparison: R²', fontsize=14)
        
        # Rotate x-axis labels if needed
        if len(model_names) > 4:
            plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'model_comparison_r2.png'), dpi=300)
        plt.close()
        
        # Create second visualization: R² comparison via bootstrap distributions
        if len(model_names) <= 5:  # Only create if not too many models
            plt.figure(figsize=(12, 6))
            
            # Plot bootstrap distributions
            for model in model_names:
                sns.kdeplot(bootstrap_results[model]['bootstrap_r2'], label=model)
            
            # Add labels
            plt.xlabel('R² Score', fontsize=12)
            plt.ylabel('Density', fontsize=12)
            plt.title('Bootstrap Distribution of R² Scores', fontsize=14)
            
            # Add legend
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'model_comparison_bootstrap.png'), dpi=300)
            plt.close()
    
    def get_results(self) -> Dict[str, Any]:
        """
        Get all validation results.
        
        Returns:
            Dictionary with all validation results
        """
        return self.results
        
    def save_results(self) -> str:
        """
        Save validation results to a file.
        
        Returns:
            Path to the saved file
        """
        if not self.output_dir:
            self.logger.warning("No output directory specified, skipping saving validation results")
            return None
            
        # Create a pickle file
        results_path = os.path.join(self.output_dir, 'validation_results.joblib')
        joblib.dump(self.results, results_path)
        
        self.logger.info(f"Validation results saved to: {results_path}")
        
        return results_path
