import os
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Tuple, Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.feature_selection import SelectFromModel, RFECV, mutual_info_regression
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
import warnings


class FeatureSelector:
    """
    Selects the most relevant features for cognitive load prediction
    using multiple selection methods and ensemble ranking.
    """
    
    def __init__(self, method: str = 'combined',
                 n_features: int = 40,
                 use_shap: bool = True,
                 target_column: str = 'avg_tlx_quantile',
                 id_column: str = 'pilot_id',
                 seed: int = 42,
                 output_dir: str = None):
        """
        Initialize the FeatureSelector object.
        
        Args:
            method: Feature selection method ('combined', 'rf', 'gb', 'correlation', 'permutation', 'rfe')
            n_features: Maximum number of features to select
            use_shap: Whether to include SHAP values in the feature selection process
            target_column: Column name for the target variable
            id_column: Column name for subject identifier
            seed: Random seed for reproducibility
            output_dir: Directory to save output files
        """
        self.method = method
        self.n_features = n_features
        self.use_shap = use_shap
        self.target_column = target_column
        self.id_column = id_column
        self.seed = seed
        self.output_dir = output_dir
        
        # Store the selected features
        self.selected_features = None
        
        # Store feature importances from different methods
        self.feature_importance = {}
        
        # Store feature sets
        self.feature_sets = {}
        
        # Logger
        self.logger = logging.getLogger(__name__)
        
        # Set random seed
        np.random.seed(self.seed)
        
        # Check if SHAP is available
        if use_shap:
            try:
                import shap
                self.shap_available = True
            except ImportError:
                self.logger.warning("SHAP not available, will not use SHAP values for feature selection")
                self.shap_available = False
        else:
            self.shap_available = False
            
    def select_features(self, data: pd.DataFrame, sample_weight: Optional[np.ndarray] = None) -> List[str]:
        """
        Perform feature selection using the specified method.
        
        Args:
            data: DataFrame with features and target
            sample_weight: Optional sample weights for importance calculation
            
        Returns:
            List of selected feature names
        """
        self.logger.info(f"\nPerforming feature selection using {self.method} method...")
        
        # Check if target column exists
        if self.target_column not in data.columns:
            self.logger.error(f"Target column '{self.target_column}' not found in data")
            raise ValueError(f"Target column '{self.target_column}' not found in data")
        
        # Define columns to exclude from feature selection
        exclude_cols = [self.id_column, self.target_column]
        if 'pilot_category' in data.columns:
            exclude_cols.append('pilot_category')
        
        # Get numerical columns
        num_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        base_feats = [col for col in num_cols if col not in exclude_cols]
        
        # Check if we have enough features to select from
        if len(base_feats) <= self.n_features:
            self.logger.warning(f"Only {len(base_feats)} features available, which is less than or equal to the requested {self.n_features} features. Using all available features.")
            self.selected_features = base_feats
            return base_feats
        
        # Target variable
        y = data[self.target_column]
        # X = data[base_feats]
        X = data[base_feats].fillna(0)
        
        # Apply different feature selection methods
        results = {}
        
        if self.method in ['combined', 'correlation']:
            # Correlation analysis
            self.logger.info("Method 1: Correlation Analysis")
            corr_list = []
            for feat in base_feats:
                corr = abs(data[feat].corr(y))
                corr_list.append({'feature': feat, 'importance': corr})
            corr_df = pd.DataFrame(corr_list).sort_values('importance', ascending=False)
            results['correlation'] = corr_df
            self.feature_importance['correlation'] = corr_df
            
        if self.method in ['combined', 'spearman']:
            # Spearman correlation (handles non-linear relationships better)
            self.logger.info("Method 2: Spearman Correlation Analysis")
            spearman_list = []
            for feat in base_feats:
                corr, _ = spearmanr(data[feat], y)
                spearman_list.append({'feature': feat, 'importance': abs(corr)})
            spearman_df = pd.DataFrame(spearman_list).sort_values('importance', ascending=False)
            results['spearman'] = spearman_df
            self.feature_importance['spearman'] = spearman_df
            
        if self.method in ['combined', 'mutual_info']:
            # Mutual information (captures any statistical dependency)
            self.logger.info("Method 3: Mutual Information Analysis")
            try:
                # Scale features for MI calculation
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                mi = mutual_info_regression(X_scaled, y)
                mi_df = pd.DataFrame({'feature': base_feats, 'importance': mi})
                mi_df = mi_df.sort_values('importance', ascending=False)
                results['mutual_info'] = mi_df
                self.feature_importance['mutual_info'] = mi_df
            except Exception as e:
                self.logger.warning(f"Error in mutual information calculation: {str(e)}")
        
        if self.method in ['combined', 'rf']:
            # Random Forest importance
            self.logger.info("Method 4: Random Forest Importance")
            rf = RandomForestRegressor(n_estimators=100, random_state=self.seed)
            rf.fit(X, y, sample_weight=sample_weight)
            rf_imp = pd.DataFrame({'feature': base_feats, 'importance': rf.feature_importances_})
            rf_imp = rf_imp.sort_values('importance', ascending=False)
            results['rf'] = rf_imp
            self.feature_importance['rf'] = rf_imp
        
        if self.method in ['combined', 'gb']:
            # Gradient Boosting importance
            self.logger.info("Method 5: Gradient Boosting Importance")
            gb = GradientBoostingRegressor(n_estimators=100, random_state=self.seed)
            gb.fit(X, y, sample_weight=sample_weight)
            gb_imp = pd.DataFrame({'feature': base_feats, 'importance': gb.feature_importances_})
            gb_imp = gb_imp.sort_values('importance', ascending=False)
            results['gb'] = gb_imp
            self.feature_importance['gb'] = gb_imp
        
        if self.method in ['combined', 'permutation'] and len(X) >= 50:
            # Permutation importance (more reliable but slower)
            self.logger.info("Method 6: Permutation Importance")
            try:
                rf_model = RandomForestRegressor(n_estimators=50, random_state=self.seed)
                rf_model.fit(X, y, sample_weight=sample_weight)
                perm = permutation_importance(rf_model, X, y, n_repeats=5, random_state=self.seed)
                perm_df = pd.DataFrame({'feature': base_feats, 'importance': perm.importances_mean})
                perm_df = perm_df.sort_values('importance', ascending=False)
                results['permutation'] = perm_df
                self.feature_importance['permutation'] = perm_df
            except Exception as e:
                self.logger.warning(f"Error in permutation importance calculation: {str(e)}")
        
        if self.shap_available and self.method in ['combined', 'shap']:
            # SHAP importance (best for interpretability)
            self.logger.info("Method 7: SHAP Importance")
            try:
                import shap
                # Train a model for SHAP analysis
                model = GradientBoostingRegressor(n_estimators=50, random_state=self.seed)
                model.fit(X, y, sample_weight=sample_weight)
                
                # Create a SHAP explainer
                explainer = shap.TreeExplainer(model)
                
                # Calculate SHAP values (use a sample if dataset is large)
                if len(X) > 500:
                    sample_idx = np.random.choice(len(X), 500, replace=False)
                    X_sample = X.iloc[sample_idx]
                    shap_values = explainer.shap_values(X_sample)
                else:
                    shap_values = explainer.shap_values(X)
                
                # Calculate mean absolute SHAP value for each feature
                shap_importances = np.abs(shap_values).mean(axis=0)
                shap_df = pd.DataFrame({'feature': base_feats, 'importance': shap_importances})
                shap_df = shap_df.sort_values('importance', ascending=False)
                results['shap'] = shap_df
                self.feature_importance['shap'] = shap_df
                
                # Create SHAP summary plot (if output directory specified)
                if self.output_dir:
                    # Use a smaller sample for plotting
                    if len(X) > 100:
                        sample_idx = np.random.choice(len(X), 100, replace=False)
                        X_sample = X.iloc[sample_idx]
                        shap_values_sample = explainer.shap_values(X_sample)
                    else:
                        X_sample = X
                        shap_values_sample = shap_values
                    
                    plt.figure(figsize=(10, 8))
                    shap.summary_plot(shap_values_sample, X_sample, show=False, max_display=20)
                    plt.tight_layout()
                    plt.savefig(os.path.join(self.output_dir, 'shap_summary_plot.png'), dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    # SHAP dependence plots for top features
                    top_features = shap_df['feature'].head(3).tolist()
                    for feature in top_features:
                        plt.figure(figsize=(10, 6))
                        feature_idx = list(X_sample.columns).index(feature)
                        shap.dependence_plot(feature_idx, shap_values_sample, X_sample, show=False)
                        plt.tight_layout()
                        plt.savefig(os.path.join(self.output_dir, f'shap_dependence_{feature}.png'), dpi=300)
                        plt.close()
            except Exception as e:
                self.logger.warning(f"Error in SHAP importance calculation: {str(e)}")
        
        if self.method in ['combined', 'rfe'] and len(X) >= 50:
            # Recursive Feature Elimination
            self.logger.info("Method 8: Recursive Feature Elimination")
            try:
                # Use a smaller subset for RFE if dataset is large
                if len(X) > 500:
                    sample_idx = np.random.choice(len(X), 500, replace=False)
                    X_sample = X.iloc[sample_idx]
                    y_sample = y.iloc[sample_idx]
                    sw_sample = sample_weight[sample_idx] if sample_weight is not None else None
                else:
                    X_sample = X
                    y_sample = y
                    sw_sample = sample_weight
                
                model = RandomForestRegressor(n_estimators=50, random_state=self.seed)
                
                # Define RFE with cross-validation
                rfe = RFECV(estimator=model, step=1, cv=5, scoring='neg_mean_squared_error', 
                           min_features_to_select=min(10, len(base_feats)))
                
                # Fit RFE
                rfe.fit(X_sample, y_sample, sample_weight=sw_sample)
                
                # Create feature ranking
                rfe_df = pd.DataFrame({
                    'feature': base_feats,
                    'selected': rfe.support_,
                    'rank': rfe.ranking_
                })
                rfe_df = rfe_df.sort_values('rank')
                
                # Convert to standard format
                rfe_importance = pd.DataFrame({
                    'feature': base_feats,
                    'importance': 1 / (rfe_df['rank'] + 0.1)  # Convert rank to importance (higher = better)
                }).sort_values('importance', ascending=False)
                
                results['rfe'] = rfe_importance
                self.feature_importance['rfe'] = rfe_importance
            except Exception as e:
                self.logger.warning(f"Error in RFE calculation: {str(e)}")
        
        # Combine feature rankings if using the combined method
        if self.method == 'combined':
            # Combine rankings from different methods
            self.logger.info("Combining rankings from all methods")
            
            # Initialize ranks
            ranks = {feat: 0 for feat in base_feats}
            feature_counts = {feat: 0 for feat in base_feats}
            
            # Aggregate ranks from different methods
            for method_name, importance_df in results.items():
                if importance_df is not None:
                    for i, row in importance_df.iterrows():
                        feature = row['feature']
                        # Add rank (lower is better)
                        ranks[feature] += i
                        feature_counts[feature] += 1
            
            # Calculate average rank
            avg_ranks = {feat: ranks[feat] / max(1, feature_counts[feat]) for feat in base_feats}
            
            # Create combined ranking dataframe
            rank_df = pd.DataFrame({'feature': list(avg_ranks.keys()), 'rank': list(avg_ranks.values())})
            rank_df = rank_df.sort_values('rank')
            
            # Calculate importance (inverse of rank)
            rank_df['importance'] = 1 / (rank_df['rank'] + 0.1)
            
            # Get top features (by importance)
            if self.n_features < len(rank_df):
                selected = rank_df['feature'].head(self.n_features).tolist()
            else:
                selected = rank_df['feature'].tolist()
            
            # Save feature importance for reporting
            self.feature_importance['combined'] = rank_df[['feature', 'importance']].sort_values('importance', ascending=False)
        else:
            # Use the single specified method
            importance_df = results.get(self.method)
            if importance_df is None:
                self.logger.error(f"Method '{self.method}' not implemented or failed")
                raise ValueError(f"Method '{self.method}' not implemented or failed")
                
            # Get top features
            if self.n_features < len(importance_df):
                selected = importance_df['feature'].head(self.n_features).tolist()
            else:
                selected = importance_df['feature'].tolist()
            
            # Store feature importance
            self.feature_importance[self.method] = importance_df
        
        self.logger.info(f"Selected {len(selected)} features")
        if len(selected) >= 5:
            self.logger.info(f"Top 5 features: {selected[:5]}")
        
        # Save selected features to a file
        if self.output_dir:
            features_path = os.path.join(self.output_dir, 'selected_features.txt')
            with open(features_path, 'w') as f:
                for feat in selected:
                    f.write(f"{feat}\n")
            self.logger.info(f"Selected features saved to: {features_path}")
            
            # Visualize feature importance
            self._visualize_feature_importance()
        
        # Store selected features
        self.selected_features = selected
        
        return selected
    
    def create_feature_sets(self, data: pd.DataFrame, selected_features: List[str]) -> Dict[str, List[str]]:
        """
        Create different feature sets for model comparison and ablation study.
        
        Args:
            data: DataFrame with features
            selected_features: List of selected feature names
            
        Returns:
            Dictionary mapping set names to feature lists
        """
        self.logger.info("\nCreating feature sets for ablation study...")
        
        # Check if features exist in the data
        missing_features = [f for f in selected_features if f not in data.columns]
        if missing_features:
            self.logger.warning(f"Some selected features are not in the data: {missing_features}")
            selected_features = [f for f in selected_features if f in data.columns]
        
        # Create various feature sets
        feature_sets = {
            'all': selected_features,
            'top_5': selected_features[:min(5, len(selected_features))],
            'top_10': selected_features[:min(10, len(selected_features))],
            'top_20': selected_features[:min(20, len(selected_features))]
        }
        
        # Create optimal feature set (top features but limited to avoid overfitting)
        # Optimal is defined as either top 20 or half of all features, whichever is smaller
        optimal_size = min(20, len(selected_features) // 2)
        feature_sets['optimal'] = selected_features[:optimal_size]
        
        # Identify feature types for ablation
        turb_features = [f for f in selected_features if 'turb' in f]
        if len(turb_features) > 0:
            feature_sets['no_turbulence'] = [f for f in selected_features if f not in turb_features]
            feature_sets['turbulence_only'] = turb_features
        
        pilot_norm_features = [f for f in selected_features if 'pilot_norm' in f or 'pilot_minmax' in f]
        if len(pilot_norm_features) > 0:
            feature_sets['no_pilot_norm'] = [f for f in selected_features if f not in pilot_norm_features]
        
        physio_features = [f for f in selected_features 
                         if any(p in f for p in ['scr', 'hr', 'pnn', 'sdrr', 'temp', 'accel', 'eda', 'ibi'])]
        if len(physio_features) > 0:
            feature_sets['no_physio'] = [f for f in selected_features if f not in physio_features]
            feature_sets['physiological_only'] = physio_features
        
        # Report on feature sets
        for name, feats in feature_sets.items():
            self.logger.info(f"Feature set '{name}': {len(feats)} features")
        
        # Store feature sets
        self.feature_sets = feature_sets
        
        return feature_sets
    
    def _visualize_feature_importance(self) -> None:
        """Create visualization of feature importance."""
        if not self.feature_importance or not self.output_dir:
            return
            
        # Visualize the primary importance source
        primary_method = self.method if self.method != 'combined' else 'combined'
        if primary_method in self.feature_importance:
            importance_df = self.feature_importance[primary_method]
            
            # Limit to top features for visualization
            top_n = min(20, len(importance_df))
            
            if 'importance' in importance_df.columns:
                # Sort by importance
                top_df = importance_df.sort_values('importance', ascending=False).head(top_n)
                values = top_df['importance']
                title = f'{primary_method.title()} Feature Importance'
                value_label = 'Importance'
            elif 'rank' in importance_df.columns:
                # Sort by rank (lower is better)
                top_df = importance_df.sort_values('rank').head(top_n)
                values = -top_df['rank']  # Negative because lower rank is better
                title = f'{primary_method.title()} Feature Ranking (Lower is Better)'
                value_label = 'Rank'
            else:
                self.logger.warning("Importance DataFrame must have 'importance' or 'rank' column")
                return
            
            # Create visualization
            plt.figure(figsize=(12, 8))
            
            # Plot horizontal bar chart
            features = top_df['feature'].values
            y_pos = np.arange(len(features))
            
            # Format feature names for readability
            display_features = [f.replace('_', ' ').title() for f in features]
            
            bars = plt.barh(y_pos, values, align='center', alpha=0.8, 
                            color=plt.cm.viridis(np.linspace(0, 1, len(features))))
            plt.yticks(y_pos, display_features)
            plt.xlabel(value_label)
            plt.title(title)
            
            # Add values to the end of each bar
            for i, v in enumerate(values):
                plt.text(v, i, f' {v:.4f}', va='center')
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f'feature_importance_{primary_method}.png'), dpi=300)
            plt.close()
            
            # Create comparison plot if we have multiple methods
            if len(self.feature_importance) > 1:
                self._create_importance_comparison()

    def _create_importance_comparison(self) -> None:
        """
        Create visualization comparing importance from different methods.
        """
        if not self.output_dir or len(self.feature_importance) <= 1:
            return
            
        # Create figure with multiple importance methods
        plt.figure(figsize=(14, 10))
        
        # Get top features from each method (union)
        top_n = 15
        all_top_features = set()
        
        for method, importance_df in self.feature_importance.items():
            top_features = importance_df['feature'].head(top_n).tolist()
            all_top_features.update(top_features)
        
        # Sort features by combined importance
        feature_scores = {feature: 0 for feature in all_top_features}
        
        for method, importance_df in self.feature_importance.items():
            for _, row in importance_df.iterrows():
                feature = row['feature']
                if feature in feature_scores:
                    # Normalize importance by max importance in this method
                    max_importance = importance_df['importance'].max()
                    feature_scores[feature] += row['importance'] / max_importance
        
        # Sort features by combined score
        sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        features = [f[0] for f in sorted_features]
        
        # Plot importance for each method
        x = np.arange(len(features))
        width = 0.8 / len(self.feature_importance)
        
        for i, (method, importance_df) in enumerate(self.feature_importance.items()):
            # Get importance values for the selected features
            imp_values = []
            for feature in features:
                try:
                    value = importance_df[importance_df['feature'] == feature]['importance'].values[0]
                    imp_values.append(value)
                except (IndexError, KeyError):
                    imp_values.append(0)
            
            # Normalize importance values
            max_imp = max(imp_values) if max(imp_values) > 0 else 1
            norm_imp = [v / max_imp for v in imp_values]
            
            # Plot bar for this method
            plt.bar(x + (i - len(self.feature_importance)/2 + 0.5) * width, 
                   norm_imp, width, label=method.title())
        
        # Add labels and legend
        plt.xlabel('Features', fontsize=12)
        plt.ylabel('Normalized Importance', fontsize=12)
        plt.title('Feature Importance Comparison Across Methods', fontsize=14)
        plt.xticks(x, features, rotation=45, ha='right')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'feature_importance_comparison.png'), dpi=300)
        plt.close()
