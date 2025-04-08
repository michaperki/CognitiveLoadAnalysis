import os
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Tuple, Optional, Union
from scipy import stats
from scipy.signal import welch, find_peaks, butter, filtfilt
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import acf, pacf

import warnings
from pandas.errors import PerformanceWarning
warnings.simplefilter("ignore", PerformanceWarning)


class FeatureEngineer:
    """
    Creates advanced features from physiological and pilot data for cognitive load prediction.
    Enhanced version with robust feature tracking and visualization.
    """
    
    def __init__(self, id_column: str = 'pilot_id', 
                 target_column: str = 'avg_tlx_quantile',
                 seed: int = 42,
                 output_dir: str = None,
                 config: Dict[str, bool] = None):
        """
        Initialize the FeatureEngineer object.
        
        Args:
            id_column: Column name for subject identifier
            target_column: Column name for the target variable
            seed: Random seed for reproducibility
            output_dir: Directory to save output files
            config: Configuration for feature engineering
        """
        self.id_column = id_column
        self.target_column = target_column
        self.seed = seed
        self.output_dir = output_dir
        self.config = config or {
            'core_features': True,
            'turbulence_interaction': True,
            'pilot_normalized': True,
            'polynomial_features': True,
            'signal_derivatives': False,
            'signal_ratios': True,
            'experience_features': True,
            'frequency_domain': False
        }
        
        # Statistics
        self.feature_engineering_stats = {}
        self.feature_importance = {}
        self.feature_tracking = {}
        
        # Logger
        self.logger = logging.getLogger(__name__)
        
        # Set random seed
        np.random.seed(self.seed)
    
    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply comprehensive feature engineering to the dataset.
        
        Args:
            data: Preprocessed DataFrame
            
        Returns:
            DataFrame with additional engineered features
        """
        # Check if data is empty
        if data is None or len(data) == 0:
            self.logger.error("No data provided for feature engineering")
            raise ValueError("Empty data provided for feature engineering")
        
        self.logger.info("\nApplying feature engineering...")
        
        # Store original number of features
        original_features = len(data.columns)
        original_feature_list = data.columns.tolist()
        
        # Create a working copy
        df = data.copy()
        
        # Apply each feature engineering method based on config
        if self.config.get('core_features', True):
            df = self._create_core_features(df)
        
        if self.config.get('turbulence_interaction', True):
            df = self._create_turbulence_interaction_features(df)
        
        if self.config.get('pilot_normalized', True):
            df = self._create_pilot_normalized_features(df)
        
        if self.config.get('polynomial_features', True):
            df = self._create_polynomial_features(df)
        
        if self.config.get('signal_derivatives', False):
            df = self._create_signal_derivatives(df)
        
        if self.config.get('signal_ratios', True):
            df = self._create_signal_ratios(df)
        
        if self.config.get('experience_features', True):
            df = self._create_experience_features(df)
        
        if self.config.get('frequency_domain', False):
            df = self._create_frequency_domain_features(df)
        
        # Report new features
        all_features = df.columns.tolist()
        new_features = [f for f in all_features if f not in original_feature_list]
        self.logger.info(f"Feature engineering complete: added {len(new_features)} new features")
        
        # Log new feature count by category
        self.logger.info(f"Total features: {len(df.columns)}")
        
        # Summarize created features by category
        category_summary = "\nFeature engineering summary:"
        for category, count in self.feature_engineering_stats.items():
            category_summary += f"\n  {category.replace('_', ' ').title()}: {count} features"
        self.logger.info(category_summary)
        
        # Analyze new features
        self._analyze_engineered_features(df, new_features)
        
        # Save feature list
        if self.output_dir:
            new_features_path = os.path.join(self.output_dir, 'engineered_features.txt')
            with open(new_features_path, 'w') as f:
                for feature in new_features:
                    f.write(f"{feature}\n")
            
            # Save feature engineering stats
            stats_path = os.path.join(self.output_dir, 'feature_engineering_stats.json')
            import json
            with open(stats_path, 'w') as f:
                json.dump(self.feature_engineering_stats, f, indent=2)
            
        return df
    
    def _create_core_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create core features that were important in previous analyses.
        
        Args:
            data: Input DataFrame
            
        Returns:
            DataFrame with additional features
        """
        df = data.copy()
        
        # Create z-score normalized features
        norm_features = []
        for col in ['avg_tlx', 'mental_effort']:
            if col in df.columns:
                # Z-score normalization
                norm = f"{col}_zscore"
                df[norm] = df.groupby(self.id_column)[col].transform(
                    lambda x: (x - x.mean()) / (x.std() if x.std() > 0 else 1)
                )
                norm_features.append(norm)
                
                # Calculate percentile rank
                rank = f"{col}_rank"
                df[rank] = df.groupby(self.id_column)[col].transform(
                    lambda x: x.rank(pct=True)
                )
                norm_features.append(rank)
        
        self.feature_engineering_stats['core'] = len(norm_features)
        self.logger.info(f"Created {len(norm_features)} core features")
        
        # Track which features were created in this step
        self.feature_tracking['core_features'] = norm_features
        
        return df
    
    def _create_turbulence_interaction_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between physiological signals and turbulence.
        
        Args:
            data: Input DataFrame
            
        Returns:
            DataFrame with additional features
        """
        df = data.copy()
        
        if 'turbulence' not in df.columns:
            self.logger.warning("Turbulence column not found, skipping turbulence interaction features")
            self.feature_engineering_stats['turbulence_interaction'] = 0
            self.feature_tracking['turbulence_interaction'] = []
            return df
            
        # Identify physiological signals
        physio_signals = [col for col in df.columns 
                         if any(col.startswith(s) for s in ['scr_', 'hr_', 'sdrr', 'pnn50', 'temp_', 'accel_', 'ibi_', 'raw_eda_'])]
        
        # Filter out any features that are already interactions
        physio_signals = [col for col in physio_signals 
                         if '_turb_' not in col and not any(suffix in col for suffix in ['_norm', '_rank', '_pilot_'])]
        
        interaction_features = []
        for signal in physio_signals:
            # Linear interaction
            interact = f'{signal}_turb_interact'
            df[interact] = df[signal] * df['turbulence']
            interaction_features.append(interact)
            
            # Ratio interaction
            ratio = f'{signal}_turb_ratio'
            df[ratio] = df[signal] / (df['turbulence'] + 1)  # +1 to avoid division by zero
            interaction_features.append(ratio)
            
            # Scaled interaction (using z-score of turbulence)
            scaled_turb = (df['turbulence'] - df['turbulence'].mean()) / df['turbulence'].std()
            scaled_interact = f'{signal}_scaled_turb_interact'
            df[scaled_interact] = df[signal] * scaled_turb
            interaction_features.append(scaled_interact)
        
        self.feature_engineering_stats['turbulence_interaction'] = len(interaction_features)
        self.logger.info(f"Created {len(interaction_features)} turbulence interaction features")
        
        # Track which features were created in this step
        self.feature_tracking['turbulence_interaction'] = interaction_features
        
        return df
    
    def _create_pilot_normalized_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create normalized features based on pilot-specific statistics.
        
        Args:
            data: Input DataFrame
            
        Returns:
            DataFrame with additional features
        """
        df = data.copy()
        
        # Identify numerical physiological signals
        signals = [col for col in df.select_dtypes(include=[np.number]).columns
                 if any(col.startswith(s) for s in ['scr_', 'hr_', 'sdrr', 'pnn50', 'temp_', 'accel_', 'ibi_', 'raw_eda_'])]
        
        # Include turbulence interaction features if they exist
        turb_signals = [col for col in df.columns if ('_turb_interact' in col or '_turb_ratio' in col)]
        signals.extend(turb_signals)
        
        # Remove any features already normalized
        signals = [col for col in signals if not any(suffix in col for suffix in ['_norm', '_minmax', '_pilot_', '_rank'])]
        
        norm_features = []
        for col in signals:
            # Z-score normalization within pilot
            norm = f"{col}_pilot_norm"
            df[norm] = df.groupby(self.id_column)[col].transform(
                lambda x: (x - x.mean()) / (x.std() if x.std() > 0 else 1)
            )
            norm_features.append(norm)
            
            # Min-max normalization within pilot
            minmax = f"{col}_pilot_minmax"
            df[minmax] = df.groupby(self.id_column)[col].transform(
                lambda x: (x - x.min()) / ((x.max() - x.min()) if (x.max() - x.min()) > 0 else 1)
            )
            norm_features.append(minmax)
        
        self.feature_engineering_stats['pilot_normalized'] = len(norm_features)
        self.logger.info(f"Created {len(norm_features)} pilot-normalized features")
        
        # Track which features were created in this step
        self.feature_tracking['pilot_normalized'] = norm_features
        
        return df
    
    def _create_polynomial_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create polynomial and transformed features.
        
        Args:
            data: Input DataFrame
            
        Returns:
            DataFrame with additional features
        """
        df = data.copy()
        
        if 'turbulence' not in df.columns:
            self.logger.warning("Turbulence column not found, skipping polynomial features")
            self.feature_engineering_stats['polynomial'] = 0
            self.feature_tracking['polynomial'] = []
            return df
            
        # Turbulence transformations
        turb_transforms = []
        
        # Polynomial transformations
        df['turbulence_squared'] = df['turbulence'] ** 2
        turb_transforms.append('turbulence_squared')
        
        df['turbulence_cubed'] = df['turbulence'] ** 3
        turb_transforms.append('turbulence_cubed')
        
        # Logarithmic transformation
        df['turbulence_log'] = np.log1p(df['turbulence'])
        turb_transforms.append('turbulence_log')
        
        # Square root transformation
        df['turbulence_sqrt'] = np.sqrt(df['turbulence'])
        turb_transforms.append('turbulence_sqrt')
        
        # Physiological signals transformations
        poly_feats = []
        base_signals = ['scr_mean', 'hr_mean', 'sdrr', 'pnn50', 'scr_count', 'raw_eda_mean', 'temp_mean', 'accel_mean', 'ibi_mean']
        available_signals = [s for s in base_signals if s in df.columns]
        
        for feat in available_signals:
            # Create polynomial interactions with turbulence
            squared = f'{feat}_squared'
            df[squared] = df[feat] ** 2
            poly_feats.append(squared)
            
            cubed = f'{feat}_cubed'
            df[cubed] = df[feat] ** 3
            poly_feats.append(cubed)
            
            log = f'{feat}_log'
            df[log] = np.log1p(np.abs(df[feat]) + 1e-6)  # Add small constant to avoid log(0)
            poly_feats.append(log)
            
            # Create turbulence interactions with transformed features
            interact_sq = f'{feat}_turb_squared'
            df[interact_sq] = df[feat] * df['turbulence_squared']
            poly_feats.append(interact_sq)
        
        all_poly_feats = turb_transforms + poly_feats
        self.feature_engineering_stats['polynomial'] = len(all_poly_feats)
        self.logger.info(f"Created {len(all_poly_feats)} polynomial and transformed features")
        
        # Track which features were created in this step
        self.feature_tracking['polynomial'] = all_poly_feats
        
        return df
    
    def _create_signal_derivatives(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create derivative features that capture rate of change.
        
        Args:
            data: Input DataFrame
            
        Returns:
            DataFrame with additional features
        """
        df = data.copy()
        
        # Identify physiological signals
        signals = ['scr_mean', 'hr_mean', 'sdrr', 'pnn50', 'temp_mean', 'accel_mean', 'ibi_mean', 'raw_eda_mean']
        available_signals = [s for s in signals if s in df.columns]
        
        if len(available_signals) == 0:
            self.logger.warning("No appropriate signals found for derivatives")
            self.feature_engineering_stats['derivatives'] = 0
            self.feature_tracking['derivatives'] = []
            return df
            
        # Sort by pilot and trial for proper derivatives
        if 'trial' in df.columns:
            df = df.sort_values([self.id_column, 'trial']).reset_index(drop=True)
        else:
            # If no trial column, use turbulence as a proxy for sorting
            if 'turbulence' in df.columns:
                df = df.sort_values([self.id_column, 'turbulence']).reset_index(drop=True)
            else:
                # Create a generic sequence
                df['_temp_seq'] = 1
                df['_temp_seq'] = df.groupby(self.id_column)['_temp_seq'].cumsum()
                df = df.sort_values([self.id_column, '_temp_seq']).reset_index(drop=True)
        
        deriv_feats = []
        for feat in available_signals:
            # First derivative (rate of change)
            deriv = f'{feat}_derivative'
            df[deriv] = df.groupby(self.id_column)[feat].diff().fillna(0)
            deriv_feats.append(deriv)
            
            # Second derivative (acceleration of change)
            accel = f'{feat}_acceleration'
            df[accel] = df.groupby(self.id_column)[deriv].diff().fillna(0)
            deriv_feats.append(accel)
            
            # Moving average of signal
            ma = f'{feat}_moving_avg'
            df[ma] = df.groupby(self.id_column)[feat].transform(
                lambda x: x.rolling(window=3, min_periods=1).mean()
            )
            deriv_feats.append(ma)
            
            # Exponential weighted moving average
            ewma = f'{feat}_exp_avg'
            df[ewma] = df.groupby(self.id_column)[feat].transform(
                lambda x: x.ewm(span=3, min_periods=1).mean()
            )
            deriv_feats.append(ewma)
        
        # Remove temporary sequence column if created
        if '_temp_seq' in df.columns:
            df = df.drop('_temp_seq', axis=1)
            
        self.feature_engineering_stats['derivatives'] = len(deriv_feats)
        self.logger.info(f"Created {len(deriv_feats)} signal derivative features")
        
        # Track which features were created in this step
        self.feature_tracking['derivatives'] = deriv_feats
        
        return df
    
    def _create_signal_ratios(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create ratio features between different signals.
        
        Args:
            data: Input DataFrame
            
        Returns:
            DataFrame with additional features
        """
        df = data.copy()
        
        # Define pairs of signals to create ratios for
        signal_pairs = [
            ('scr_mean', 'hr_mean'),
            ('scr_std', 'hr_std'),
            ('scr_mean', 'temp_mean'),
            ('hr_mean', 'temp_mean'),
            ('sdrr', 'pnn50'),
            ('raw_eda_mean', 'accel_mean'),
            ('raw_eda_std', 'accel_std'),
            ('temp_mean', 'raw_eda_mean')
        ]
        
        # Only use available pairs
        available_pairs = [pair for pair in signal_pairs 
                         if pair[0] in df.columns 
                         and pair[1] in df.columns]
        
        if len(available_pairs) == 0:
            self.logger.warning("No appropriate signal pairs found for ratios")
            self.feature_engineering_stats['signal_ratios'] = 0
            self.feature_tracking['signal_ratios'] = []
            return df
            
        ratio_feats = []
        for signal1, signal2 in available_pairs:
            # Create ratio
            ratio = f'{signal1}_{signal2}_ratio'
            df[ratio] = df[signal1] / (df[signal2] + 1e-6)  # Adding small constant to avoid division by zero
            ratio_feats.append(ratio)
            
            # Create product
            product = f'{signal1}_{signal2}_product'
            df[product] = df[signal1] * df[signal2]
            ratio_feats.append(product)
            
            # Create difference
            diff = f'{signal1}_{signal2}_diff'
            df[diff] = df[signal1] - df[signal2]
            ratio_feats.append(diff)
            
            # Create normalized difference
            if signal1 != signal2:
                norm_diff = f'{signal1}_{signal2}_norm_diff'
                df[norm_diff] = (df[signal1] - df[signal2]) / (np.abs(df[signal1]) + np.abs(df[signal2]) + 1e-6)
                ratio_feats.append(norm_diff)
        
        self.feature_engineering_stats['signal_ratios'] = len(ratio_feats)
        self.logger.info(f"Created {len(ratio_feats)} signal ratio features")
        
        # Track which features were created in this step
        self.feature_tracking['signal_ratios'] = ratio_feats
        
        return df
    
    def _create_experience_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create features based on pilot experience level.
        
        Args:
            data: Input DataFrame
            
        Returns:
            DataFrame with additional features
        """
        df = data.copy()
        
        if 'pilot_category' not in df.columns:
            self.logger.warning("Pilot category column not found, skipping experience features")
            self.feature_engineering_stats['experience'] = 0
            self.feature_tracking['experience'] = []
            return df
            
        # Create numerical experience level
        exp_map = {'minimal_exp': 0, 'commercial': 1, 'air_force': 2}
        df['experience_level'] = df['pilot_category'].map(exp_map)
        
        exp_feats = ['experience_level']
        
        # Create experience-turbulence interaction
        if 'turbulence' in df.columns:
            # Linear interaction
            interact = 'exp_turb_interact'
            df[interact] = df['experience_level'] * df['turbulence']
            exp_feats.append(interact)
            
            # Squared interaction
            sq_interact = 'exp_turb_squared'
            df[sq_interact] = df['experience_level'] * (df['turbulence'] ** 2)
            exp_feats.append(sq_interact)
            
            # Ratio interaction
            ratio = 'turb_exp_ratio'
            df[ratio] = df['turbulence'] / (df['experience_level'] + 1)  # +1 to avoid division by zero
            exp_feats.append(ratio)
            
            # Categorical interaction features
            for category in ['minimal_exp', 'commercial', 'air_force']:
                # Create binary indicator
                indicator = f'is_{category}'
                df[indicator] = (df['pilot_category'] == category).astype(int)
                exp_feats.append(indicator)
                
                # Create interaction with turbulence
                cat_interact = f'{category}_turb_interact'
                df[cat_interact] = df[indicator] * df['turbulence']
                exp_feats.append(cat_interact)
        
        # Create experience-physiological interactions
        physio_signals = ['scr_mean', 'hr_mean', 'sdrr', 'pnn50', 'temp_mean', 'accel_mean', 'raw_eda_mean']
        available_signals = [s for s in physio_signals if s in df.columns]
        
        for feat in available_signals:
            # Create interaction with experience level
            exp_feat = f'{feat}_exp_interact'
            df[exp_feat] = df[feat] * df['experience_level']
            exp_feats.append(exp_feat)
            
            # Create interaction with each category
            for category in ['minimal_exp', 'commercial', 'air_force']:
                if f'is_{category}' in df.columns:
                    cat_feat = f'{feat}_{category}_interact'
                    df[cat_feat] = df[feat] * df[f'is_{category}']
                    exp_feats.append(cat_feat)
        
        self.feature_engineering_stats['experience'] = len(exp_feats)
        self.logger.info(f"Created {len(exp_feats)} experience-based features")
        
        # Track which features were created in this step
        self.feature_tracking['experience'] = exp_feats
        
        return df
    
    def _create_frequency_domain_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create frequency domain features for physiological signals.
        
        Args:
            data: Input DataFrame
            
        Returns:
            DataFrame with additional features
        """
        df = data.copy()
        
        # Check if we have trial information for frequency analysis
        if 'trial' not in df.columns:
            self.logger.warning("Trial column not found, skipping frequency domain features")
            self.feature_engineering_stats['frequency_domain'] = 0
            self.feature_tracking['frequency_domain'] = []
            return df
            
        # Identify signals for frequency analysis
        signals = ['scr_mean', 'hr_mean', 'raw_eda_mean']
        available_signals = [s for s in signals if s in df.columns]
        
        if len(available_signals) == 0:
            self.logger.warning("No appropriate signals found for frequency analysis")
            self.feature_engineering_stats['frequency_domain'] = 0
            self.feature_tracking['frequency_domain'] = []
            return df
            
        # Sort by pilot and trial
        df = df.sort_values([self.id_column, 'trial']).reset_index(drop=True)
        
        freq_feats = []
        
        # Process each pilot independently
        for pilot in df[self.id_column].unique():
            pilot_data = df[df[self.id_column] == pilot]
            
            for feat in available_signals:
                signal = pilot_data[feat].values
                
                # Skip if too few data points
                if len(signal) < 4:
                    continue
                
                try:
                    # Calculate autocorrelation (lag 1)
                    acf_val = acf(signal, nlags=1)[1]
                    df.loc[pilot_data.index, f'{feat}_autocorr'] = acf_val
                    freq_feats.append(f'{feat}_autocorr')
                    
                    # Calculate partial autocorrelation (lag 1)
                    pacf_val = pacf(signal, nlags=1)[1]
                    df.loc[pilot_data.index, f'{feat}_parcorr'] = pacf_val
                    freq_feats.append(f'{feat}_parcorr')
                    
                    # Simple spectral features (if enough data points)
                    if len(signal) >= 8:
                        # Approximate frequency analysis
                        f, psd = welch(signal, fs=1.0, nperseg=min(len(signal), 8))
                        
                        # Dominant frequency
                        dom_freq_idx = np.argmax(psd)
                        dom_freq = f[dom_freq_idx]
                        df.loc[pilot_data.index, f'{feat}_dom_freq'] = dom_freq
                        freq_feats.append(f'{feat}_dom_freq')
                        
                        # Power in dominant frequency
                        dom_power = psd[dom_freq_idx]
                        df.loc[pilot_data.index, f'{feat}_dom_power'] = dom_power
                        freq_feats.append(f'{feat}_dom_power')
                except Exception as e:
                    # Quietly handle errors in frequency analysis
                    self.logger.warning(f"Error in frequency analysis for {feat}: {str(e)}")
        
        # Fill any missing values created by the frequency analysis
        for feat in list(set(freq_feats)):
            if feat in df.columns:
                df[feat] = df[feat].fillna(0)
                
        # Remove duplicates from the frequency features list
        freq_feats = list(set(freq_feats))
        
        self.feature_engineering_stats['frequency_domain'] = len(freq_feats)
        self.logger.info(f"Created {len(freq_feats)} frequency domain features")
        
        # Track which features were created in this step
        self.feature_tracking['frequency_domain'] = freq_feats
        
        return df
    
    def _analyze_engineered_features(self, data: pd.DataFrame, new_features: List[str]) -> None:
        """
        Analyze the engineered features to identify potential issues.
        
        Args:
            data: DataFrame with features
            new_features: List of newly created feature names
        """
        if not new_features:
            self.logger.warning("No new features to analyze")
            return
            
        # Check for constant features
        constant_features = []
        for feature in new_features:
            if feature in data.columns:
                if data[feature].nunique() <= 1:
                    constant_features.append(feature)
        
        if constant_features:
            self.logger.warning(f"Found {len(constant_features)} constant features that may not be useful")
            
        # Check for highly correlated features
        if len(new_features) > 1:
            new_df = data[new_features].copy()
            
            # Handle non-numeric columns
            numeric_cols = new_df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                corr_matrix = new_df[numeric_cols].corr().abs()
                upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                high_corr_pairs = [(upper_tri.index[i], upper_tri.columns[j], upper_tri.iloc[i, j])
                                for i, j in zip(*np.where(upper_tri > 0.95))]
                
                if high_corr_pairs:
                    self.logger.warning(f"Found {len(high_corr_pairs)} pairs of highly correlated features (>0.95)")
                    
        # Check for correlation with target
        if self.target_column in data.columns:
            target_corrs = {}
            for feature in new_features:
                if feature in data.columns and pd.api.types.is_numeric_dtype(data[feature]):
                    corr = data[feature].corr(data[self.target_column])
                    target_corrs[feature] = corr
            
            # Find top correlations
            top_corrs = sorted(target_corrs.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
            
            if top_corrs:
                corr_summary = "\nTop features correlated with target:"
                for feature, corr in top_corrs:
                    corr_summary += f"\n  {feature}: {corr:.4f}"
                self.logger.info(corr_summary)
        
        # Create feature correlation visualization
        if self.output_dir and len(new_features) > 1:
            # Create a visualization of feature correlation with target
            if self.target_column in data.columns:
                plt.figure(figsize=(14, 10))
                
                # Calculate correlations with target
                numeric_cols = [col for col in new_features 
                               if col in data.columns and pd.api.types.is_numeric_dtype(data[col])]
                
                if numeric_cols:
                    corrs = []
                    for col in numeric_cols:
                        corr = data[col].corr(data[self.target_column])
                        corrs.append((col, corr))
                    
                    corrs = sorted(corrs, key=lambda x: abs(x[1]), reverse=True)
                    features, values = zip(*corrs[:20])  # Top 20
                    
                    # Create bar chart
                    plt.barh(range(len(features)), [abs(v) for v in values], align='center')
                    plt.yticks(range(len(features)), features)
                    plt.xlabel('Absolute Correlation with Target')
                    plt.title('Feature Correlation with Target')
                    plt.tight_layout()
                    
                    # Save figure
                    plt.savefig(os.path.join(self.output_dir, 'feature_target_correlation.png'), dpi=300)
                    plt.close()
                    
            # Create correlation heatmap for key features
            if len(numeric_cols) > 1:
                plt.figure(figsize=(16, 14))
                
                # Get top features by variance
                variances = data[numeric_cols].var().sort_values(ascending=False)
                top_vars = variances.head(min(20, len(variances))).index.tolist()
                
                # Create correlation matrix
                corr_matrix = data[top_vars].corr()
                
                # Plot heatmap
                sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0, 
                           square=True, linewidths=0.5)
                plt.title('Feature Correlation Matrix')
                plt.tight_layout()
                
                # Save figure
                plt.savefig(os.path.join(self.output_dir, 'feature_correlation_heatmap.png'), dpi=300)
                plt.close()
                
    def analyze_turbulence_relationship(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze the relationship between turbulence and cognitive load.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Dictionary with correlation statistics
        """
        self.logger.info("\nAnalyzing turbulence-cognitive load relationship...")
        
        if 'turbulence' not in data.columns:
            self.logger.warning("Turbulence column not found in data")
            return {}
            
        if self.target_column not in data.columns:
            self.logger.warning(f"Target column {self.target_column} not found in data")
            return {}
            
        # Calculate correlation
        corr = data['turbulence'].corr(data[self.target_column])
        self.logger.info(f"Correlation between turbulence and cognitive load: {corr:.4f}")
        
        # Create visualization
        if self.output_dir:
            plt.figure(figsize=(14, 8))
            
            # Add jittered stripplot
            sns.boxplot(x='turbulence', y=self.target_column, data=data, 
                       palette='viridis')
            sns.stripplot(x='turbulence', y=self.target_column, data=data,
                         size=4, color=".3", linewidth=0, alpha=0.4, jitter=True)
            
            # Add regression line
            sns.regplot(x='turbulence', y=self.target_column, data=data,
                        scatter=False, color='red')
            
            # Add statistics to plot
            plt.title(f'Turbulence vs. Cognitive Load (Correlation: {corr:.4f})', fontsize=14)
            plt.xlabel('Turbulence Level', fontsize=12)
            plt.ylabel('Cognitive Load (TLX Quantile)', fontsize=12)
            
            # Add means for each group
            means = data.groupby('turbulence')[self.target_column].mean()
            stds = data.groupby('turbulence')[self.target_column].std()
            counts = data.groupby('turbulence')[self.target_column].count()
            
            for i, (level, mean) in enumerate(means.items()):
                std = stds[level]
                plt.text(i, mean + 0.03, f'Mean: {mean:.4f}\nSD: {std:.4f}\nn={counts[level]}', 
                         ha='center', va='bottom', fontsize=10)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'turbulence_relationship.png'), dpi=300)
            plt.close()
        
        # Create ANOVA to test for differences between turbulence levels
        try:
            from scipy import stats
            groups = [data[data['turbulence'] == level][self.target_column] 
                    for level in sorted(data['turbulence'].unique())]
            f_stat, p_val = stats.f_oneway(*groups)
            self.logger.info(f"ANOVA test: F={f_stat:.4f}, p={p_val:.6f}")
        except Exception as e:
            self.logger.warning(f"Error performing ANOVA test: {str(e)}")
            f_stat, p_val = None, None
        
        # Create per-category analysis
        category_correlations = {}
        if 'pilot_category' in data.columns:
            plt.figure(figsize=(16, 8))
            
            # Plot by category
            sns.lmplot(x='turbulence', y=self.target_column, data=data,
                      hue='pilot_category', height=6, aspect=1.5, scatter_kws={'alpha': 0.4}, 
                      palette='viridis')
            
            plt.title('Turbulence vs. Cognitive Load by Pilot Category', fontsize=14)
            plt.xlabel('Turbulence Level', fontsize=12)
            plt.ylabel('Cognitive Load (TLX Quantile)', fontsize=12)
            plt.tight_layout()
            if self.output_dir:
                plt.savefig(os.path.join(self.output_dir, 'turbulence_by_category.png'), dpi=300)
            plt.close()
            
            # Calculate correlation by category
            for category in data['pilot_category'].unique():
                cat_data = data[data['pilot_category'] == category]
                cat_corr = cat_data['turbulence'].corr(cat_data[self.target_column])
                category_correlations[category] = cat_corr
                self.logger.info(f"Correlation for {category} pilots: {cat_corr:.4f}")
                
                # Perform ANOVA by category
                try:
                    cat_groups = [cat_data[cat_data['turbulence'] == level][self.target_column] 
                                for level in sorted(cat_data['turbulence'].unique())]
                    if len(cat_groups) > 1 and all(len(g) > 0 for g in cat_groups):
                        cat_f_stat, cat_p_val = stats.f_oneway(*cat_groups)
                        self.logger.info(f"ANOVA test for {category} pilots: F={cat_f_stat:.4f}, p={cat_p_val:.6f}")
                except Exception as e:
                    self.logger.warning(f"Error performing ANOVA test for {category} pilots: {str(e)}")
        
        # Return correlation statistics
        results = {
            'overall_correlation': corr,
            'anova_f': f_stat,
            'anova_p': p_val
        }
        
        if category_correlations:
            results['category_correlations'] = category_correlations
            
        return results
        
    def get_feature_tracking(self) -> Dict[str, List[str]]:
        """
        Get the tracking information for engineered features.
        
        Returns:
            Dictionary with feature tracking information
        """
        return self.feature_tracking
        
    def save_feature_tracking(self) -> str:
        """
        Save feature tracking information to a file.
        
        Returns:
            Path to the saved file
        """
        if not self.output_dir:
            self.logger.warning("No output directory specified, skipping saving feature tracking")
            return None
            
        import json
        
        # Save to file
        tracking_path = os.path.join(self.output_dir, 'feature_tracking.json')
        with open(tracking_path, 'w') as f:
            json.dump(self.feature_tracking, f, indent=2)
            
        self.logger.info(f"Feature tracking saved to: {tracking_path}")
        
        return tracking_path
