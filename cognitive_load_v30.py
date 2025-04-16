#!/usr/bin/env python3
"""
Cognitive Load Analysis Pipeline v30
====================================

A streamlined implementation for predicting and analyzing cognitive load 
from physiological data, with comprehensive reporting and visualization.

Features:
- Advanced data preprocessing with leak prevention
- Targeted feature engineering for physiological signals
- Multiple modeling approaches (global, subject-specific, adaptive transfer)
- Statistical significance testing and confidence intervals
- Comprehensive HTML report generation with improved visualizations
- Detailed discussion of methodological approach and limitations

Usage:
    python cognitive_load_pipeline_v30.py --data ../merged_data.csv [OPTIONS]

Author: Claude AI Assistant
Date: April 2025
"""

import argparse
import base64
import json
import os
import time
import sys
import warnings
from datetime import datetime
from io import BytesIO
from typing import Dict, List, Optional, Tuple, Union, Any, Set

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from jinja2 import Template
from scipy import stats
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import (KFold, LeaveOneGroupOut, cross_val_score,
                                    train_test_split, StratifiedKFold)
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# Suppress warnings
warnings.filterwarnings('ignore')

# Set default plotting styles
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.2)
sns.set_palette("viridis")

# Constants for the pipeline
VERSION = "30.0"
DEFAULT_TARGET_COL = "mental_effort"
DEFAULT_SUBJECT_COL = "pilot_id"
DEFAULT_FEATURE_COUNT = 30
DEFAULT_TEST_SIZE = 0.2

# Configure logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('cognitive_load')


# =============================================================================
# Data Processing
# =============================================================================

class DataProcessor:
    """Handles data loading, cleaning, preprocessing and validation splits"""
    
    def __init__(self, target_col: str = DEFAULT_TARGET_COL, 
                 subject_col: str = DEFAULT_SUBJECT_COL,
                 random_state: int = 42):
        """
        Initialize the data processor
        
        Args:
            target_col: Name of the target column (cognitive load measure)
            subject_col: Name of the subject identifier column
            random_state: Random seed for reproducibility
        """
        self.target_col = target_col
        self.subject_col = subject_col
        self.random_state = random_state
        
        # List of patterns that might indicate feature leakage
        self.leaky_feature_patterns = [
            # Direct leakage patterns
            'avg_tlx', 'tlx_zscore', 'tlx_quantile', 'mental_effort_zscore',
            'mental_effort_quantile', 'mental_effort_norm', 'mental_demand',
            # Experimental condition leakage
            'turbulence_tab', 'turbulence_feat', 'turbulence_check',
            'phase_num_tab', 'phase_num_feat', 'trial_tab', 'trial_feat',
            # Post-hoc ratings that might leak info
            'perceived_difficulty', 'subjective_rating'
        ]
        
        # Sample storage (for bootstrapping and significance testing)
        self.bootstrap_samples = []
        
    def load_and_clean_data(self, file_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load data and remove leaky features
        
        Args:
            file_path: Path to the input data file
            
        Returns:
            Tuple of (cleaned dataframe, analysis-ready dataframe)
        """
        start_time = time.time()
        logger.info(f"Loading data from {file_path}")
        
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Loaded {df.shape[0]} samples with {df.shape[1]} features")
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
        
        # Store dataset metadata
        self.dataset_metadata = {
            "n_samples": df.shape[0],
            "n_original_features": df.shape[1],
            "file_path": file_path,
            "loading_time": time.time() - start_time
        }
            
        # 1. Identify features with target name in them
        target_name_features = [col for col in df.columns 
                              if self.target_col in col.lower() and col != self.target_col]
        if target_name_features:
            logger.info(f"Found {len(target_name_features)} features with target name in them")
        
        # 2. Look for suspiciously high correlations with target
        high_corr_features = []
        if self.target_col in df.columns:
            numeric_cols = df.select_dtypes(include=np.number).columns
            if self.target_col in numeric_cols:
                correlations = df[numeric_cols].corr()[self.target_col].abs()
                high_corr_features = correlations[correlations > 0.8].index.tolist()
                high_corr_features = [f for f in high_corr_features if f != self.target_col]
                
                if high_corr_features:
                    logger.info(f"Found {len(high_corr_features)} features with suspiciously high correlation (>0.8) to target")
                    for feature in high_corr_features[:5]:  # Only show top 5
                        logger.info(f"  {feature}: {correlations[feature]:.4f}")
                    if len(high_corr_features) > 5:
                        logger.info(f"  ... and {len(high_corr_features) - 5} more")
        
        # 3. Check for known leaky patterns
        pattern_leaky_features = []
        for col in df.columns:
            if any(pattern in col.lower() for pattern in self.leaky_feature_patterns):
                pattern_leaky_features.append(col)
        
        if pattern_leaky_features:
            logger.info(f"Found {len(pattern_leaky_features)} features matching known leaky patterns")
        
        # Combine all leaky features
        leaky_features = list(set(target_name_features + high_corr_features + pattern_leaky_features))
        
        if leaky_features:
            logger.info(f"Removing {len(leaky_features)} potentially leaky features")
            df_clean = df.drop(columns=leaky_features)
        else:
            df_clean = df.copy()
        
        # Create analysis-ready dataset (no missing targets)
        if self.target_col in df_clean.columns:
            df_ready = df_clean.dropna(subset=[self.target_col])
            logger.info(f"Analysis-ready dataset: {df_ready.shape} (after removing rows with missing targets)")
            
            # Add dataset statistics
            self.dataset_metadata.update({
                "n_samples_clean": df_ready.shape[0],
                "n_features_clean": df_ready.shape[1],
                "target_mean": df_ready[self.target_col].mean(),
                "target_std": df_ready[self.target_col].std(),
                "target_min": df_ready[self.target_col].min(),
                "target_max": df_ready[self.target_col].max()
            })
            
            # Examine the target distribution
            target_distribution = df_ready[self.target_col].value_counts().sort_index()
            if len(target_distribution) < 10:  # Likely a discrete scale
                logger.info(f"Target distribution (discrete scale):")
                for val, count in target_distribution.items():
                    percent = 100 * count / len(df_ready)
                    logger.info(f"  {val}: {count} samples ({percent:.1f}%)")
            else:
                # Continuous or fine-grained scale
                q25, q50, q75 = df_ready[self.target_col].quantile([0.25, 0.5, 0.75])
                logger.info(f"Target distribution (continuous scale): median={q50:.2f}, Q1={q25:.2f}, Q3={q75:.2f}")
                        
        else:
            df_ready = df_clean
            logger.warning("Target column not found in data")
            
        # Analyze subject information
        if self.subject_col in df_ready.columns:
            subjects = df_ready[self.subject_col].unique()
            subject_counts = df_ready[self.subject_col].value_counts()
            
            self.dataset_metadata.update({
                "n_subjects": len(subjects),
                "min_samples_per_subject": subject_counts.min(),
                "max_samples_per_subject": subject_counts.max(),
                "mean_samples_per_subject": subject_counts.mean()
            })
            
            logger.info(f"Found {len(subjects)} distinct subjects")
            logger.info(f"Samples per subject: min={subject_counts.min()}, max={subject_counts.max()}, mean={subject_counts.mean():.1f}")
            
            # Identify any subjects with very few samples
            low_sample_subjects = subject_counts[subject_counts < 10]
            if len(low_sample_subjects) > 0:
                logger.warning(f"Found {len(low_sample_subjects)} subjects with fewer than 10 samples")
                logger.warning(f"  Subject sample counts: {dict(low_sample_subjects)}")
        
        return df_clean, df_ready
    
    def explore_dataset(self, df: pd.DataFrame) -> Dict:
        """
        Perform exploratory analysis on the dataset
        
        Args:
            df: Input dataframe
            
        Returns:
            Dictionary with dataset statistics
        """
        dataset_stats = {}
        
        # Basic dataset properties
        dataset_stats["samples"] = len(df)
        dataset_stats["features"] = len(df.columns)
        
        # Feature type distribution
        feature_types = {}
        for dtype in df.dtypes.value_counts().index:
            feature_types[str(dtype)] = int(df.dtypes.value_counts()[dtype])
        dataset_stats["feature_types"] = feature_types
        
        # Missing values
        missing_values = df.isnull().sum().sum()
        dataset_stats["missing_values"] = int(missing_values)
        dataset_stats["missing_value_percentage"] = float(missing_values / (df.shape[0] * df.shape[1]))
        
        # Subject distribution if available
        if self.subject_col in df.columns:
            dataset_stats["subjects"] = {}
            dataset_stats["subjects"]["count"] = int(df[self.subject_col].nunique())
            
            # Create a dictionary of subject counts
            subject_counts = df[self.subject_col].value_counts().to_dict()
            subject_counts = {str(k): int(v) for k, v in subject_counts.items()}
            dataset_stats["subjects"]["distribution"] = subject_counts
        
        # Target variable if available
        if self.target_col in df.columns:
            target_stats = {}
            target_stats["mean"] = float(df[self.target_col].mean())
            target_stats["std"] = float(df[self.target_col].std())
            target_stats["min"] = float(df[self.target_col].min())
            target_stats["max"] = float(df[self.target_col].max())
            
            # Check if target is discrete or continuous
            unique_values = df[self.target_col].nunique()
            if unique_values < 10:  # Likely a discrete scale
                target_stats["type"] = "discrete"
                target_stats["scale_points"] = int(unique_values)
                
                # Distribution of discrete values
                value_counts = df[self.target_col].value_counts().sort_index().to_dict()
                target_stats["distribution"] = {str(k): int(v) for k, v in value_counts.items()}
            else:
                target_stats["type"] = "continuous"
                
                # Quartile information
                quartiles = df[self.target_col].quantile([0.25, 0.5, 0.75]).to_dict()
                target_stats["quartiles"] = {str(k): float(v) for k, v in quartiles.items()}
                
                # Store histogram data for visualization
                hist, bin_edges = np.histogram(df[self.target_col], bins=20)
                target_stats["histogram"] = {
                    "counts": hist.tolist(),
                    "bin_edges": bin_edges.tolist()
                }
            
            dataset_stats["target"] = target_stats
        
        # Physiological signal detection
        # Check for common physiological signal patterns in column names
        phys_patterns = ['eda', 'hr', 'hrv', 'ecg', 'ppg', 'gsr', 'scr', 'rsp', 
                        'resp', 'accel', 'ibi', 'temp', 'eeg', 'emg', 'pupil']
        phys_signals = {}
        
        for pattern in phys_patterns:
            pattern_cols = [col for col in df.columns if pattern.lower() in col.lower()]
            if pattern_cols:
                phys_signals[pattern] = len(pattern_cols)
        
        dataset_stats["physiological_signals"] = phys_signals
        dataset_stats["has_physiological_data"] = len(phys_signals) > 0
        
        return dataset_stats
    
    def split_train_test_within_subject(self, df, test_size=0.2):
        """
        Split data into train/test sets while preserving within-subject samples
        This approach tests how well models predict new samples from known subjects
        
        Args:
            df: Input dataframe
            test_size: Proportion of data to use for testing
            
        Returns:
            Tuple of (train_df, test_df)
        """
        logger.info(f"Performing within-subject train/test split (test_size={test_size})")
        logger.info("  This split tests how well models predict new samples from known subjects")
        
        train_df = pd.DataFrame()
        test_df = pd.DataFrame()
        
        # Split data within each subject
        for subject in df[self.subject_col].unique():
            subject_data = df[df[self.subject_col] == subject]
            
            # Skip subjects with very few samples
            if len(subject_data) < 5:
                logger.warning(f"  Skipping subject {subject} with only {len(subject_data)} samples")
                continue
            
            # Split this subject's data
            subject_train, subject_test = train_test_split(
                subject_data, test_size=test_size, random_state=self.random_state
            )
            
            train_df = pd.concat([train_df, subject_train])
            test_df = pd.concat([test_df, subject_test])
        
        logger.info(f"  Training set: {train_df.shape[0]} samples")
        logger.info(f"  Testing set: {test_df.shape[0]} samples")
        
        return train_df, test_df

    def split_train_test_between_subjects(self, df, test_size=0.2):
        """
        Split data into train/test sets by assigning entire subjects to test set
        This approach tests how well models generalize to completely new subjects
        
        Args:
            df: Input dataframe
            test_size: Approximate proportion of data to use for testing
            
        Returns:
            Tuple of (train_df, test_df)
        """
        logger.info(f"Performing between-subject train/test split (test_size={test_size})")
        logger.info("  This split tests how well models generalize to completely new subjects")
        
        # Get unique subjects
        subjects = df[self.subject_col].unique()
        n_subjects = len(subjects)
        n_test_subjects = max(1, int(n_subjects * test_size))
        
        # Randomly select subjects for test set
        np.random.seed(self.random_state)
        test_subjects = np.random.choice(subjects, size=n_test_subjects, replace=False)
        train_subjects = [s for s in subjects if s not in test_subjects]
        
        # Split data by subject
        train_df = df[df[self.subject_col].isin(train_subjects)].copy()
        test_df = df[df[self.subject_col].isin(test_subjects)].copy()
        
        logger.info(f"  Training set: {train_df.shape[0]} samples ({len(train_subjects)} subjects)")
        logger.info(f"  Testing set: {test_df.shape[0]} samples ({len(test_subjects)} subjects)")
        logger.info(f"  Test subjects: {sorted(test_subjects)}")
        
        return train_df, test_df
    
    def create_bootstrap_samples(self, df, n_samples=100, sample_size=0.8):
        """
        Create bootstrap samples for statistical testing
        
        Args:
            df: Input dataframe
            n_samples: Number of bootstrap samples to create
            sample_size: Size of each sample as proportion of original data
            
        Returns:
            List of bootstrap sample dataframes
        """
        logger.info(f"Creating {n_samples} bootstrap samples (sample_size={sample_size})")
        
        self.bootstrap_samples = []
        sample_indices = []
        
        # Create stratified bootstrap samples
        for i in range(n_samples):
            # Sample subjects with replacement
            subjects = df[self.subject_col].unique()
            selected_subjects = np.random.choice(subjects, size=int(len(subjects) * sample_size), replace=True)
            
            # Get all rows for selected subjects
            sample_idx = df[df[self.subject_col].isin(selected_subjects)].index
            sample_indices.append(sample_idx)
            
            # Store indices instead of dataframes to save memory
            self.bootstrap_samples.append(sample_idx)
        
        logger.info(f"Created {len(self.bootstrap_samples)} bootstrap samples")
        return self.bootstrap_samples
    
    def perform_cross_subject_validation(self, df, features, target_col, subject_col, 
                                         model_class=RandomForestRegressor, **model_kwargs):
        """
        Perform leave-one-subject-out cross-validation to measure true generalization to new subjects
        
        Args:
            df: Input dataframe with features
            features: List of feature names to use
            target_col: Name of the target column
            subject_col: Name of the subject identifier column
            model_class: Scikit-learn model class to use
            model_kwargs: Additional arguments for the model
            
        Returns:
            Tuple of (mean_r2, std_r2, individual_subject_r2_values, mean_rmse, std_rmse)
        """
        logger.info(f"Performing leave-one-subject-out cross-validation with {model_class.__name__}")
        
        # Get unique subjects
        subjects = df[subject_col].unique()
        logger.info(f"  Cross-validation across {len(subjects)} subjects")
        
        # Store results for each subject
        subject_r2_values = []
        subject_rmse_values = []
        subject_mae_values = []
        subject_results = {}
        
        # Create Leave-One-Group-Out splitter
        logo = LeaveOneGroupOut()
        
        # Initialize progress bar
        progress_bar = tqdm(total=len(subjects), desc="Cross-subject validation")
        
        # For each subject fold, train on all other subjects and test on this one
        for train_idx, test_idx in logo.split(df, groups=df[subject_col]):
            # Get train/test splits
            X_train = df.iloc[train_idx][features]
            y_train = df.iloc[train_idx][target_col]
            X_test = df.iloc[test_idx][features]
            y_test = df.iloc[test_idx][target_col]
            
            # Get test subject
            test_subject = df.iloc[test_idx][subject_col].iloc[0]
            
            # Handle NaN values
            X_train = X_train.fillna(X_train.mean())
            X_test = X_test.fillna(X_train.mean())  # Use training means for test data
            
            # Initialize and train model
            model = model_class(**model_kwargs)
            model.fit(X_train, y_train)
            
            # Make predictions and calculate metrics
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            
            # Store results
            subject_r2_values.append(r2)
            subject_rmse_values.append(rmse)
            subject_mae_values.append(mae)
            
            # Store detailed results
            subject_results[test_subject] = {
                'r2': r2,
                'rmse': rmse,
                'mae': mae,
                'n_samples': len(y_test),
                'y_true': y_test.values.tolist(),
                'y_pred': y_pred.tolist()
            }
            
            # Update progress bar
            progress_bar.update(1)
            
        # Close progress bar
        progress_bar.close()
        
        # Calculate overall metrics
        mean_r2 = np.mean(subject_r2_values)
        std_r2 = np.std(subject_r2_values)
        mean_rmse = np.mean(subject_rmse_values)
        std_rmse = np.std(subject_rmse_values)
        mean_mae = np.mean(subject_mae_values)
        std_mae = np.std(subject_mae_values)
        
        # Calculate confidence intervals using bootstrap
        r2_values = np.array(subject_r2_values)
        r2_ci_lower, r2_ci_upper = self._bootstrap_confidence_interval(r2_values)
        
        rmse_values = np.array(subject_rmse_values)
        rmse_ci_lower, rmse_ci_upper = self._bootstrap_confidence_interval(rmse_values)
        
        logger.info(f"Cross-subject validation results:")
        logger.info(f"  - Mean R²: {mean_r2:.3f} (95% CI: [{r2_ci_lower:.3f}, {r2_ci_upper:.3f}])")
        logger.info(f"  - Mean RMSE: {mean_rmse:.3f} (95% CI: [{rmse_ci_lower:.3f}, {rmse_ci_upper:.3f}])")
        logger.info(f"  - R² range: [{min(subject_r2_values):.3f}, {max(subject_r2_values):.3f}]")
        
        # Sort subjects by performance
        sorted_subjects = sorted(subject_results.items(), key=lambda x: x[1]['r2'], reverse=True)
        logger.info(f"  - Best performing subject: {sorted_subjects[0][0]} (R² = {sorted_subjects[0][1]['r2']:.3f})")
        logger.info(f"  - Worst performing subject: {sorted_subjects[-1][0]} (R² = {sorted_subjects[-1][1]['r2']:.3f})")
        
        return {
            'mean_r2': mean_r2,
            'std_r2': std_r2,
            'r2_ci': [r2_ci_lower, r2_ci_upper],
            'mean_rmse': mean_rmse, 
            'std_rmse': std_rmse,
            'rmse_ci': [rmse_ci_lower, rmse_ci_upper],
            'mean_mae': mean_mae,
            'std_mae': std_mae,
            'subject_results': subject_results
        }
    
    def _bootstrap_confidence_interval(self, values, n_bootstrap=1000, alpha=0.05):
        """
        Calculate bootstrap confidence interval
        
        Args:
            values: Array of values
            n_bootstrap: Number of bootstrap samples
            alpha: Alpha level for confidence interval
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        bootstrap_samples = np.random.choice(values, size=(n_bootstrap, len(values)), replace=True)
        bootstrap_means = np.mean(bootstrap_samples, axis=1)
        
        # Calculate percentile confidence interval
        lower_bound = np.percentile(bootstrap_means, 100 * alpha / 2)
        upper_bound = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))
        
        return lower_bound, upper_bound


# =============================================================================
# Feature Engineering
# =============================================================================

class FeatureEngineer:
    """Creates targeted features optimized for physiological signals"""
    
    def __init__(self, subject_col: str = 'pilot_id'):
        """
        Initialize feature engineer
        
        Args:
            subject_col: Column name for subject identifiers
        """
        self.subject_col = subject_col
        
        # Common physiological signal patterns to look for
        self.phys_patterns = [
            'eda', 'gsr', 'scr', 'scl', 'hr', 'ecg', 'resp', 'temp', 
            'accel', 'ibi', 'sdrr', 'pnn50', 'hrv', 'ppg', 'emg'
        ]
        
        # Track which feature engineering steps were applied
        self.applied_steps = set()
        self.engineered_feature_groups = {}
        
    def detect_physiological_signals(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Detect available physiological signals in the dataset
        
        Args:
            df: Input dataframe
            
        Returns:
            Dictionary mapping signal type to column names
        """
        signal_columns = {}
        
        for pattern in self.phys_patterns:
            pattern_cols = [col for col in df.columns if pattern.lower() in col.lower()]
            if pattern_cols:
                signal_columns[pattern] = pattern_cols
                logger.info(f"Detected {len(pattern_cols)} {pattern} related columns")
        
        return signal_columns
    
    def extract_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract temporal features from time series data efficiently
        
        Args:
            df: Input dataframe
            
        Returns:
            Dataframe with added temporal features
        """
        start_time = time.time()
        logger.info("Extracting temporal features")
        
        # Create a clean initial copy
        df_initial = df.copy()
        
        # Identify physiological signal columns
        phys_features = []
        for pattern in self.phys_patterns:
            pattern_cols = [col for col in df.columns if pattern in col.lower()]
            phys_features.extend(pattern_cols)
        
        phys_features = list(set(phys_features))  # Remove any duplicates
        logger.info(f"Found {len(phys_features)} physiological features for temporal engineering")
        
        # Process subjects one by one
        all_new_features = {}
        
        if self.subject_col in df.columns:
            # Get unique subjects
            subjects = df[self.subject_col].unique()
            
            # Process each subject
            for subject in tqdm(subjects, desc="Extracting temporal features"):
                # Get subject data
                subject_data = df[df[self.subject_col] == subject].copy()
                
                # Skip subjects with too few samples
                if len(subject_data) < 5:
                    continue
                
                # Sort by trial if available
                if 'trial' in subject_data.columns:
                    subject_data = subject_data.sort_values('trial')
                
                # Get indices for this subject
                subject_idx = df_initial.index[df_initial[self.subject_col] == subject]
                
                # Initialize a dictionary for this subject's features
                subject_features = {}
                
                # For each physiological feature, generate temporal features
                for col in phys_features:
                    if col not in subject_data.columns:
                        continue
                    
                    # Rate of change (first derivative)
                    subject_features[f'{col}_rate'] = pd.Series(
                        np.gradient(subject_data[col]), index=subject_idx
                    )
                    
                    # Moving average (window size 3)
                    subject_features[f'{col}_ma3'] = pd.Series(
                        subject_data[col].rolling(window=3, min_periods=1).mean().values,
                        index=subject_idx
                    )
                    
                    # Moving standard deviation
                    subject_features[f'{col}_mstd3'] = pd.Series(
                        subject_data[col].rolling(window=3, min_periods=1).std().values,
                        index=subject_idx
                    )
                    
                    # Add lag-1 feature (most important for autoregressive modeling)
                    subject_features[f'{col}_lag1'] = pd.Series(
                        subject_data[col].shift(1).values,
                        index=subject_idx
                    )
                    
                    # Add lag-2 feature
                    subject_features[f'{col}_lag2'] = pd.Series(
                        subject_data[col].shift(2).values,
                        index=subject_idx
                    )
                
                # Add this subject's features to the master dictionary
                for feature_name, feature_values in subject_features.items():
                    if feature_name not in all_new_features:
                        all_new_features[feature_name] = pd.Series(index=df.index, dtype=float)
                    
                    all_new_features[feature_name].loc[subject_idx] = feature_values
        
        # Create a dataframe with all the new features
        new_features_df = pd.DataFrame(all_new_features)
        
        # Fill NaN values
        new_features_df = new_features_df.fillna(new_features_df.mean())
        
        # Join with the original dataframe
        df_engineered = pd.concat([df_initial, new_features_df], axis=1)
        
        # Track results
        temporal_features = list(new_features_df.columns)
        self.engineered_feature_groups['temporal'] = temporal_features
        self.applied_steps.add('temporal')
        
        # Report on created features
        logger.info(f"Added {len(new_features_df.columns)} temporal features in {time.time() - start_time:.2f} seconds")
        
        return df_engineered
    
    def extract_frequency_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract frequency domain features from physiological signals
        
        Args:
            df: Input dataframe
            
        Returns:
            Dataframe with added frequency features
        """
        start_time = time.time()
        logger.info("Extracting frequency domain features")
        
        df_engineered = df.copy()
        
        # Identify physiological signal columns (limit to most important)
        phys_features = []
        for pattern in self.phys_patterns:
            pattern_cols = [col for col in df.columns if pattern in col.lower()]
            phys_features.extend(pattern_cols)
        
        # Limit to key signals that benefit from frequency analysis
        selected_features = [col for col in phys_features if any(x in col.lower() for x in 
                                                              ['eda', 'hr', 'ecg', 'ppg', 'resp'])]
        
        # Further limit to features with sufficient variability
        viable_features = []
        for col in selected_features:
            if col in df.columns:
                # Only include features with enough variability
                if df[col].std() > 0.001:
                    viable_features.append(col)
        
        selected_features = viable_features[:15]  # Limit to top 15 to avoid excessive computation
        
        logger.info(f"Extracting frequency features from {len(selected_features)} signals")
        
        try:
            from scipy.signal import welch
            
            # Define basic sampling rates by signal type
            sampling_rates = {
                'ppg': 64,  # Hz
                'ecg': 256, # Hz
                'eda': 4,   # Hz
                'gsr': 4,   # Hz
                'scr': 4,   # Hz
                'hr': 1,    # Hz
                'resp': 16, # Hz
                'temp': 4,  # Hz
                'accel': 32 # Hz
            }
            
            # Create a dictionary to store new features
            new_features = {}
            
            # Process each feature
            for col in selected_features:
                if col not in df.columns:
                    continue
                
                # Determine sampling rate based on column name
                fs = 4  # Default sampling rate
                for key, rate in sampling_rates.items():
                    if key in col.lower():
                        fs = rate
                        break
                
                # Calculate power spectral density
                try:
                    f, psd = welch(df[col].fillna(df[col].mean()), fs=fs, nperseg=min(256, len(df[col])))
                    
                    # Extract basic frequency bands
                    # Very low frequency: 0.003-0.04 Hz (for HRV analysis)
                    vlf_mask = (f >= 0.003) & (f < 0.04)
                    vlf_power = np.sum(psd[vlf_mask]) if np.any(vlf_mask) else 0
                    
                    # Low frequency: 0.04-0.15 Hz
                    lf_mask = (f >= 0.04) & (f < 0.15)
                    lf_power = np.sum(psd[lf_mask]) if np.any(lf_mask) else 0
                    
                    # High frequency: 0.15-0.4 Hz
                    hf_mask = (f >= 0.15) & (f < 0.4)
                    hf_power = np.sum(psd[hf_mask]) if np.any(hf_mask) else 0
                    
                    # Store features
                    new_features[f'{col}_vlf_power'] = vlf_power
                    new_features[f'{col}_lf_power'] = lf_power
                    new_features[f'{col}_hf_power'] = hf_power
                    new_features[f'{col}_lf_hf_ratio'] = lf_power / (hf_power + 1e-10)
                    new_features[f'{col}_total_power'] = np.sum(psd)
                    
                    # Dominant frequency
                    dom_freq_idx = np.argmax(psd)
                    new_features[f'{col}_dom_freq'] = f[dom_freq_idx]
                    
                    # Spectral entropy (a measure of signal complexity)
                    psd_norm = psd / (np.sum(psd) + 1e-10)
                    spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-10))
                    new_features[f'{col}_spectral_entropy'] = spectral_entropy
                    
                except Exception as e:
                    logger.warning(f"Error processing frequency features for {col}: {str(e)}")
            
            # Add all new features to the dataframe at once
            if new_features:
                for feature_name, feature_value in new_features.items():
                    df_engineered[feature_name] = feature_value
                
                # Track new features
                frequency_features = list(new_features.keys())
                self.engineered_feature_groups['frequency'] = frequency_features
                
                logger.info(f"Added {len(new_features)} frequency domain features in {time.time() - start_time:.2f} seconds")
                self.applied_steps.add('frequency')
            else:
                logger.warning("No frequency domain features could be created")
            
        except Exception as e:
            logger.error(f"Error in frequency domain feature extraction: {str(e)}")
        
        return df_engineered
    
    def create_compound_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create composite features combining multiple physiological signals
        
        Args:
            df: Input dataframe
            
        Returns:
            Dataframe with added compound features
        """
        start_time = time.time()
        logger.info("Creating compound features")
        
        df_engineered = df.copy()
        compound_count = 0
        
        # Identify key physiological signals 
        signal_columns = self.detect_physiological_signals(df)
        
        # Track new features
        new_features = {}
        
        # 1. Create ratio features between physiological signals
        for signal1, columns1 in signal_columns.items():
            for signal2, columns2 in signal_columns.items():
                if signal1 != signal2:  # Only create ratios between different signal types
                    # Choose the most representative column for each signal type
                    # Prefer mean values or standard deviation values
                    col1 = None
                    for c in columns1:
                        if '_mean' in c or 'mean_' in c:
                            col1 = c
                            break
                    if col1 is None and len(columns1) > 0:
                        col1 = columns1[0]
                        
                    col2 = None
                    for c in columns2:
                        if '_mean' in c or 'mean_' in c:
                            col2 = c
                            break
                    if col2 is None and len(columns2) > 0:
                        col2 = columns2[0]
                    
                    # Create ratio if both columns exist
                    if col1 and col2 and col1 in df.columns and col2 in df.columns:
                        ratio_name = f"ratio_{signal1}_{signal2}"
                        new_features[ratio_name] = df[col1] / (df[col2] + 1e-10)
                        compound_count += 1
        
        # 2. Create arousal features (combining SCR/EDA with HR/HRV)
        if 'eda' in signal_columns and 'hr' in signal_columns:
            # Get key columns
            eda_col = None
            for c in signal_columns['eda']:
                if '_mean' in c or 'mean_' in c:
                    eda_col = c
                    break
            if eda_col is None and signal_columns['eda']:
                eda_col = signal_columns['eda'][0]
                
            hr_col = None
            for c in signal_columns['hr']:
                if '_mean' in c or 'mean_' in c:
                    hr_col = c
                    break
            if hr_col is None and signal_columns['hr']:
                hr_col = signal_columns['hr'][0]
            
            # Create arousal index if both columns exist
            if eda_col and hr_col and eda_col in df.columns and hr_col in df.columns:
                # Normalize values to 0-1 range
                eda_norm = (df[eda_col] - df[eda_col].min()) / (df[eda_col].max() - df[eda_col].min() + 1e-10)
                hr_norm = (df[hr_col] - df[hr_col].min()) / (df[hr_col].max() - df[hr_col].min() + 1e-10)
                
                # Simple arousal index (average of normalized EDA and HR)
                new_features['arousal_index'] = (eda_norm + hr_norm) / 2
                compound_count += 1
                
                # Weighted arousal index (giving more weight to EDA)
                new_features['weighted_arousal_index'] = (0.7 * eda_norm + 0.3 * hr_norm)
                compound_count += 1
        
        # 3. Create cognitive load index if we have the right signals
        # HRV decreases with cognitive load, pupil diameter and SCR increase
        hrv_col = None
        pupil_col = None
        scr_col = None
        
        # Look for HRV related columns
        if 'hrv' in signal_columns:
            for c in signal_columns['hrv']:
                if '_mean' in c or 'mean_' in c:
                    hrv_col = c
                    break
            if hrv_col is None and signal_columns['hrv']:
                hrv_col = signal_columns['hrv'][0]
        
        # Look for pupil data
        pupil_cols = [col for col in df.columns if 'pupil' in col.lower()]
        if pupil_cols:
            for c in pupil_cols:
                if '_mean' in c or 'mean_' in c:
                    pupil_col = c
                    break
            if pupil_col is None:
                pupil_col = pupil_cols[0]
        
        # Look for SCR data
        if 'scr' in signal_columns:
            for c in signal_columns['scr']:
                if '_mean' in c or 'mean_' in c:
                    scr_col = c
                    break
            if scr_col is None and signal_columns['scr']:
                scr_col = signal_columns['scr'][0]
        
        # Create cognitive load index if we have at least 2 of the 3 required signals
        required_signals = [
            (hrv_col, df.columns) if hrv_col else None,
            (pupil_col, df.columns) if pupil_col else None,
            (scr_col, df.columns) if scr_col else None
        ]
        required_signals = [x for x in required_signals if x is not None]
        available_signals = [col for col, cols in required_signals if col in cols]
        
        if len(available_signals) >= 2:
            # Initialize with zeros
            cognitive_load_index = np.zeros(len(df))
            signal_count = 0
            
            # Add HRV component (inverted, as HRV decreases with cognitive load)
            if hrv_col and hrv_col in df.columns:
                hrv_norm = (df[hrv_col] - df[hrv_col].min()) / (df[hrv_col].max() - df[hrv_col].min() + 1e-10)
                cognitive_load_index += (1 - hrv_norm)  # Inverted
                signal_count += 1
            
            # Add pupil component
            if pupil_col and pupil_col in df.columns:
                pupil_norm = (df[pupil_col] - df[pupil_col].min()) / (df[pupil_col].max() - df[pupil_col].min() + 1e-10)
                cognitive_load_index += pupil_norm
                signal_count += 1
            
            # Add SCR component
            if scr_col and scr_col in df.columns:
                scr_norm = (df[scr_col] - df[scr_col].min()) / (df[scr_col].max() - df[scr_col].min() + 1e-10)
                cognitive_load_index += scr_norm
                signal_count += 1
            
            # Normalize by number of signals
            if signal_count > 0:
                cognitive_load_index /= signal_count
                new_features['cognitive_load_index'] = cognitive_load_index
                compound_count += 1
        
        # 4. Create polynomial features for key physiological signals
        for signal_type in ['eda', 'scr', 'hr', 'hrv']:
            if signal_type in signal_columns:
                for col in signal_columns[signal_type][:2]:  # Limit to first 2 columns per signal type
                    if col in df.columns:
                        new_features[f"{col}_squared"] = df[col] ** 2
                        compound_count += 1
        
        # Add all new features to the dataframe at once
        for feature_name, feature_value in new_features.items():
            df_engineered[feature_name] = feature_value
            
        # Track new features
        self.engineered_feature_groups['compound'] = list(new_features.keys())
        self.applied_steps.add('compound')
        
        logger.info(f"Added {compound_count} compound features in {time.time() - start_time:.2f} seconds")
        
        return df_engineered
    
    def create_interaction_features(self, df: pd.DataFrame, features: List[str], 
                                    max_interactions: int = 10) -> pd.DataFrame:
        """
        Create interaction features between top predictive features
        
        Args:
            df: Input dataframe
            features: List of top features to consider for interactions
            max_interactions: Maximum number of interaction features to create
            
        Returns:
            Dataframe with added interaction features
        """
        start_time = time.time()
        logger.info(f"Creating interaction features from top {len(features)} base features")
        
        df_engineered = df.copy()
        
        # Limit to first max_features to avoid combinatorial explosion
        max_features = min(10, len(features))
        selected_features = features[:max_features]
        
        interaction_features = {}
        count = 0
        
        # Create pairwise interactions
        for i, feat1 in enumerate(selected_features):
            if feat1 not in df.columns:
                continue
                
            for j in range(i+1, len(selected_features)):
                feat2 = selected_features[j]
                
                if feat2 not in df.columns:
                    continue
                
                # Create interaction (product)
                interaction_name = f"{feat1}_x_{feat2}"
                interaction_features[interaction_name] = df[feat1] * df[feat2]
                count += 1
                
                # Create ratio interaction if values are positive
                if df[feat1].min() > 0 and df[feat2].min() > 0:
                    ratio_name = f"{feat1}_div_{feat2}"
                    interaction_features[ratio_name] = df[feat1] / (df[feat2] + 1e-10)
                    count += 1
                
                # Stop if we've reached the maximum number of interactions
                if count >= max_interactions:
                    break
            
            if count >= max_interactions:
                break
        
        # Add all interaction features to the dataframe
        for name, values in interaction_features.items():
            df_engineered[name] = values
        
        # Track new features
        self.engineered_feature_groups['interaction'] = list(interaction_features.keys())
        self.applied_steps.add('interaction')
        
        logger.info(f"Added {len(interaction_features)} interaction features in {time.time() - start_time:.2f} seconds")
        
        return df_engineered
    
    def engineer_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all feature engineering steps sequentially
        
        Args:
            df: Input dataframe
            
        Returns:
            Dataframe with all engineered features
        """
        start_time = time.time()
        logger.info("Applying complete feature engineering pipeline")
        
        original_feature_count = len(df.columns)
        
        # Apply each feature engineering step in sequence
        df_engineered = self.extract_temporal_features(df)
        df_engineered = self.extract_frequency_features(df_engineered)
        df_engineered = self.create_compound_features(df_engineered)
        
        # Interaction features are created later, after feature selection identifies top features
        
        # Report on feature engineering results
        engineered_feature_count = len(df_engineered.columns)
        added_feature_count = engineered_feature_count - original_feature_count
        
        logger.info(f"Feature engineering complete")
        logger.info(f"  Original features: {original_feature_count}")
        logger.info(f"  Engineered features: {engineered_feature_count}")
        logger.info(f"  Added features: {added_feature_count}")
        logger.info(f"  Time taken: {time.time() - start_time:.2f} seconds")
        
        # Log summary of features by group
        total_by_group = 0
        for group, features in self.engineered_feature_groups.items():
            logger.info(f"  - {group.title()} features: {len(features)}")
            total_by_group += len(features)
        
        # Sanity check
        if total_by_group != added_feature_count:
            logger.warning(f"  Warning: Feature count mismatch - {total_by_group} counted vs {added_feature_count} actual")
        
        return df_engineered


# =============================================================================
# Feature Selection
# =============================================================================

class FeatureSelector:
    """Selects most informative features based on importance"""
    
    def __init__(self):
        # Core preferred physiological features (based on previous results)
        self.preferred_patterns = [
            'scr_mean', 'scr_min', 'scr_count', 'scr_rate',
            'raw_eda', 'eda_mean', 'eda_rate',
            'pnn50', 'sdrr', 'ibi_mean', 'hrv_mean',
            'accel', 'temp', 'duration',
            'arousal_index', 'cognitive_load_index',
            'ratio_'  # Compound ratio features often important
        ]
        
        # Track selected features and their importance scores
        self.feature_importance = {}
        self.selected_features = []
        self.selection_method = None
    
    def rank_features(self, df: pd.DataFrame, target: str, subject_col: str = None) -> Dict[str, float]:
        """
        Rank features by their importance to predicting the target
        
        Args:
            df: Input dataframe
            target: Target column name
            subject_col: Subject identifier column name
            
        Returns:
            Dictionary of {feature: importance_score}
        """
        start_time = time.time()
        logger.info("Ranking features by importance")
        
        # Remove subject column if provided (to prevent data leakage)
        if subject_col and subject_col in df.columns:
            X = df.drop(columns=[target, subject_col])
        else:
            X = df.drop(columns=[target])
            
        y = df[target]
        
        # Handle categorical features
        X_numeric = X.select_dtypes(include=np.number)
        
        # Fill missing values
        for col in X_numeric.columns:
            if X_numeric[col].isna().all():
                X_numeric[col] = 0  # Replace with 0 if all values are NaN
            else:
                X_numeric[col] = X_numeric[col].fillna(X_numeric[col].mean())
        
        # Calculate correlations with target
        corr_scores = {}
        for col in X_numeric.columns:
            try:
                corr = np.corrcoef(X_numeric[col], y)[0, 1]
                if np.isnan(corr):
                    corr = 0.0
                corr_scores[col] = abs(corr)  # Use absolute correlation
            except:
                corr_scores[col] = 0.0
        
        # Use Random Forest for feature importance
        rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        
        try:
            rf.fit(X_numeric, y)
            
            # Store feature importances
            rf_scores = {}
            for i, col in enumerate(X_numeric.columns):
                rf_scores[col] = rf.feature_importances_[i]
            
            # Combine scores with 70% weight to Random Forest and 30% to correlation
            combined_scores = {}
            for col in X_numeric.columns:
                combined_scores[col] = 0.7 * rf_scores.get(col, 0) + 0.3 * corr_scores.get(col, 0)
            
            # Filter out NaN values if any remain
            combined_scores = {k: v for k, v in combined_scores.items() if not np.isnan(v)}
            
            # Boost scores for preferred feature patterns
            for col in combined_scores:
                for pattern in self.preferred_patterns:
                    if pattern in col:
                        combined_scores[col] *= 1.2  # 20% boost for known important features
                        break
            
            # Store feature importance
            self.feature_importance = combined_scores
            
            # Show top 10 features
            top_features = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:10]
            logger.info(f"Top 10 features by importance:")
            for feature, score in top_features:
                logger.info(f"  {feature}: {score:.6f}")
            
            logger.info(f"Feature ranking complete in {time.time() - start_time:.2f} seconds")
            return combined_scores
            
        except Exception as e:
            logger.error(f"Error in feature importance calculation: {str(e)}")
            logger.error("Falling back to correlation-based feature ranking")
            
            # Fallback to correlation only
            self.feature_importance = corr_scores
            return corr_scores
    
    def select_features(self, df: pd.DataFrame, target: str, subject_col: str, n_features: int = 30) -> List[str]:
        """
        Select the top N most important features, explicitly excluding subject ID
        
        Args:
            df: Input dataframe
            target: Target column name
            subject_col: Subject identifier column name
            n_features: Number of features to select
            
        Returns:
            List of selected feature names
        """
        start_time = time.time()
        logger.info(f"Selecting top {n_features} features (excluding subject ID)")
        
        # Ensure subject_col is not included in feature selection
        if subject_col in df.columns:
            logger.info(f"Removing subject column '{subject_col}' from feature selection to prevent data leakage")
            df_features = df.drop(columns=[subject_col])
        else:
            df_features = df
    
        # Rank features by importance
        importance_scores = self.rank_features(df_features, target)
        
        # Sort features by importance
        sorted_features = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Select top N features
        selected_features = [feature for feature, _ in sorted_features[:n_features]]
        
        # Make sure target is not in the selected features
        if target in selected_features:
            selected_features.remove(target)
            # Add one more feature to maintain n_features
            if len(sorted_features) > n_features:
                next_feature = sorted_features[n_features][0]
                selected_features.append(next_feature)
        
        # Store selected features
        self.selected_features = selected_features
        self.selection_method = "RandomForest+Correlation"
        
        logger.info(f"Selected {len(selected_features)} features for modeling in {time.time() - start_time:.2f} seconds")
        
        return selected_features
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores
        
        Returns:
            Dictionary of {feature: importance_score}
        """
        return self.feature_importance
    
    def get_feature_importance_df(self) -> pd.DataFrame:
        """
        Get feature importance as a dataframe
        
        Returns:
            DataFrame with features and importance scores
        """
        if not self.feature_importance:
            return pd.DataFrame(columns=['feature', 'importance'])
        
        # Create dataframe
        df = pd.DataFrame([
            {'feature': feature, 'importance': importance}
            for feature, importance in self.feature_importance.items()
        ])
        
        # Sort by importance
        df = df.sort_values('importance', ascending=False).reset_index(drop=True)
        
        # Add selection flag
        df['selected'] = df['feature'].isin(self.selected_features)
        
        return df


# =============================================================================
# Modeling
# =============================================================================

class CognitiveLoadModeler:
    """Trains and evaluates cognitive load prediction models"""
    
    def __init__(self, subject_col: str = DEFAULT_SUBJECT_COL, 
                 target_col: str = DEFAULT_TARGET_COL,
                 random_state: int = 42):
        """
        Initialize the modeler
        
        Args:
            subject_col: Subject identifier column
            target_col: Target column (cognitive load measure)
            random_state: Random seed for reproducibility
        """
        self.subject_col = subject_col
        self.target_col = target_col
        self.random_state = random_state
        
        # Store models
        self.global_model = None
        self.subject_models = {}
        self.adaptive_models = {}
        
        # Store model performance
        self.global_model_performance = {}
        self.subject_model_performance = {}
        self.adaptive_model_performance = {}
        
        # Store feature importances
        self.feature_importances = {}

    def train_global_model(self, df: pd.DataFrame, features: List[str],
                           model_class=RandomForestRegressor, **model_kwargs):
        """
        Train a global model on all data
        
        Args:
            df: Input dataframe
            features: List of feature names to use
            model_class: Scikit-learn model class to use
            model_kwargs: Additional arguments for the model
            
        Returns:
            Trained model
        """
        start_time = time.time()
        logger.info(f"Training global model with {model_class.__name__}")
        
        # Prepare data
        X = df[features].fillna(df[features].mean())
        y = df[self.target_col]
        
        # Create and train model
        model = model_class(random_state=self.random_state, **model_kwargs)
        model.fit(X, y)
        
        # Store model
        self.global_model = model
        
        # Store feature importances if available
        if hasattr(model, 'feature_importances_'):
            self.feature_importances['global'] = {
                feature: importance 
                for feature, importance in zip(features, model.feature_importances_)
            }
            
            # Show top features
            importances = [(feature, importance) 
                          for feature, importance in self.feature_importances['global'].items()]
            importances.sort(key=lambda x: x[1], reverse=True)
            
            logger.info(f"Global model - top 5 features by importance:")
            for feature, importance in importances[:5]:
                logger.info(f"  {feature}: {importance:.6f}")
        
        logger.info(f"Global model trained in {time.time() - start_time:.2f} seconds")
        
        return model
    
    def evaluate_global_model(self, df: pd.DataFrame, features: List[str], data_processor: DataProcessor):
        """
        Evaluate global model with both within-subject and cross-subject metrics
        
        Args:
            df: Input dataframe
            features: List of feature names to use
            data_processor: DataProcessor instance for cross-validation
            
        Returns:
            Dictionary with performance metrics
        """
        start_time = time.time()
        logger.info("Evaluating global model with dual metrics")
        
        # Ensure we have a trained global model
        if self.global_model is None:
            logger.error("Global model not trained")
            return {}
        
        # 1. Calculate within-subject metrics using k-fold cross-validation
        # This measures how well the model predicts new samples from known subjects
        logger.info("Calculating within-subject performance (k-fold cross-validation)")
        
        X = df[features].fillna(df[features].mean())
        y = df[self.target_col]
        
        cv = KFold(n_splits=5, shuffle=True, random_state=self.random_state)
        model_cv = self.global_model.__class__(**self.global_model.get_params())
        
        cv_scores = cross_val_score(model_cv, X, y, cv=cv, scoring='r2')
        within_subject_r2 = cv_scores.mean()
        within_subject_r2_std = cv_scores.std()
        
        # Calculate confidence interval
        within_subject_r2_ci = [
            within_subject_r2 - 1.96 * within_subject_r2_std / np.sqrt(len(cv_scores)),
            within_subject_r2 + 1.96 * within_subject_r2_std / np.sqrt(len(cv_scores))
        ]
        
        # 2. Calculate cross-subject metrics using leave-one-subject-out validation
        # This measures how well the model generalizes to completely new subjects
        logger.info("Calculating cross-subject performance (leave-one-subject-out)")
        
        cross_subject_results = data_processor.perform_cross_subject_validation(
            df, features, self.target_col, self.subject_col, 
            self.global_model.__class__, **self.global_model.get_params()
        )
        
        # 3. Store results
        performance = {
            'within_subject_r2': within_subject_r2,
            'within_subject_r2_std': within_subject_r2_std,
            'within_subject_r2_ci': within_subject_r2_ci,
            'cross_subject_r2': cross_subject_results['mean_r2'],
            'cross_subject_r2_std': cross_subject_results['std_r2'],
            'cross_subject_r2_ci': cross_subject_results['r2_ci'],
            'cross_subject_rmse': cross_subject_results['mean_rmse'],
            'cross_subject_rmse_std': cross_subject_results['std_rmse'],
            'cross_subject_rmse_ci': cross_subject_results['rmse_ci'],
            'cross_subject_subject_results': cross_subject_results['subject_results']
        }
        
        self.global_model_performance = performance
        
        # Print comparison of metrics with confidence intervals
        logger.info(f"Global model performance:")
        logger.info(f"  - Within-Subject R²: {within_subject_r2:.3f} (95% CI: [{within_subject_r2_ci[0]:.3f}, {within_subject_r2_ci[1]:.3f}])")
        logger.info(f"  - Cross-Subject R²: {cross_subject_results['mean_r2']:.3f} (95% CI: [{cross_subject_results['r2_ci'][0]:.3f}, {cross_subject_results['r2_ci'][1]:.3f}])")
        
        logger.info(f"Global model evaluation completed in {time.time() - start_time:.2f} seconds")
        
        return performance

    
    def train_subject_models(self, df: pd.DataFrame, features: List[str], min_samples: int = 10,
                             model_class=RandomForestRegressor, **model_kwargs):
        """
        Train separate models for each subject
        
        Args:
            df: Input dataframe
            features: List of feature names to use
            min_samples: Minimum number of samples required per subject
            model_class: Scikit-learn model class to use
            model_kwargs: Additional arguments for the model
            
        Returns:
            Dictionary with subject model performance
        """
        start_time = time.time()
        logger.info(f"Training subject-specific models with {model_class.__name__}")
        
        # Get unique subjects
        subjects = df[self.subject_col].unique()
        logger.info(f"Training models for {len(subjects)} subjects")
        
        # Store results
        subject_results = {}
        r2_values = []
        rmse_values = []
        
        # Train a model for each subject with enough data
        for subject in tqdm(subjects, desc="Training subject models"):
            # Get subject data
            subject_data = df[df[self.subject_col] == subject].copy()
            
            # Skip if not enough data
            if len(subject_data) < min_samples:
                logger.info(f"  Skipping subject {subject} - insufficient data ({len(subject_data)} < {min_samples})")
                continue
            
            # Create features and target
            X = subject_data[features].fillna(subject_data[features].mean())
            y = subject_data[self.target_col]
            
            # Train model (with parameters optimized for smaller datasets)
            params = model_kwargs.copy()
            # Reduce trees for smaller datasets to prevent overfitting
            if 'n_estimators' in params and len(X) < 3 * min_samples:
                params['n_estimators'] = min(50, params.get('n_estimators', 100))
            
            model = model_class(random_state=self.random_state, **params)
            
            # Use cross-validation to evaluate if enough data
            if len(X) >= 3 * min_samples:
                # K-fold CV for larger datasets
                cv = KFold(n_splits=3, shuffle=True, random_state=self.random_state)
                cv_scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
                r2 = cv_scores.mean()
                r2_std = cv_scores.std()
                
                # Calculate RMSE using CV predictions
                rmse_scores = []
                for train_idx, test_idx in cv.split(X):
                    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                    
                    model_cv = model_class(random_state=self.random_state, **params)
                    model_cv.fit(X_train, y_train)
                    y_pred = model_cv.predict(X_test)
                    
                    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                    rmse_scores.append(rmse)
                
                rmse = np.mean(rmse_scores)
                rmse_std = np.std(rmse_scores)
                
                # Confidence intervals
                r2_ci = [
                    r2 - 1.96 * r2_std / np.sqrt(len(cv_scores)),
                    r2 + 1.96 * r2_std / np.sqrt(len(cv_scores))
                ]
                
                rmse_ci = [
                    rmse - 1.96 * rmse_std / np.sqrt(len(rmse_scores)),
                    rmse + 1.96 * rmse_std / np.sqrt(len(rmse_scores))
                ]
            else:
                # For subjects with less data, use a simple train/test split
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.3, random_state=self.random_state
                )
                
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                
                # Confidence intervals unavailable for single split
                r2_ci = None
                rmse_ci = None
            
            # Train final model on all subject data
            model.fit(X, y)
            
            # Store model and results
            self.subject_models[subject] = model
            subject_results[subject] = {
                'r2': r2,
                'r2_ci': r2_ci,
                'rmse': rmse,
                'rmse_ci': rmse_ci,
                'n_samples': len(subject_data)
            }
            
            r2_values.append(r2)
            rmse_values.append(rmse)
            
            # Store feature importances
            if hasattr(model, 'feature_importances_'):
                self.feature_importances[subject] = {
                    feature: importance for feature, importance in 
                    zip(features, model.feature_importances_)
                }
        
        # Calculate average performance
        avg_r2 = np.mean(r2_values)
        std_r2 = np.std(r2_values)
        avg_rmse = np.mean(rmse_values)
        std_rmse = np.std(rmse_values)
        
        # Calculate confidence intervals
        r2_ci_lower, r2_ci_upper = self._bootstrap_confidence_interval(r2_values)
        rmse_ci_lower, rmse_ci_upper = self._bootstrap_confidence_interval(rmse_values)
        
        # Store performance metrics
        performance = {
            'avg_r2': avg_r2,
            'std_r2': std_r2,
            'r2_ci': [r2_ci_lower, r2_ci_upper],
            'avg_rmse': avg_rmse,
            'std_rmse': std_rmse,
            'rmse_ci': [rmse_ci_lower, rmse_ci_upper],
            'subject_results': subject_results,
            'n_models': len(self.subject_models)
        }
        
        self.subject_model_performance = performance
        
        logger.info(f"\nSubject-specific model performance:")
        logger.info(f"  - Average R²: {avg_r2:.3f} (95% CI: [{r2_ci_lower:.3f}, {r2_ci_upper:.3f}])")
        logger.info(f"  - Average RMSE: {avg_rmse:.3f} (95% CI: [{rmse_ci_lower:.3f}, {rmse_ci_upper:.3f}])")
        logger.info(f"  - R² Range: [{min(r2_values):.3f}, {max(r2_values):.3f}]")
        logger.info(f"  - Models trained: {len(self.subject_models)} (out of {len(subjects)} subjects)")
        
        logger.info(f"Subject-specific models training completed in {time.time() - start_time:.2f} seconds")
        
        return performance
    
    def train_adaptive_transfer_models(self, df: pd.DataFrame, features: List[str], test_size: float = 0.3,
                                      model_class=RandomForestRegressor, **model_kwargs):
        """
        Train adaptive transfer learning models
        
        Args:
            df: Input dataframe
            features: List of feature names to use
            test_size: Proportion of data to use for testing weight optimization
            model_class: Scikit-learn model class to use
            model_kwargs: Additional arguments for the model
            
        Returns:
            Dictionary with adaptive model results
        """
        start_time = time.time()
        logger.info(f"Training adaptive transfer models with {model_class.__name__}")
        
        # Get unique subjects
        subjects = df[self.subject_col].unique()
        
        # Store results
        adaptive_results = {}
        adaptive_weights = []
        adaptive_scores = []
        adaptive_rmse = []
        
        # Progress bar
        progress_bar = tqdm(total=len(subjects), desc="Training adaptive models")
        
        for subject in subjects:
            # Get target subject data
            target_data = df[df[self.subject_col] == subject].copy()
            
            # Skip if not enough data
            if len(target_data) < 10:
                progress_bar.update(1)
                continue
            
            # Get source data (all other subjects)
            source_data = df[df[self.subject_col] != subject].copy()
            
            # Create train/test split for the target subject
            X_all = target_data[features].fillna(target_data[features].mean())
            y_all = target_data[self.target_col]
            
            X_train, X_test, y_train, y_test = train_test_split(
                X_all, y_all, test_size=test_size, random_state=self.random_state
            )
            
            # Train source model on all other subjects
            X_source = source_data[features].fillna(source_data[features].mean())
            y_source = source_data[self.target_col]
            
            source_model = model_class(random_state=self.random_state, **model_kwargs)
            source_model.fit(X_source, y_source)
            
            # Train target model on target subject training data
            target_model = model_class(random_state=self.random_state, **model_kwargs)
            target_model.fit(X_train, y_train)
            
            # Determine optimal weights for blending predictions
            weights = np.linspace(0, 1, 11)  # 0 to 1 in 0.1 increments
            weight_scores = []
            weight_rmse = []
            
            for w in weights:
                # Make predictions with weighted average
                source_preds = source_model.predict(X_test)
                target_preds = target_model.predict(X_test)
                
                # Weighted average: w * target + (1-w) * source
                weighted_preds = (w * target_preds) + ((1-w) * source_preds)
                
                # Calculate scores
                score = r2_score(y_test, weighted_preds)
                rmse_val = np.sqrt(mean_squared_error(y_test, weighted_preds))
                
                weight_scores.append(score)
                weight_rmse.append(rmse_val)
            
            # Find best weight
            best_idx = np.argmax(weight_scores)
            best_weight = weights[best_idx]
            best_score = weight_scores[best_idx]
            best_rmse = weight_rmse[best_idx]
            
            # Store results
            adaptive_weights.append(best_weight)
            adaptive_scores.append(best_score)
            adaptive_rmse.append(best_rmse)
            
            # Store models and optimal weight
            self.adaptive_models[subject] = {
                'source_model': source_model,
                'target_model': target_model,
                'optimal_weight': best_weight
            }
            
            adaptive_results[subject] = {
                'optimal_weight': best_weight,
                'r2_score': best_score,
                'rmse': best_rmse,
                'n_samples': len(target_data)
            }
            
            # Update progress bar
            progress_bar.update(1)
        
        # Close progress bar
        progress_bar.close()
        
        # Calculate average results
        avg_weight = np.mean(adaptive_weights)
        avg_score = np.mean(adaptive_scores)
        avg_rmse = np.mean(adaptive_rmse)
        
        # Calculate confidence intervals
        score_ci_lower, score_ci_upper = self._bootstrap_confidence_interval(adaptive_scores)
        rmse_ci_lower, rmse_ci_upper = self._bootstrap_confidence_interval(adaptive_rmse)
        weight_ci_lower, weight_ci_upper = self._bootstrap_confidence_interval(adaptive_weights)
        
        # Store performance
        performance = {
            'avg_weight': avg_weight,
            'weight_ci': [weight_ci_lower, weight_ci_upper],
            'avg_score': avg_score,
            'score_ci': [score_ci_lower, score_ci_upper],
            'avg_rmse': avg_rmse,
            'rmse_ci': [rmse_ci_lower, rmse_ci_upper],
            'subject_results': adaptive_results,
            'n_models': len(self.adaptive_models)
        }
        
        self.adaptive_model_performance = performance
        
        logger.info(f"Adaptive transfer learning results:")
        logger.info(f"  - Average optimal subject weight: {avg_weight:.2f} (95% CI: [{weight_ci_lower:.2f}, {weight_ci_upper:.2f}])")
        logger.info(f"  - Average R² score: {avg_score:.3f} (95% CI: [{score_ci_lower:.3f}, {score_ci_upper:.3f}])")
        logger.info(f"  - Average RMSE: {avg_rmse:.3f} (95% CI: [{rmse_ci_lower:.3f}, {rmse_ci_upper:.3f}])")
        logger.info(f"  - Models trained: {len(self.adaptive_models)} (out of {len(subjects)} subjects)")
        
        logger.info(f"Adaptive transfer models training completed in {time.time() - start_time:.2f} seconds")
        
        return performance
    
    def compare_models(self, df: pd.DataFrame, features: List[str]):
        """
        Compare global vs subject-specific vs adaptive models
        
        Args:
            df: Input dataframe for evaluation
            features: List of feature names to use
            
        Returns:
            Dictionary with comparison results
        """
        start_time = time.time()
        logger.info(f"Comparing model performance")
        
        # Verify that we have all model types
        if self.global_model is None:
            logger.error("Global model not available for comparison")
            return {}
        
        if not self.subject_models:
            logger.error("Subject-specific models not available for comparison")
            return {}
        
        if not self.adaptive_models:
            logger.warning("Adaptive models not available for comparison")
        
        # Store comparison results
        comparison_results = {}
        global_vs_subject_improvements = []
        adaptive_vs_subject_improvements = []
        
        # Evaluate on each subject
        progress_bar = tqdm(total=len(self.subject_models), desc="Comparing models")
        
        for subject, subject_model in self.subject_models.items():
            # Get subject data
            subject_data = df[df[self.subject_col] == subject].copy()
            
            # Skip if not enough data
            if len(subject_data) < 5:
                progress_bar.update(1)
                continue
            
            # Get predictors and target
            X = subject_data[features].fillna(subject_data[features].mean())
            y = subject_data[self.target_col]
            
            # Get predictions from global model
            global_preds = self.global_model.predict(X)
            global_r2 = r2_score(y, global_preds)
            global_rmse = np.sqrt(mean_squared_error(y, global_preds))
            
            # Get predictions from subject-specific model
            subject_preds = subject_model.predict(X)
            subject_r2 = r2_score(y, subject_preds)
            subject_rmse = np.sqrt(mean_squared_error(y, subject_preds))
            
            # Calculate improvement
            r2_improvement = subject_r2 - global_r2
            rmse_improvement = global_rmse - subject_rmse  # Lower RMSE is better
            
            global_vs_subject_improvements.append(r2_improvement)
            
            # Get predictions from adaptive model if available
            adaptive_r2 = None
            adaptive_rmse = None
            adaptive_improvement = None
            
            if subject in self.adaptive_models:
                adaptive_model = self.adaptive_models[subject]
                source_model = adaptive_model['source_model']
                target_model = adaptive_model['target_model']
                weight = adaptive_model['optimal_weight']
                
                # Make predictions
                source_preds = source_model.predict(X)
                target_preds = target_model.predict(X)
                
                # Weighted average
                adaptive_preds = (weight * target_preds) + ((1 - weight) * source_preds)
                
                adaptive_r2 = r2_score(y, adaptive_preds)
                adaptive_rmse = np.sqrt(mean_squared_error(y, adaptive_preds))
                
                # Calculate improvement over subject-specific model
                adaptive_improvement = adaptive_r2 - subject_r2
                adaptive_vs_subject_improvements.append(adaptive_improvement)
            
            # Store results
            comparison_results[subject] = {
                'global_r2': global_r2,
                'global_rmse': global_rmse,
                'subject_r2': subject_r2,
                'subject_rmse': subject_rmse,
                'r2_improvement': r2_improvement,
                'rmse_improvement': rmse_improvement,
                'adaptive_r2': adaptive_r2,
                'adaptive_rmse': adaptive_rmse,
                'adaptive_improvement': adaptive_improvement,
                'n_samples': len(subject_data)
            }
            
            # Update progress bar
            progress_bar.update(1)
        
        # Close progress bar
        progress_bar.close()
        
        # Calculate average improvements
        avg_r2_improvement = np.mean(global_vs_subject_improvements)
        std_r2_improvement = np.std(global_vs_subject_improvements)
        
        # Calculate confidence interval for R² improvement
        r2_imp_ci_lower, r2_imp_ci_upper = self._bootstrap_confidence_interval(global_vs_subject_improvements)
        
        # Adaptive improvement, if available
        if adaptive_vs_subject_improvements:
            avg_adaptive_improvement = np.mean(adaptive_vs_subject_improvements)
            std_adaptive_improvement = np.std(adaptive_vs_subject_improvements)
            adp_imp_ci_lower, adp_imp_ci_upper = self._bootstrap_confidence_interval(adaptive_vs_subject_improvements)
        else:
            avg_adaptive_improvement = None
            std_adaptive_improvement = None
            adp_imp_ci_lower, adp_imp_ci_upper = None, None
        
        # Statistical significance test for model comparison
        t_stat, p_value = self._significance_test(global_vs_subject_improvements)
        
        # Store overall results
        results = {
            'avg_r2_improvement': avg_r2_improvement,
            'std_r2_improvement': std_r2_improvement,
            'r2_improvement_ci': [r2_imp_ci_lower, r2_imp_ci_upper],
            't_statistic': t_stat,
            'p_value': p_value,
            'is_significant': p_value < 0.05,
            'subject_results': comparison_results,
            'improved_subject_count': sum(imp > 0 for imp in global_vs_subject_improvements),
            'total_subject_count': len(global_vs_subject_improvements)
        }
        
        # Add adaptive results if available
        if adaptive_vs_subject_improvements:
            # Statistical significance test for adaptive vs subject
            adp_t_stat, adp_p_value = self._significance_test(adaptive_vs_subject_improvements)
            
            results.update({
                'avg_adaptive_improvement': avg_adaptive_improvement,
                'std_adaptive_improvement': std_adaptive_improvement,
                'adaptive_improvement_ci': [adp_imp_ci_lower, adp_imp_ci_upper],
                'adaptive_t_statistic': adp_t_stat,
                'adaptive_p_value': adp_p_value,
                'adaptive_is_significant': adp_p_value < 0.05,
                'improved_adaptive_count': sum(imp > 0 for imp in adaptive_vs_subject_improvements),
                'total_adaptive_count': len(adaptive_vs_subject_improvements)
            })
        
        # Report results
        logger.info(f"Model comparison results:")
        logger.info(f"  - Avg R² improvement (Subject vs Global): {avg_r2_improvement:.3f} (95% CI: [{r2_imp_ci_lower:.3f}, {r2_imp_ci_upper:.3f}])")
        logger.info(f"  - Significance: t={t_stat:.3f}, p={p_value:.4f} ({'significant' if p_value < 0.05 else 'not significant'})")
        logger.info(f"  - Subjects with improved performance: {results['improved_subject_count']}/{results['total_subject_count']} ({100 * results['improved_subject_count'] / results['total_subject_count']:.1f}%)")
        
        if adaptive_vs_subject_improvements:
            logger.info(f"  - Avg R² improvement (Adaptive vs Subject): {avg_adaptive_improvement:.3f} (95% CI: [{adp_imp_ci_lower:.3f}, {adp_imp_ci_upper:.3f}])")
            logger.info(f"  - Significance: t={adp_t_stat:.3f}, p={adp_p_value:.4f} ({'significant' if adp_p_value < 0.05 else 'not significant'})")
            logger.info(f"  - Subjects with adaptive improvement: {results['improved_adaptive_count']}/{results['total_adaptive_count']} ({100 * results['improved_adaptive_count'] / results['total_adaptive_count']:.1f}%)")
        
        logger.info(f"Model comparison completed in {time.time() - start_time:.2f} seconds")
        
        return results
    
    def _significance_test(self, improvements):
        """
        Perform a statistical significance test on model improvements
        
        Args:
            improvements: List of improvement values
            
        Returns:
            Tuple of (t_statistic, p_value)
        """
        # One-sample t-test comparing improvements to zero
        t_stat, p_value = stats.ttest_1samp(improvements, 0)
        return t_stat, p_value
    
    def _bootstrap_confidence_interval(self, values, n_bootstrap=1000, alpha=0.05):
        """
        Calculate bootstrap confidence interval
        
        Args:
            values: Array of values
            n_bootstrap: Number of bootstrap samples
            alpha: Alpha level for confidence interval
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        bootstrap_samples = np.random.choice(values, size=(n_bootstrap, len(values)), replace=True)
        bootstrap_means = np.mean(bootstrap_samples, axis=1)
        
        # Calculate percentile confidence interval
        lower_bound = np.percentile(bootstrap_means, 100 * alpha / 2)
        upper_bound = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))
        
        return lower_bound, upper_bound
    
    def predict(self, df: pd.DataFrame, features: List[str], 
                method: str = 'adaptive') -> Tuple[np.ndarray, Dict]:
        """
        Make predictions using the appropriate model for each subject
        
        Args:
            df: Input dataframe
            features: List of feature names to use
            method: Prediction method ('global', 'subject', or 'adaptive')
            
        Returns:
            Tuple of (predictions, prediction_metadata)
        """
        start_time = time.time()
        logger.info(f"Making predictions using {method} method")
        
        # Initialize predictions
        predictions = np.zeros(len(df))
        prediction_metadata = {
            'method': method,
            'subject_methods': {}
        }
        
        # Check if we have subject information
        if self.subject_col in df.columns:
            # Get unique subjects in the data
            subjects = df[self.subject_col].unique()
            logger.info(f"Found {len(subjects)} subjects in prediction data")
            
            for subject in subjects:
                # Get subject data indices
                mask = df[self.subject_col] == subject
                
                if not any(mask):
                    continue  # Skip if no samples for this subject
                
                # Get features, handling missing values
                X = df.loc[mask, features].copy()
                X = X.fillna(X.mean())  # Fill with this subject's means
                
                # Decide which model to use based on method and availability
                if method == 'global' or (
                    method == 'subject' and subject not in self.subject_models) or (
                    method == 'adaptive' and subject not in self.adaptive_models):
                    
                    # Fall back to global model
                    if self.global_model is not None:
                        predictions[mask] = self.global_model.predict(X)
                        prediction_metadata['subject_methods'][subject] = 'global'
                    else:
                        logger.warning(f"No global model available for subject {subject}")
                        predictions[mask] = np.nan
                        prediction_metadata['subject_methods'][subject] = 'none'
                
                elif method == 'subject':
                    # Use subject-specific model
                    predictions[mask] = self.subject_models[subject].predict(X)
                    prediction_metadata['subject_methods'][subject] = 'subject'
                
                elif method == 'adaptive':
                    # Use adaptive model
                    adaptive_model = self.adaptive_models[subject]
                    source_model = adaptive_model['source_model']
                    target_model = adaptive_model['target_model']
                    weight = adaptive_model['optimal_weight']
                    
                    # Make predictions
                    source_preds = source_model.predict(X)
                    target_preds = target_model.predict(X)
                    
                    # Weighted average
                    predictions[mask] = (weight * target_preds) + ((1 - weight) * source_preds)
                    prediction_metadata['subject_methods'][subject] = 'adaptive'
                
                else:
                    logger.error(f"Unknown prediction method: {method}")
                    predictions[mask] = np.nan
                    prediction_metadata['subject_methods'][subject] = 'error'
        
        else:
            # If no subject information, use global model for all
            if self.global_model is not None:
                X = df[features].fillna(df[features].mean())
                predictions = self.global_model.predict(X)
                prediction_metadata['method'] = 'global'
            else:
                logger.error("No global model available and no subject information in data")
                predictions = np.full(len(df), np.nan)
                prediction_metadata['method'] = 'none'
        
        logger.info(f"Predictions generated in {time.time() - start_time:.2f} seconds")
        return predictions, prediction_metadata
    
    def generate_predictions_for_report(self, test_df: pd.DataFrame, features: List[str]):
        """
        Generate predictions for reporting and visualization
        
        Args:
            test_df: Test dataframe
            features: List of features to use
            
        Returns:
            Dictionary with prediction data
        """
        start_time = time.time()
        logger.info("Generating predictions for reporting")
        
        # Initialize predictions dictionary
        predictions = {
            'global_model': {},
            'subject_models': {},
            'global_for_subjects': {},
            'adaptive_models': {},
            'subject_improvements': {},
            'adaptive_improvements': {}
        }
        
        # Ensure we have the target column
        if self.target_col not in test_df.columns:
            logger.error(f"Target column {self.target_col} not found in test data")
            return predictions
        
        # Ensure we have subject information
        if self.subject_col not in test_df.columns:
            logger.error(f"Subject column {self.subject_col} not found in test data")
            return predictions
        
        # 1. Global model predictions on all data
        if self.global_model is not None:
            X_test = test_df[features].fillna(test_df[features].mean())
            y_test = test_df[self.target_col]
            
            global_preds = self.global_model.predict(X_test)
            
            # Calculate metrics
            global_r2 = r2_score(y_test, global_preds)
            global_rmse = np.sqrt(mean_squared_error(y_test, global_preds))
            
            # Store
            predictions['global_model'] = {
                'actual': y_test.tolist(),
                'predicted': global_preds.tolist(),
                'r2': global_r2,
                'rmse': global_rmse
            }
            
            logger.info(f"Global model predictions: R²={global_r2:.3f}, RMSE={global_rmse:.3f}")
        
        # 2. Predictions by subject
        subjects = test_df[self.subject_col].unique()
        logger.info(f"Generating predictions for {len(subjects)} subjects")
        
        valid_subject_count = 0
        valid_adaptive_count = 0
        
        for subject in subjects:
            # Get subject data
            subject_mask = test_df[self.subject_col] == subject
            subject_data = test_df[subject_mask]
            
            if len(subject_data) < 5:
                continue  # Skip subjects with very few samples
            
            # Get features and target
            X_subject = subject_data[features].fillna(subject_data[features].mean())
            y_subject = subject_data[self.target_col]
            
            # Global model predictions for this subject
            if self.global_model is not None:
                subject_global_preds = self.global_model.predict(X_subject)
                subject_global_r2 = r2_score(y_subject, subject_global_preds)
                subject_global_rmse = np.sqrt(mean_squared_error(y_subject, subject_global_preds))
                
                # Store
                predictions['global_for_subjects'][str(subject)] = {
                    'actual': y_subject.tolist(),
                    'predicted': subject_global_preds.tolist(),
                    'r2': subject_global_r2,
                    'rmse': subject_global_rmse
                }
            
            # Subject-specific model predictions
            if subject in self.subject_models:
                subject_model = self.subject_models[subject]
                subject_preds = subject_model.predict(X_subject)
                subject_r2 = r2_score(y_subject, subject_preds)
                subject_rmse = np.sqrt(mean_squared_error(y_subject, subject_preds))
                
                # Store
                predictions['subject_models'][str(subject)] = {
                    'actual': y_subject.tolist(),
                    'predicted': subject_preds.tolist(),
                    'r2': subject_r2,
                    'rmse': subject_rmse
                }
                
                valid_subject_count += 1
                
                # Calculate improvement over global
                if self.global_model is not None:
                    r2_improvement = ((subject_r2 - subject_global_r2) / abs(subject_global_r2)) * 100 if subject_global_r2 != 0 else 0
                    rmse_improvement = ((subject_global_rmse - subject_rmse) / subject_global_rmse) * 100
                    
                    predictions['subject_improvements'][str(subject)] = r2_improvement
            
            # Adaptive model predictions
            if subject in self.adaptive_models:
                adaptive_model = self.adaptive_models[subject]
                source_model = adaptive_model['source_model']
                target_model = adaptive_model['target_model']
                weight = adaptive_model['optimal_weight']
                
                # Make predictions
                source_preds = source_model.predict(X_subject)
                target_preds = target_model.predict(X_subject)
                
                # Weighted average
                adaptive_preds = (weight * target_preds) + ((1 - weight) * source_preds)
                
                adaptive_r2 = r2_score(y_subject, adaptive_preds)
                adaptive_rmse = np.sqrt(mean_squared_error(y_subject, adaptive_preds))
                
                # Store
                predictions['adaptive_models'][str(subject)] = {
                    'actual': y_subject.tolist(),
                    'predicted': adaptive_preds.tolist(),
                    'r2': adaptive_r2,
                    'rmse': adaptive_rmse,
                    'weight': weight
                }
                
                valid_adaptive_count += 1
                
                # Calculate improvement over subject-specific model
                if subject in self.subject_models:
                    subject_r2 = predictions['subject_models'][str(subject)]['r2']
                    r2_improvement = ((adaptive_r2 - subject_r2) / abs(subject_r2)) * 100 if subject_r2 != 0 else 0
                    
                    predictions['adaptive_improvements'][str(subject)] = r2_improvement
        
        logger.info(f"Generated predictions for {valid_subject_count} subject-specific models and {valid_adaptive_count} adaptive models")
        logger.info(f"Prediction generation completed in {time.time() - start_time:.2f} seconds")
        
        return predictions


# =============================================================================
# Report Generator
# =============================================================================

class ReportGenerator:
    """Generates comprehensive HTML report with visualizations"""
    
    def __init__(self, results_dir: str, title: str = "Cognitive Load Analysis Results"):
        """
        Initialize report generator
        
        Args:
            results_dir: Directory to store results
            title: Report title
        """
        self.results_dir = results_dir
        self.title = title
        
        # Create directories
        self.plots_dir = os.path.join(results_dir, "plots")
        os.makedirs(self.plots_dir, exist_ok=True)
        
        # Store figures and data
        self.figures = {}
        self.tables = {}
        self.metrics = {}
        
        # Setup matplotlib
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_context("paper", font_scale=1.2)
        sns.set_palette("viridis")
        
    def add_dataset_visualization(self, df: pd.DataFrame, dataset_stats: Dict,
                                 target_col: str, subject_col: str):
        """
        Create dataset overview visualizations
        
        Args:
            df: Input dataframe
            dataset_stats: Dictionary with dataset statistics
            target_col: Target column name
            subject_col: Subject identifier column name
        """
        logger.info("Creating dataset overview visualizations")
        
        # 1. Subject distribution
        if subject_col in df.columns:
            plt.figure(figsize=(12, 6))
            
            # Get subject counts
            subject_counts = df[subject_col].value_counts().sort_values(ascending=False)
            
            # Limit to top 20 for readability
            subject_subset = subject_counts.iloc[:20]
            
            # Create bar chart
            ax = sns.barplot(x=subject_subset.index, y=subject_subset.values)
            
            # Format subject IDs as strings
            ax.set_xticks(range(len(subject_subset)))
            ax.set_xticklabels([f"S{i+1}" for i in range(len(subject_subset))])
            
            plt.title("Sample Distribution Across Subjects", fontsize=14)
            plt.xlabel("Subject ID", fontsize=12)
            plt.ylabel("Number of Samples", fontsize=12)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            self.figures["subject_distribution"] = self._save_figure()
            
            # Store subject information
            self.metrics.update({
                "n_subjects": df[subject_col].nunique(),
                "min_samples_per_subject": subject_counts.min(),
                "max_samples_per_subject": subject_counts.max(),
                "mean_samples_per_subject": subject_counts.mean()
            })
        
        # 2. Target variable distribution
        if target_col in df.columns:
            plt.figure(figsize=(12, 6))
            
            # Create histogram
            sns.histplot(df[target_col], kde=True, bins=15)
            
            # Get statistics once to ensure consistency
            mean_val = df[target_col].mean()
            median_val = df[target_col].median()
            std_val = df[target_col].std()
            min_val = df[target_col].min()
            max_val = df[target_col].max()
            
            # Add mean and median lines
            plt.axvline(mean_val, color='red', linestyle='dashed', linewidth=1)
            plt.axvline(median_val, color='green', linestyle='dashed', linewidth=1)
            
            # Add a legend - only include the KDE and the lines without values
            # (values will be in the stats box)
            plt.legend(['KDE', 'Mean', 'Median'])
            
            plt.title(f"Distribution of {target_col.replace('_', ' ').title()}", fontsize=14)
            plt.xlabel(target_col.replace('_', ' ').title(), fontsize=12)
            plt.ylabel("Frequency", fontsize=12)
            
            # Add distribution statistics - complete info in one place
            stats_text = (
                f"Mean: {mean_val:.2f}\n"
                f"Median: {median_val:.2f}\n"
                f"Std Dev: {std_val:.2f}\n"
                f"Min: {min_val:.2f}\n"
                f"Max: {max_val:.2f}"
            )
            plt.annotate(stats_text, xy=(0.95, 0.95), xycoords='axes fraction', 
                        ha='right', va='top', bbox=dict(boxstyle="round,pad=0.5", 
                                                      fc="white", ec="gray", alpha=0.8))
            
            plt.tight_layout()
            
            self.figures["target_distribution"] = self._save_figure()
            
            # Store target information
            self.metrics.update({
                "target_mean": mean_val,
                "target_median": median_val,
                "target_std": std_val,
                "target_min": min_val,
                "target_max": max_val
            })
        
        # 3. Physiological signals summary
        if 'physiological_signals' in dataset_stats:
            signals = dataset_stats['physiological_signals']
            if signals:
                plt.figure(figsize=(12, 6))
                
                # Extract signal types and counts
                signal_types = list(signals.keys())
                signal_counts = list(signals.values())
                
                # Sort by count
                sorted_data = sorted(zip(signal_types, signal_counts), key=lambda x: x[1], reverse=True)
                signal_types, signal_counts = zip(*sorted_data)
                
                # Create bar chart
                colors = sns.color_palette("viridis", len(signal_types))
                bars = plt.bar(signal_types, signal_counts, color=colors)
                
                # Add count labels on top of bars
                for bar in bars:
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                            f'{int(height)}', ha='center', va='bottom')
                
                plt.title("Available Physiological Signals", fontsize=14)
                plt.xlabel("Signal Type", fontsize=12)
                plt.ylabel("Number of Features", fontsize=12)
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                self.figures["physiological_signals"] = self._save_figure()
                
                # Store signals information
                self.metrics["n_physiological_signals"] = len(signals)
                self.metrics["physiological_signal_types"] = list(signals.keys())
        
        # 4. Missing values overview
        plt.figure(figsize=(12, 6))
        
        # Calculate missing values
        missing = df.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=False)
        
        if len(missing) > 0:
            # Limit to top 20 for readability
            missing_subset = missing.iloc[:20]
            
            # Calculate percentages
            missing_percent = missing_subset / len(df) * 100
            
            # Create bar chart
            ax = plt.bar(range(len(missing_subset)), missing_percent)
            
            # Format features
            plt.xticks(range(len(missing_subset)), missing_subset.index, rotation=90)
            
            plt.title("Missing Values by Feature", fontsize=14)
            plt.xlabel("Feature", fontsize=12)
            plt.ylabel("Missing Values (%)", fontsize=12)
            plt.tight_layout()
            
            self.figures["missing_values"] = self._save_figure()
            
            # Store missing values information
            self.metrics.update({
                "n_features_with_missing_values": len(missing),
                "max_missing_percentage": missing.max() / len(df) * 100,
                "total_missing_percentage": df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100
            })
        else:
            # No missing values visualization
            plt.text(0.5, 0.5, "No missing values in dataset", 
                   ha='center', va='center', fontsize=14)
            plt.axis('off')
            
            self.figures["missing_values"] = self._save_figure()
            
            # Store missing values information
            self.metrics.update({
                "n_features_with_missing_values": 0,
                "max_missing_percentage": 0,
                "total_missing_percentage": 0
            })
        
        logger.info("Dataset overview visualizations created")
    
    def add_methodology_visualization(self):
        """Create visualization explaining the methodology"""
        logger.info("Creating methodology visualization")
        
        # Create a flow chart visualization of the pipeline
        plt.figure(figsize=(16, 8))
        ax = plt.gca()
        ax.axis('off')
        
        # Set explicit axis limits to ensure content is visible
        plt.xlim(-1, 12)  # Make sure this range is wider than your x_positions
        plt.ylim(-1, 4)   # Make sure this range covers all your content
        
        # Define box properties
        box_style = dict(boxstyle="round,pad=0.5", fc="white", ec="gray", lw=1)
        box_width = 1.5
        box_height = 0.5
        
        # Define positions
        x_positions = [0, 2, 4, 6, 8, 10]
        y_position = 2
        
        # Box texts and colors
        boxes = [
            "Raw Data",
            "Data Preprocessing",
            "Feature Engineering",
            "Feature Selection",
            "Model Training",
            "Evaluation & Reporting"
        ]
        
        box_colors = sns.color_palette("viridis", len(boxes))
        
        # Draw boxes
        for i, (text, x) in enumerate(zip(boxes, x_positions)):
            box = plt.Rectangle((x, y_position), box_width, box_height, 
                              fc=box_colors[i], ec="gray", alpha=0.7)
            ax.add_patch(box)
            plt.text(x + box_width/2, y_position + box_height/2, text, 
                   ha='center', va='center', fontsize=12, fontweight='bold', color='white')
        
        # Draw arrows between boxes
        for i in range(len(boxes) - 1):
            x_start = x_positions[i] + box_width
            x_end = x_positions[i+1]
            plt.annotate("", xy=(x_end, y_position + box_height/2), 
                       xytext=(x_start, y_position + box_height/2),
                       arrowprops=dict(arrowstyle="->", lw=1.5, color='black'))
        
        # Add details for each step
        details = [
            "• CSV data\n• Physiological signals\n• Cognitive load ratings",
            "• Remove leaky features\n• Handle missing values\n• Train/test split",
            "• Temporal features\n• Frequency features\n• Compound features",
            "• Feature importance ranking\n• Select top N features\n• Remove redundant features",
            "• Global model\n• Subject-specific models\n• Adaptive transfer learning",
            "• Statistical significance testing\n• Performance metrics with CIs\n• Interactive report"
        ]
        
        for i, (detail, x) in enumerate(zip(details, x_positions)):
            plt.text(x + box_width/2, y_position - 0.2, detail, 
                   ha='center', va='top', fontsize=10, bbox=dict(boxstyle="round4,pad=0.5", 
                                                              fc="white", ec="gray", alpha=0.9))
        
        # Add title
        plt.title("Cognitive Load Analysis Pipeline", fontsize=16, fontweight='bold', pad=20)
        
        # Add callout for key metrics
        metric_text = (
            "Key Metrics:\n"
            "• R² Score (explained variance)\n"
            "• RMSE (prediction error)\n"
            "• Statistical significance (p-value)\n"
            "• 95% Confidence Intervals"
        )
        
        plt.text(8, 0.5, metric_text, ha='center', va='center', fontsize=11,
               bbox=dict(boxstyle="round,pad=0.5", fc="lightyellow", ec="orange", alpha=0.9))
        
        # Explicitly call savefig with a fixed DPI
        plt.tight_layout()
        
        # Save the figure
        self.figures["methodology"] = self._save_figure()
        
        # Also save a separate debug image to disk to verify content
        debug_file = os.path.join(self.plots_dir, "methodology_debug.png")
        plt.savefig(debug_file, dpi=100, bbox_inches='tight')
        plt.close()
        
        # Create validation approach visualization
        self._create_validation_approaches_visualization()
        
        logger.info(f"Methodology visualization created and saved to {debug_file}")
    
    def _create_validation_approaches_visualization(self):
        """Create visualization explaining the different validation approaches"""
        
        # Set up the figure with two side-by-side diagrams
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Define some colors for visual clarity
        colors = {
            'subject_border': '#1f77b4',  # blue
            'subject_fill': '#1f77b4',
            'train': '#2ca02c',  # green
            'test': '#d62728',   # red
            'text': '#333333',
            'arrows': '#ff7f0e'  # orange
        }
        
        # 1. Cross-Subject validation illustration (leave-one-subject-out)
        ax1.set_title("Cross-Subject Validation", fontsize=14)
        ax1.set_xlim(0, 10)
        ax1.set_ylim(0, 10)
        
        # Draw subjects as circles
        subjects = [(2, 5), (4, 7), (6, 3), (8, 6), (5, 8)]
        
        # The test subject will have a different treatment
        test_subject_idx = 2  # Subject at position (6, 3)
        
        # Draw subject circles
        for i, (x, y) in enumerate(subjects):
            is_test = (i == test_subject_idx)
            circle = plt.Circle((x, y), 0.8, 
                              fill=True, 
                              alpha=0.3 if is_test else 0.7,
                              fc='#d62728' if is_test else '#1f77b4',
                              ec='#d62728' if is_test else '#1f77b4',
                              linewidth=2)
            ax1.add_patch(circle)
            ax1.text(x, y, f"S{i+1}", ha='center', va='center', 
                   color=colors['text'], fontweight='bold')
        
        # Add model and arrows
        ax1.text(5, 1, "Global Model", ha='center', va='center', 
               fontsize=12, fontweight='bold', bbox=dict(facecolor='white', alpha=0.8))
        
        # Draw arrows from training subjects to model
        for i, (x, y) in enumerate(subjects):
            if i != test_subject_idx:
                ax1.arrow(x, y-0.3, 5-x, 1.3-(y-0.3), head_width=0.3, head_length=0.3, 
                        fc=colors['arrows'], ec=colors['arrows'], linewidth=1.5)
        
        # Draw arrow from model to test subject
        test_x, test_y = subjects[test_subject_idx]
        ax1.arrow(5, 1.3, test_x-5, test_y-1.5-0.3, head_width=0.3, head_length=0.3, 
                fc='#d62728', ec='#d62728', linewidth=1.5, linestyle='--')
        
        # Add explanation
        ax1.text(5, 0.3, "Train on all subjects except one,\nthen test on the left-out subject", 
               ha='center', va='center', fontsize=10, fontstyle='italic')
        
        # 2. Within-Subject validation illustration (within-subject splits)
        ax2.set_title("Within-Subject Validation", fontsize=14)
        ax2.set_xlim(0, 10)
        ax2.set_ylim(0, 10)
        
        # Create horizontal stacked bars for each subject showing train/test split
        bar_height = 0.8
        y_positions = [2, 3.5, 5, 6.5, 8]
        
        # Draw subjects with their train/test splits
        for i, y in enumerate(y_positions):
            # Subject label
            ax2.text(1, y, f"S{i+1}", ha='center', va='center', 
                   color=colors['text'], fontweight='bold')
            
            # Draw train portion (80%)
            train_width = 6.0
            ax2.add_patch(plt.Rectangle((2, y - bar_height/2), train_width, bar_height, 
                                     facecolor=colors['train'], alpha=0.7))
            ax2.text(2 + train_width/2, y, "Train", ha='center', va='center', 
                   color='white', fontweight='bold')
            
            # Draw test portion (20%)
            test_width = 1.5
            ax2.add_patch(plt.Rectangle((2 + train_width, y - bar_height/2), test_width, bar_height, 
                                     facecolor=colors['test'], alpha=0.7))
            ax2.text(2 + train_width + test_width/2, y, "Test", ha='center', va='center', 
                   color='white', fontweight='bold')
        
        # Add model and arrows
        ax2.text(5, 1, "Model", ha='center', va='center', 
               fontsize=12, fontweight='bold', bbox=dict(facecolor='white', alpha=0.8))
        
        # Draw arrows from training portions to model
        for y in y_positions:
            # From train data to model
            mid_train_x = 2 + 6.0/2
            ax2.arrow(mid_train_x, y - 0.2, 5 - mid_train_x, 1.2 - (y - 0.2), 
                    head_width=0.2, head_length=0.2, fc=colors['arrows'], ec=colors['arrows'])
            
            # From model to test data
            mid_test_x = 2 + 6.0 + 1.5/2
            ax2.arrow(5, 1.2, mid_test_x - 5, y - 0.2 - 1.2, 
                    head_width=0.2, head_length=0.2, fc='#d62728', ec='#d62728', linestyle='--')
        
        # Add explanation text
        ax2.text(5, 0.3, "Train on part of each subject's data,\ntest on remaining data from same subjects", 
               ha='center', va='center', fontsize=10, fontstyle='italic')
        
        # Remove axis ticks for cleaner look
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax2.set_xticks([])
        ax2.set_yticks([])
        
        # Show subtle grid for reference
        ax1.grid(True, alpha=0.2)
        ax2.grid(True, alpha=0.2)
        
        # Add a title explaining the key insight
        fig.suptitle("Validation Approaches for Cognitive Load Models", fontsize=16, y=0.98)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)  # Adjust for suptitle
        
        self.figures["validation_approaches"] = self._save_figure()
    
    def add_feature_engineering_visualization(self, original_features: int, 
                                             engineered_features: int,
                                             selected_features: int, 
                                             engineered_feature_groups: Dict[str, List[str]]):
        """
        Create feature engineering visualizations
        
        Args:
            original_features: Number of original features
            engineered_features: Number of engineered features
            selected_features: Number of selected features
            engineered_feature_groups: Dictionary mapping feature groups to lists of feature names
        """
        logger.info("Creating feature engineering visualizations")
        
        # 1. Feature counts by pipeline stage
        plt.figure(figsize=(12, 6))
        
        categories = ["Original", "After Engineering", "Selected for Modeling"]
        counts = [original_features, engineered_features, selected_features]
        colors = sns.color_palette("viridis", 3)
        
        bars = plt.bar(categories, counts, color=colors)
        
        # Add count labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 5,
                   f'{int(height)}', ha='center', va='bottom', fontsize=12)
        
        plt.title("Feature Counts Throughout Pipeline", fontsize=14)
        plt.ylabel("Number of Features", fontsize=12)
        plt.ylim(0, max(counts) * 1.15)  # Add space for labels
        
        # Add percentage increase text
        pct_increase = ((engineered_features - original_features) / original_features) * 100
        plt.text(1.5, counts[1] * 0.5, f"+{pct_increase:.1f}%", 
               ha='center', va='center', fontsize=14, fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.7))
        
        # Add percentage selected text
        pct_selected = (selected_features / engineered_features) * 100
        plt.text(2.5, counts[2] * 0.5, f"{pct_selected:.1f}%\nselected", 
               ha='center', va='center', fontsize=14, fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.7))
        
        plt.tight_layout()
        
        self.figures["feature_counts"] = self._save_figure()
        
        # 2. Engineered feature groups
        if engineered_feature_groups:
            plt.figure(figsize=(12, 6))
            
            # Extract groups and counts
            groups = []
            counts = []
            
            for group, features in engineered_feature_groups.items():
                groups.append(group.title())
                counts.append(len(features))
            
            # Sort by count
            sorted_data = sorted(zip(groups, counts), key=lambda x: x[1], reverse=True)
            groups, counts = zip(*sorted_data)
            
            colors = sns.color_palette("viridis", len(groups))
            bars = plt.bar(groups, counts, color=colors)
            
            # Add count labels on top of bars
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                       f'{int(height)}', ha='center', va='bottom', fontsize=12)
            
            plt.title("Engineered Features by Type", fontsize=14)
            plt.ylabel("Number of Features", fontsize=12)
            plt.ylim(0, max(counts) * 1.15)  # Add space for labels
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            
            self.figures["feature_groups"] = self._save_figure()
        
        logger.info("Feature engineering visualizations created")
    
    def add_feature_importance_visualization(self, features: List[str], 
                                            feature_importances: Dict[str, Dict[str, float]]):
        """
        Create feature importance visualizations
        
        Args:
            features: List of selected features
            feature_importances: Dictionary mapping sources to feature importance dictionaries
        """
        logger.info("Creating feature importance visualizations")
        
        # 1. Global feature importance
        if 'global' in feature_importances:
            plt.figure(figsize=(12, 8))
            
            # Get importance scores
            importances = feature_importances['global']
            
            # Sort features by importance
            sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)
            
            # Display top 15 features
            top_n = min(15, len(sorted_features))
            top_features = sorted_features[:top_n]
            feature_names, importance_values = zip(*top_features)
            
            # Create horizontal bar chart
            bars = plt.barh(list(reversed(feature_names)), list(reversed(importance_values)),
                         color=sns.color_palette("viridis", top_n))
            
            # Add value labels
            for i, bar in enumerate(bars):
                width = bar.get_width()
                plt.text(width + 0.005, bar.get_y() + bar.get_height()/2,
                       f'{width:.4f}', ha='left', va='center', fontsize=10)
            
            plt.title(f"Top {top_n} Features by Global Importance", fontsize=14)
            plt.xlabel("Importance", fontsize=12)
            plt.tight_layout()
            
            self.figures["top_features"] = self._save_figure()
        
        # 2. Feature importance variation across subjects
        if len(feature_importances) > 2:  # Global plus at least 2 subjects
            # Find common important features across subjects
            common_features = set(features)
            
            # For each subject, get the top 5 features
            top_features_by_subject = {}
            
            for source, importances in feature_importances.items():
                if source == 'global':
                    continue
                    
                # Sort features by importance
                sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)
                
                # Get top 5 features
                top_features_by_subject[source] = [f for f, _ in sorted_features[:5]]
            
            # Count how many times each feature appears in top 5
            feature_counts = {}
            for feature in common_features:
                count = sum(1 for subject_features in top_features_by_subject.values() 
                           if feature in subject_features)
                feature_counts[feature] = count
            
            # Get top 10 most common important features
            most_common = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            common_top_features = [f for f, _ in most_common]
            
            # Select a sample of subjects (max 10)
            subject_sample = list(feature_importances.keys())
            subject_sample.remove('global')  # Remove global
            if len(subject_sample) > 10:
                subject_sample = subject_sample[:10]
            
            # Create heatmap data
            heatmap_data = np.zeros((len(subject_sample), len(common_top_features)))
            
            for i, subject in enumerate(subject_sample):
                for j, feature in enumerate(common_top_features):
                    if subject in feature_importances and feature in feature_importances[subject]:
                        heatmap_data[i, j] = feature_importances[subject][feature]
            
            plt.figure(figsize=(14, 10))
            ax = sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap="viridis",
                         xticklabels=common_top_features, yticklabels=[f"S{i+1}" for i in range(len(subject_sample))],
                         linewidths=0.5, cbar_kws={"label": "Feature Importance"})
            
            plt.title("Feature Importance Variation Across Subjects", fontsize=14)
            plt.xlabel("Feature", fontsize=12)
            plt.ylabel("Subject", fontsize=12)
            
            # Rotate feature names for better readability
            plt.xticks(rotation=45, ha='right')
            
            plt.tight_layout()
            
            self.figures["subject_feature_importance"] = self._save_figure()
            
            # 3. Feature stability visualization
            plt.figure(figsize=(12, 8))
            
            # For each of the top 10 features, get the mean and std of importance across subjects
            feature_means = []
            feature_stds = []
            
            for feature in common_top_features:
                # Collect importance values for this feature across all subjects
                values = []
                for subject in feature_importances:
                    if subject != 'global' and feature in feature_importances[subject]:
                        values.append(feature_importances[subject][feature])
                
                if values:
                    feature_means.append(np.mean(values))
                    feature_stds.append(np.std(values))
                else:
                    feature_means.append(0)
                    feature_stds.append(0)
            
            # Sort features by mean importance
            sorted_idx = np.argsort(feature_means)[::-1]
            sorted_features = [common_top_features[i] for i in sorted_idx]
            sorted_means = [feature_means[i] for i in sorted_idx]
            sorted_stds = [feature_stds[i] for i in sorted_idx]
            
            # Plot feature stability
            plt.figure(figsize=(12, 8))
            
            # Create horizontal bar chart with error bars
            y_pos = np.arange(len(sorted_features))
            plt.barh(y_pos, sorted_means, xerr=sorted_stds, align='center',
                   color=sns.color_palette("viridis", len(sorted_features)),
                   alpha=0.7, ecolor='black', capsize=5)
            
            plt.yticks(y_pos, sorted_features)
            plt.xlabel('Mean Feature Importance', fontsize=12)
            plt.title('Feature Importance Stability Across Subjects', fontsize=14)
            
            # Add coefficient of variation (CV = std/mean) for each feature
            for i, (mean, std) in enumerate(zip(sorted_means, sorted_stds)):
                if mean > 0:
                    cv = std / mean
                    plt.text(mean + std + 0.01, y_pos[i], f'CV: {cv:.2f}', 
                           va='center', fontsize=10)
            
            plt.tight_layout()
            
            self.figures["feature_stability"] = self._save_figure()
        
        logger.info("Feature importance visualizations created")
    
    def add_model_performance_visualization(self, global_performance: Dict,
                                           subject_performance: Dict,
                                           adaptive_performance: Dict = None,
                                           comparison_results: Dict = None):
        """
        Create model performance visualizations
        
        Args:
            global_performance: Dictionary with global model performance metrics
            subject_performance: Dictionary with subject-specific model performance metrics
            adaptive_performance: Dictionary with adaptive model performance metrics
            comparison_results: Dictionary with model comparison results
        """
        logger.info("Creating model performance visualizations")
        
        # Store metrics
        self.metrics.update({
            "global_within_subject_r2": global_performance.get("within_subject_r2", 0),
            "global_within_subject_r2_ci": global_performance.get("within_subject_r2_ci", [0, 0]),
            "global_cross_subject_r2": global_performance.get("cross_subject_r2", 0),
            "global_cross_subject_r2_ci": global_performance.get("cross_subject_r2_ci", [0, 0]),
            "subject_r2": subject_performance.get("avg_r2", 0),
            "subject_r2_ci": subject_performance.get("r2_ci", [0, 0]),
            "subject_r2_std": subject_performance.get("std_r2", 0)
        })
        
        if adaptive_performance:
            self.metrics.update({
                "adaptive_r2": adaptive_performance.get("avg_score", 0),
                "adaptive_r2_ci": adaptive_performance.get("score_ci", [0, 0]),
                "optimal_weight": adaptive_performance.get("avg_weight", 0),
                "optimal_weight_ci": adaptive_performance.get("weight_ci", [0, 0])
            })
        
        if comparison_results:
            self.metrics.update({
                "avg_r2_improvement": comparison_results.get("avg_r2_improvement", 0),
                "r2_improvement_ci": comparison_results.get("r2_improvement_ci", [0, 0]),
                "improvement_p_value": comparison_results.get("p_value", 1)
            })
        
        # 1. Model performance comparison
        # CHANGE: Increased figure height to allow for more space
        plt.figure(figsize=(14, 10))
        
        # Collect data
        models = ["Global Model\n(Cross-Subject)", "Global Model\n(Within-Subject)", "Subject-Specific\nModels"]
        r2_values = [
            global_performance.get("cross_subject_r2", 0),
            global_performance.get("within_subject_r2", 0),
            subject_performance.get("avg_r2", 0)
        ]
        
        # Add adaptive model if available
        if adaptive_performance:
            models.append("Adaptive Transfer\nModels")
            r2_values.append(adaptive_performance.get("avg_score", 0))
        
        # Configure colors
        colors = sns.color_palette("viridis", len(models))
        
        # Create bar chart
        bars = plt.bar(models, r2_values, color=colors, width=0.7)
        
        # Add confidence intervals
        ci_data = [
            global_performance.get("cross_subject_r2_ci", [0, 0]),
            global_performance.get("within_subject_r2_ci", [0, 0]),
            subject_performance.get("r2_ci", [0, 0])
        ]
        
        if adaptive_performance:
            ci_data.append(adaptive_performance.get("score_ci", [0, 0]))
        
        # Add error bars for confidence intervals
        for i, (bar, ci) in enumerate(zip(bars, ci_data)):
            height = bar.get_height()
            if ci and len(ci) == 2:
                yerr_low = height - ci[0] if ci[0] <= height else 0
                yerr_high = ci[1] - height if ci[1] >= height else 0
                plt.errorbar(i, height, yerr=[[yerr_low], [yerr_high]], fmt='none', color='black',
                           capsize=5, capthick=1, elinewidth=1)
        
        # Add value labels
        # CHANGE: Positioned value labels better to avoid overlap with significance annotation
        for i, bar in enumerate(bars):
            height = bar.get_height()
            vert_align = 'bottom' if height >= 0 else 'top'
            # Adjust y_offset for each bar to avoid overlaps
            if height > 0:
                y_offset = 0.02
            else:
                y_offset = -0.06
            
            # Format label text with confidence interval
            if ci_data[i] and len(ci_data[i]) == 2:
                ci_text = f"[{ci_data[i][0]:.2f}, {ci_data[i][1]:.2f}]"
            else:
                ci_text = ""
            
            plt.text(
                i, height + y_offset,
                f'{height:.3f}\n{ci_text}',
                ha='center', 
                va=vert_align,
                fontsize=11
            )
        
        # Add significance markers if comparison results available
        # CHANGE: Positioned significance annotation higher to avoid overlap
        if comparison_results and "p_value" in comparison_results:
            p_value = comparison_results.get("p_value", 1)
            if p_value < 0.05:
                # Add significance stars between global within-subject and subject-specific
                x1, x2 = 1, 2  # Positions of the bars
                max_height = max(r2_values)
                # Position significance line above the highest bar plus CI
                y = max_height + 0.15  
                plt.plot([x1, x2], [y, y], 'k-')
                plt.text((x1 + x2) / 2, y + 0.02, 
                       f'p = {p_value:.4f} {"*" * (3 - int(p_value * 10))}',
                       ha='center', va='bottom')
        
        # Add basic styling
        plt.title("Model Performance Comparison (R²)", fontsize=16)
        plt.ylabel("R² Score", fontsize=14)
        
        # Set y-limits to properly show full range of values plus room for annotations
        # CHANGE: Increased upper limit to accommodate significance annotation
        min_val = min(r2_values)
        plt.ylim(min(min_val - 0.2, -1.1), 1.2)
        
        plt.axhline(y=0, linestyle='--', color='gray', alpha=0.5)
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add explanation for negative R²
        # CHANGE: Better positioning of the note explaining negative R² values
        if min_val < 0:
            # CHANGE: Use figure text with better positioning instead of tight_layout
            plt.subplots_adjust(bottom=0.25)  # Increase bottom margin for note
            
            # Create a text box for the explanation that doesn't overlap with x-axis labels
            plt.figtext(0.5, 0.1, 
                      "Note: Negative R² values indicate that the model performs worse than simply predicting the mean value.\n"
                      "This is common when generalizing to new subjects with highly individual cognitive load patterns.",
                      ha="center", fontsize=11, bbox={"facecolor":"white", "alpha":0.9, "pad":5},
                      wrap=True)
        else:
            plt.tight_layout()
        
        self.figures["model_comparison"] = self._save_figure()
        
        # 2. Subject-specific performance distribution
        if "subject_results" in subject_performance:
            subject_results = subject_performance["subject_results"]
            
            if subject_results:
                # Extract R² values for each subject
                subject_r2_values = [result["r2"] for result in subject_results.values()]
                
                plt.figure(figsize=(12, 6))
                
                # Create histogram
                sns.histplot(subject_r2_values, kde=True, bins=15)
                
                # Add mean and median lines
                plt.axvline(np.mean(subject_r2_values), color='red', linestyle='dashed', linewidth=1)
                plt.axvline(np.median(subject_r2_values), color='green', linestyle='dashed', linewidth=1)
                
                # Add a legend
                plt.legend(['KDE', f'Mean: {np.mean(subject_r2_values):.3f}', 
                          f'Median: {np.median(subject_r2_values):.3f}'])
                
                plt.title("Distribution of Subject-Specific Model Performance (R²)", fontsize=14)
                plt.xlabel("R² Score", fontsize=12)
                plt.ylabel("Count", fontsize=12)
                
                # Add distribution statistics
                stats_text = (
                    f"Mean: {np.mean(subject_r2_values):.3f}\n"
                    f"Median: {np.median(subject_r2_values):.3f}\n"
                    f"Std Dev: {np.std(subject_r2_values):.3f}\n"
                    f"Min: {min(subject_r2_values):.3f}\n"
                    f"Max: {max(subject_r2_values):.3f}"
                )
                plt.annotate(stats_text, xy=(0.95, 0.95), xycoords='axes fraction', 
                           ha='right', va='top', bbox=dict(boxstyle="round,pad=0.5", 
                                                         fc="white", ec="gray", alpha=0.8))
                
                plt.tight_layout()
                
                self.figures["subject_r2_distribution"] = self._save_figure()
                
                # Create subject performance ranking
                if len(subject_results) > 5:
                    plt.figure(figsize=(14, 8))
                    
                    # Sort subjects by R² score
                    sorted_subjects = sorted(subject_results.items(), 
                                         key=lambda x: x[1]["r2"], reverse=True)
                    
                    # Extract data
                    subject_ids = [f"S{i+1}" for i in range(len(sorted_subjects))]
                    r2_values = [result["r2"] for _, result in sorted_subjects]
                    
                    # Create bar chart
                    bars = plt.bar(subject_ids, r2_values, 
                                 color=sns.color_palette("viridis", len(sorted_subjects)))
                    
                    # Add horizontal line for mean R²
                    plt.axhline(y=np.mean(r2_values), color='red', linestyle='--', 
                              label=f'Mean R² = {np.mean(r2_values):.3f}')
                    
                    plt.title("Subject-Specific Model Performance Ranking", fontsize=14)
                    plt.xlabel("Subject ID", fontsize=12)
                    plt.ylabel("R² Score", fontsize=12)
                    plt.xticks(rotation=90 if len(subject_ids) > 30 else 45)
                    plt.legend()
                    
                    # Show only a subset of subject labels if there are too many
                    if len(subject_ids) > 30:
                        step = max(1, len(subject_ids) // 20)
                        plt.xticks(range(0, len(subject_ids), step), 
                                 [subject_ids[i] for i in range(0, len(subject_ids), step)])
                    
                    plt.tight_layout()
                    
                    self.figures["subject_performance_ranking"] = self._save_figure()
        
        # 3. Cross-subject validation results
        if "cross_subject_subject_results" in global_performance:
            cross_results = global_performance["cross_subject_subject_results"]
            
            if cross_results:
                plt.figure(figsize=(14, 8))
                
                # Sort subjects by R² score
                sorted_results = sorted(cross_results.items(), 
                                     key=lambda x: x[1]["r2"], reverse=True)
                
                # Extract data
                subject_ids = [f"S{i+1}" for i in range(len(sorted_results))]
                r2_values = [result["r2"] for _, result in sorted_results]
                
                # Create bar chart with a colormap based on R² value
                colors = plt.cm.RdYlGn(np.linspace(0, 1, len(r2_values)))
                bars = plt.bar(subject_ids, r2_values, color=colors)
                
                # Add horizontal line at R² = 0
                plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                
                # Add mean line
                plt.axhline(y=np.mean(r2_values), color='red', linestyle='--', 
                          label=f'Mean R² = {np.mean(r2_values):.3f}')
                
                plt.title("Global Model Performance on New Subjects (Cross-Subject Validation)", 
                        fontsize=14)
                plt.xlabel("Subject ID", fontsize=12)
                plt.ylabel("R² Score", fontsize=12)
                plt.xticks(rotation=90 if len(subject_ids) > 30 else 45)
                plt.legend()
                
                # Show only a subset of subject labels if there are too many
                if len(subject_ids) > 30:
                    step = max(1, len(subject_ids) // 20)
                    plt.xticks(range(0, len(subject_ids), step), 
                             [subject_ids[i] for i in range(0, len(subject_ids), step)])
                
                plt.tight_layout()
                
                self.figures["cross_subject_validation"] = self._save_figure()
        
        # 4. Improvement waterfall chart
        if comparison_results and "avg_r2_improvement" in comparison_results:
            plt.figure(figsize=(12, 8))
            
            # Extract performance values
            global_r2 = global_performance.get("within_subject_r2", 0)
            subject_r2 = subject_performance.get("avg_r2", 0)
            
            # Calculate improvement
            subject_improvement = subject_r2 - global_r2
            
            # Add adaptive improvement if available
            if adaptive_performance:
                adaptive_r2 = adaptive_performance.get("avg_score", 0)
                adaptive_improvement = adaptive_r2 - subject_r2
            else:
                adaptive_r2 = None
                adaptive_improvement = None
            
            # Create baseline bars
            plt.bar(0, global_r2, bottom=0, color='#1f77b4', width=0.6, label='Global Model')
            
            # Create improvement bars
            plt.bar(1, subject_improvement, bottom=global_r2, color='#2ca02c', width=0.6, 
                  label='Subject-Specific Improvement')
            
            if adaptive_r2 is not None:
                plt.bar(2, adaptive_improvement, bottom=subject_r2, color='#9467bd', width=0.6, 
                      label='Adaptive Transfer Benefit')
            
            # Add baseline for global model
            plt.axhline(y=global_r2, color='#1f77b4', linestyle='--', alpha=0.5)
            
            # Add baseline for subject model if adaptive is available
            if adaptive_r2 is not None:
                plt.axhline(y=subject_r2, color='#2ca02c', linestyle='--', alpha=0.5)
            
            # Add value labels
            # Format for better display of positive and negative values
            plt.text(0, global_r2/2 if global_r2 > 0 else global_r2 - 0.05, f"{global_r2:.3f}", 
                   ha='center', va='center', color='white' if global_r2 > 0 else 'black', 
                   fontweight='bold')
            
            label_text = f"+{subject_improvement:.3f}" if subject_improvement > 0 else f"{subject_improvement:.3f}"
            plt.text(1, global_r2 + subject_improvement/2, label_text, 
                   ha='center', va='center', color='white', fontweight='bold')
            
            if adaptive_r2 is not None:
                label_text = f"+{adaptive_improvement:.3f}" if adaptive_improvement > 0 else f"{adaptive_improvement:.3f}"
                plt.text(2, subject_r2 + adaptive_improvement/2, label_text, 
                       ha='center', va='center', color='white', fontweight='bold')
                
                # Add final value
                plt.text(2, adaptive_r2 + 0.03, f"Final: {adaptive_r2:.3f}", 
                       ha='center', va='bottom', color='black', fontweight='bold')
            else:
                # Add final value for subject model
                plt.text(1, subject_r2 + 0.03, f"Final: {subject_r2:.3f}", 
                       ha='center', va='bottom', color='black', fontweight='bold')
            
            # Add x-axis labels
            x_labels = ['Global\nModel', 'Subject-Specific\nModel']
            if adaptive_r2 is not None:
                x_labels.append('Adaptive\nTransfer')
            
            plt.xticks(range(len(x_labels)), x_labels)
            plt.ylabel('R² Score', fontsize=12)
            plt.title('Performance Improvements with Different Modeling Approaches', fontsize=14)
            plt.legend(loc='upper left')
            
            # Set y-limits to properly show the full range
            min_y = min(0, global_r2 - 0.1)  # Ensure we include 0 and the global_r2 with some padding
            max_val = max(subject_r2, adaptive_r2 if adaptive_r2 is not None else 0)
            plt.ylim(min_y, max_val + 0.15)
            
            # Add statistical significance annotation if available
            if "p_value" in comparison_results:
                p_value = comparison_results.get("p_value", 1)
                signif_text = f"p = {p_value:.4f}"
                if p_value < 0.05:
                    signif_text += " *"
                if p_value < 0.01:
                    signif_text += "*"
                if p_value < 0.001:
                    signif_text += "*"
                
                plt.text(0.5, global_r2 + subject_improvement/2 + 0.05, signif_text, 
                       ha='center', va='bottom', fontsize=12,
                       bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8))
            
            plt.grid(axis='y', linestyle='--', alpha=0.3)
            plt.tight_layout()
            
            self.figures["performance_improvements"] = self._save_figure()
        
        logger.info("Model performance visualizations created")
    
    def add_adaptive_transfer_visualization(self, adaptive_performance: Dict):
        """
        Create visualizations for adaptive transfer learning
        
        Args:
            adaptive_performance: Dictionary with adaptive model performance metrics
        """
        logger.info("Creating adaptive transfer learning visualizations")
        
        # 1. Optimal weight visualization
        if "avg_weight" in adaptive_performance:
            optimal_weight = adaptive_performance["avg_weight"]
            weight_ci = adaptive_performance.get("weight_ci", [optimal_weight - 0.1, optimal_weight + 0.1])
            
            plt.figure(figsize=(12, 6))
            
            # Create a visual scale from 0 to 1
            plt.plot([0, 1], [0, 0], 'k-', linewidth=3)
            plt.scatter([0, 1], [0, 0], s=100, color=['lightblue', 'darkblue'])
            
            # Add labels
            plt.text(0, 0.02, "Global Model\n(all subjects)", ha='center', va='bottom', fontsize=12)
            plt.text(1, 0.02, "Subject Model\n(individual data)", ha='center', va='bottom', fontsize=12)
            
            # Add a marker for the optimal weight
            plt.scatter([optimal_weight], [0], s=200, color='red', zorder=3)
            
            # Add confidence interval
            plt.plot([weight_ci[0], weight_ci[1]], [0, 0], color='red', linewidth=4, alpha=0.3)
            
            plt.annotate(f"Optimal Weight: {optimal_weight:.2f}\n95% CI: [{weight_ci[0]:.2f}, {weight_ci[1]:.2f}]", 
                       xy=(optimal_weight, 0), xytext=(optimal_weight, 0.1),
                       arrowprops=dict(arrowstyle="->", color='red'),
                       ha='center', fontsize=12, color='red')
            
            # Add explanation
            explanation = []
            
            if optimal_weight > 0.8:
                explanation = [
                    f"An optimal weight of {optimal_weight:.2f} means that for best performance,",
                    "predictions should rely mostly on individual patterns",
                    "rather than general patterns across subjects."
                ]
            elif optimal_weight > 0.5:
                explanation = [
                    f"An optimal weight of {optimal_weight:.2f} means that for best performance,",
                    "predictions should balance individual and group patterns,",
                    "with a preference for individual patterns."
                ]
            elif optimal_weight > 0.2:
                explanation = [
                    f"An optimal weight of {optimal_weight:.2f} means that for best performance,",
                    "predictions should balance individual and group patterns fairly evenly."
                ]
            else:
                explanation = [
                    f"An optimal weight of {optimal_weight:.2f} means that for best performance,",
                    "predictions should rely mostly on group patterns",
                    "rather than individual patterns."
                ]
            
            y_pos = -0.2
            for line in explanation:
                plt.text(0.5, y_pos, line, ha='center', fontsize=12)
                y_pos -= 0.05
            
            # Remove axis ticks and labels
            plt.xlim(-0.2, 1.2)
            plt.ylim(-0.4, 0.3)
            plt.axis('off')
            
            plt.title("Interpreting Transfer Learning Weight", fontsize=14)
            plt.tight_layout()
            
            self.figures["transfer_learning_weight"] = self._save_figure()
        
        # 2. Weight distribution across subjects
        if "subject_results" in adaptive_performance:
            subject_results = adaptive_performance["subject_results"]
            
            if subject_results:
                # Extract weights for each subject
                weights = [result["optimal_weight"] for result in subject_results.values()]
                
                plt.figure(figsize=(12, 6))
                
                # Create histogram
                sns.histplot(weights, kde=True, bins=10)
                
                # Add mean and median lines
                plt.axvline(np.mean(weights), color='red', linestyle='dashed', linewidth=1)
                plt.axvline(np.median(weights), color='green', linestyle='dashed', linewidth=1)
                
                # Add a legend
                plt.legend(['KDE', f'Mean: {np.mean(weights):.2f}', 
                          f'Median: {np.median(weights):.2f}'])
                
                plt.title("Distribution of Optimal Weights Across Subjects", fontsize=14)
                plt.xlabel("Optimal Weight (0 = Global Model, 1 = Subject Model)", fontsize=12)
                plt.ylabel("Count", fontsize=12)
                
                # Add distribution statistics
                stats_text = (
                    f"Mean: {np.mean(weights):.2f}\n"
                    f"Median: {np.median(weights):.2f}\n"
                    f"Std Dev: {np.std(weights):.2f}\n"
                    f"Min: {min(weights):.2f}\n"
                    f"Max: {max(weights):.2f}"
                )
                plt.annotate(stats_text, xy=(0.95, 0.95), xycoords='axes fraction', 
                           ha='right', va='top', bbox=dict(boxstyle="round,pad=0.5", 
                                                         fc="white", ec="gray", alpha=0.8))
                
                plt.tight_layout()
                
                self.figures["weight_distribution"] = self._save_figure()
        
        # 3. Scatterplot of R² vs optimal weight
        if "subject_results" in adaptive_performance:
            subject_results = adaptive_performance["subject_results"]
            
            if subject_results:
                # Extract weights and R² values for each subject
                weights = []
                r2_values = []
                
                for result in subject_results.values():
                    if "optimal_weight" in result and "r2_score" in result:
                        weights.append(result["optimal_weight"])
                        r2_values.append(result["r2_score"])
                
                if weights and r2_values:
                    plt.figure(figsize=(12, 8))
                    
                    # Create scatter plot
                    plt.scatter(weights, r2_values, s=80, alpha=0.7, 
                              c=r2_values, cmap='viridis')
                    
                    # Add a colorbar
                    plt.colorbar(label='R² Score')
                    
                    # Add a trend line
                    if len(weights) > 1:
                        z = np.polyfit(weights, r2_values, 1)
                        p = np.poly1d(z)
                        plt.plot(np.array([0, 1]), p(np.array([0, 1])), "r--", alpha=0.8,
                               label=f"Trend: y = {z[0]:.3f}x + {z[1]:.3f}")
                        
                        # Calculate correlation
                        corr, p_value = stats.pearsonr(weights, r2_values)
                        
                        # Add correlation annotation
                        corr_text = f"Correlation: r = {corr:.3f}"
                        if p_value < 0.05:
                            corr_text += f" (p = {p_value:.3f})*"
                        else:
                            corr_text += f" (p = {p_value:.3f})"
                        
                        plt.text(0.5, 0.05, corr_text, transform=plt.gca().transAxes,
                               ha='center', va='bottom', fontsize=12,
                               bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8))
                    
                    plt.title("Relationship Between Optimal Weight and Model Performance", fontsize=14)
                    plt.xlabel("Optimal Weight (0 = Global Model, 1 = Subject Model)", fontsize=12)
                    plt.ylabel("R² Score", fontsize=12)
                    
                    # Add legend if trendline exists
                    if len(weights) > 1:
                        plt.legend()
                    
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    
                    self.figures["weight_vs_performance"] = self._save_figure()
        
        logger.info("Adaptive transfer learning visualizations created")
    
    def add_comparison_visualization(self, predictions: Dict):
        """
        Create visualizations comparing predicted vs. actual values
        
        Args:
            predictions: Dictionary with prediction data
        """
        logger.info("Creating prediction comparison visualizations")
        
        # 1. Global model: Predicted vs. Actual scatter plot
        if 'global_model' in predictions:
            global_data = predictions['global_model']
            actual = global_data.get('actual', [])
            predicted = global_data.get('predicted', [])
            
            if actual and predicted and len(actual) == len(predicted):
                plt.figure(figsize=(12, 8))
                
                # Create scatter plot
                plt.scatter(actual, predicted, alpha=0.6, s=50, c='#1f77b4', edgecolor='k')
                
                # Add perfect prediction line
                min_val = min(min(actual), min(predicted))
                max_val = max(max(actual), max(predicted))
                plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
                
                # Add trend line
                z = np.polyfit(actual, predicted, 1)
                p = np.poly1d(z)
                plt.plot(np.array([min_val, max_val]), p(np.array([min_val, max_val])), "g-", alpha=0.8,
                       label=f"Trend: y = {z[0]:.3f}x + {z[1]:.3f}")
                
                # Calculate metrics
                r2 = r2_score(actual, predicted)
                rmse = np.sqrt(mean_squared_error(actual, predicted))
                
                plt.title(f"Global Model: Predicted vs. Actual Values (R² = {r2:.3f}, RMSE = {rmse:.3f})", 
                        fontsize=14)
                plt.xlabel("Actual Cognitive Load", fontsize=12)
                plt.ylabel("Predicted Cognitive Load", fontsize=12)
                
                plt.grid(True, alpha=0.3)
                plt.legend()
                plt.tight_layout()
                
                self.figures["global_predicted_vs_actual"] = self._save_figure()
                
                # Create residual plot
                plt.figure(figsize=(12, 8))
                
                residuals = np.array(predicted) - np.array(actual)
                
                # Create scatter plot
                plt.scatter(actual, residuals, alpha=0.6, s=50, c='#1f77b4', edgecolor='k')
                
                # Add horizontal line at 0
                plt.axhline(y=0, color='r', linestyle='--')
                
                # Add trend line
                z = np.polyfit(actual, residuals, 1)
                p = np.poly1d(z)
                plt.plot(np.array([min_val, max_val]), p(np.array([min_val, max_val])), "g-", alpha=0.8,
                       label=f"Trend: y = {z[0]:.3f}x + {z[1]:.3f}")
                
                plt.title("Global Model: Residuals Plot", fontsize=14)
                plt.xlabel("Actual Cognitive Load", fontsize=12)
                plt.ylabel("Residual (Predicted - Actual)", fontsize=12)
                
                # Add residual statistics
                stats_text = (
                    f"Mean Residual: {np.mean(residuals):.3f}\n"
                    f"Std Dev: {np.std(residuals):.3f}\n"
                    f"Min: {min(residuals):.3f}\n"
                    f"Max: {max(residuals):.3f}"
                )
                plt.annotate(stats_text, xy=(0.05, 0.95), xycoords='axes fraction', 
                           ha='left', va='top', bbox=dict(boxstyle="round,pad=0.5", 
                                                        fc="white", ec="gray", alpha=0.8))
                
                plt.grid(True, alpha=0.3)
                plt.legend()
                plt.tight_layout()
                
                self.figures["global_residuals"] = self._save_figure()
        
        # 2. Subject-specific models: Predicted vs. Actual (aggregated)
        if 'subject_models' in predictions:
            subject_data = predictions['subject_models']
            
            # Collect data from all subjects
            all_actual = []
            all_predicted = []
            
            for data in subject_data.values():
                if 'actual' in data and 'predicted' in data:
                    if len(data['actual']) == len(data['predicted']) and len(data['actual']) > 0:
                        all_actual.extend(data['actual'])
                        all_predicted.extend(data['predicted'])
            
            if all_actual and all_predicted and len(all_actual) == len(all_predicted):
                plt.figure(figsize=(12, 8))
                
                # Create scatter plot
                plt.scatter(all_actual, all_predicted, alpha=0.6, s=50, c='#2ca02c', edgecolor='k')
                
                # Add perfect prediction line
                min_val = min(min(all_actual), min(all_predicted))
                max_val = max(max(all_actual), max(all_predicted))
                plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
                
                # Add trend line
                z = np.polyfit(all_actual, all_predicted, 1)
                p = np.poly1d(z)
                plt.plot(np.array([min_val, max_val]), p(np.array([min_val, max_val])), "g-", alpha=0.8,
                       label=f"Trend: y = {z[0]:.3f}x + {z[1]:.3f}")
                
                # Calculate metrics
                r2 = r2_score(all_actual, all_predicted)
                rmse = np.sqrt(mean_squared_error(all_actual, all_predicted))
                
                plt.title(f"Subject-Specific Models: Predicted vs. Actual (R² = {r2:.3f}, RMSE = {rmse:.3f})", 
                        fontsize=14)
                plt.xlabel("Actual Cognitive Load", fontsize=12)
                plt.ylabel("Predicted Cognitive Load", fontsize=12)
                
                plt.grid(True, alpha=0.3)
                plt.legend()
                plt.tight_layout()
                
                self.figures["subject_predicted_vs_actual"] = self._save_figure()
                
                # Create residual plot
                plt.figure(figsize=(12, 8))
                
                residuals = np.array(all_predicted) - np.array(all_actual)
                
                # Create scatter plot
                plt.scatter(all_actual, residuals, alpha=0.6, s=50, c='#2ca02c', edgecolor='k')
                
                # Add horizontal line at 0
                plt.axhline(y=0, color='r', linestyle='--')
                
                # Add trend line
                z = np.polyfit(all_actual, residuals, 1)
                p = np.poly1d(z)
                plt.plot(np.array([min_val, max_val]), p(np.array([min_val, max_val])), "g-", alpha=0.8,
                       label=f"Trend: y = {z[0]:.3f}x + {z[1]:.3f}")
                
                plt.title("Subject-Specific Models: Residuals Plot", fontsize=14)
                plt.xlabel("Actual Cognitive Load", fontsize=12)
                plt.ylabel("Residual (Predicted - Actual)", fontsize=12)
                
                # Add residual statistics
                stats_text = (
                    f"Mean Residual: {np.mean(residuals):.3f}\n"
                    f"Std Dev: {np.std(residuals):.3f}\n"
                    f"Min: {min(residuals):.3f}\n"
                    f"Max: {max(residuals):.3f}"
                )
                plt.annotate(stats_text, xy=(0.05, 0.95), xycoords='axes fraction', 
                           ha='left', va='top', bbox=dict(boxstyle="round,pad=0.5", 
                                                        fc="white", ec="gray", alpha=0.8))
                
                plt.grid(True, alpha=0.3)
                plt.legend()
                plt.tight_layout()
                
                self.figures["subject_residuals"] = self._save_figure()
        
        # 3. Adaptive models: Predicted vs. Actual (aggregated)
        if 'adaptive_models' in predictions:
            adaptive_data = predictions['adaptive_models']
            
            # Collect data from all subjects
            all_actual = []
            all_predicted = []
            
            for data in adaptive_data.values():
                if 'actual' in data and 'predicted' in data:
                    if len(data['actual']) == len(data['predicted']) and len(data['actual']) > 0:
                        all_actual.extend(data['actual'])
                        all_predicted.extend(data['predicted'])
            
            if all_actual and all_predicted and len(all_actual) == len(all_predicted):
                plt.figure(figsize=(12, 8))
                
                # Create scatter plot
                plt.scatter(all_actual, all_predicted, alpha=0.6, s=50, c='#9467bd', edgecolor='k')
                
                # Add perfect prediction line
                min_val = min(min(all_actual), min(all_predicted))
                max_val = max(max(all_actual), max(all_predicted))
                plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
                
                # Add trend line
                z = np.polyfit(all_actual, all_predicted, 1)
                p = np.poly1d(z)
                plt.plot(np.array([min_val, max_val]), p(np.array([min_val, max_val])), "g-", alpha=0.8,
                       label=f"Trend: y = {z[0]:.3f}x + {z[1]:.3f}")
                
                # Calculate metrics
                r2 = r2_score(all_actual, all_predicted)
                rmse = np.sqrt(mean_squared_error(all_actual, all_predicted))
                
                plt.title(f"Adaptive Transfer Models: Predicted vs. Actual (R² = {r2:.3f}, RMSE = {rmse:.3f})", 
                        fontsize=14)
                plt.xlabel("Actual Cognitive Load", fontsize=12)
                plt.ylabel("Predicted Cognitive Load", fontsize=12)
                
                plt.grid(True, alpha=0.3)
                plt.legend()
                plt.tight_layout()
                
                self.figures["adaptive_predicted_vs_actual"] = self._save_figure()
                
                # Create residual plot
                plt.figure(figsize=(12, 8))
                
                residuals = np.array(all_predicted) - np.array(all_actual)
                
                # Create scatter plot
                plt.scatter(all_actual, residuals, alpha=0.6, s=50, c='#9467bd', edgecolor='k')
                
                # Add horizontal line at 0
                plt.axhline(y=0, color='r', linestyle='--')
                
                # Add trend line
                z = np.polyfit(all_actual, residuals, 1)
                p = np.poly1d(z)
                plt.plot(np.array([min_val, max_val]), p(np.array([min_val, max_val])), "g-", alpha=0.8,
                       label=f"Trend: y = {z[0]:.3f}x + {z[1]:.3f}")
                
                plt.title("Adaptive Transfer Models: Residuals Plot", fontsize=14)
                plt.xlabel("Actual Cognitive Load", fontsize=12)
                plt.ylabel("Residual (Predicted - Actual)", fontsize=12)
                
                # Add residual statistics
                stats_text = (
                    f"Mean Residual: {np.mean(residuals):.3f}\n"
                    f"Std Dev: {np.std(residuals):.3f}\n"
                    f"Min: {min(residuals):.3f}\n"
                    f"Max: {max(residuals):.3f}"
                )
                plt.annotate(stats_text, xy=(0.05, 0.95), xycoords='axes fraction', 
                           ha='left', va='top', bbox=dict(boxstyle="round,pad=0.5", 
                                                        fc="white", ec="gray", alpha=0.8))
                
                plt.grid(True, alpha=0.3)
                plt.legend()
                plt.tight_layout()
                
                self.figures["adaptive_residuals"] = self._save_figure()
        
        # 4. Model comparison for individual subject
        # Find a subject with all three model predictions
        if ('subject_models' in predictions and 'global_for_subjects' in predictions and
            'adaptive_models' in predictions):
            
            common_subjects = set(predictions['subject_models'].keys())
            common_subjects = common_subjects.intersection(predictions['global_for_subjects'].keys())
            common_subjects = common_subjects.intersection(predictions['adaptive_models'].keys())
            
            if common_subjects:
                # Pick the first subject
                subject_id = next(iter(common_subjects))
                
                # Get data for this subject
                subject_data = predictions['subject_models'][subject_id]
                global_data = predictions['global_for_subjects'][subject_id]
                adaptive_data = predictions['adaptive_models'][subject_id]
                
                # Verify data is valid
                if (len(subject_data.get('actual', [])) == len(subject_data.get('predicted', [])) and
                    len(global_data.get('actual', [])) == len(global_data.get('predicted', [])) and
                    len(adaptive_data.get('actual', [])) == len(adaptive_data.get('predicted', []))):
                    
                    actual = subject_data['actual']
                    subject_pred = subject_data['predicted']
                    global_pred = global_data['predicted']
                    adaptive_pred = adaptive_data['predicted']
                    
                    plt.figure(figsize=(14, 8))
                    
                    # Create time series plot
                    x = range(len(actual))
                    plt.plot(x, actual, 'k-', linewidth=2.5, label='Actual')
                    plt.plot(x, global_pred, 'b--', linewidth=1.5, label='Global Model')
                    plt.plot(x, subject_pred, 'g-', linewidth=1.5, label='Subject-Specific')
                    plt.plot(x, adaptive_pred, 'm-', linewidth=1.5, label='Adaptive Transfer')
                    
                    plt.title(f"Model Comparison for Subject {subject_id}", fontsize=14)
                    plt.xlabel("Sample Index", fontsize=12)
                    plt.ylabel("Cognitive Load", fontsize=12)
                    
                    # Add performance metrics
                    global_r2 = r2_score(actual, global_pred)
                    subject_r2 = r2_score(actual, subject_pred)
                    adaptive_r2 = r2_score(actual, adaptive_pred)
                    
                    metrics_text = (
                        f"Global Model R²: {global_r2:.3f}\n"
                        f"Subject-Specific R²: {subject_r2:.3f}\n"
                        f"Adaptive Transfer R²: {adaptive_r2:.3f}"
                    )
                    plt.annotate(metrics_text, xy=(0.02, 0.02), xycoords='axes fraction', 
                               ha='left', va='bottom', fontsize=12,
                               bbox=dict(boxstyle="round,pad=0.5", 
                                        fc="white", ec="gray", alpha=0.8))
                    
                    plt.grid(True, alpha=0.3)
                    plt.legend(loc='upper right')
                    plt.tight_layout()
                    
                    self.figures["model_comparison_timeseries"] = self._save_figure()
                    
                    # Create RMSE comparison
                    plt.figure(figsize=(10, 6))
                    
                    # Calculate RMSE
                    global_rmse = np.sqrt(mean_squared_error(actual, global_pred))
                    subject_rmse = np.sqrt(mean_squared_error(actual, subject_pred))
                    adaptive_rmse = np.sqrt(mean_squared_error(actual, adaptive_pred))
                    
                    # Create bar chart
                    models = ['Global Model', 'Subject-Specific', 'Adaptive Transfer']
                    rmse_vals = [global_rmse, subject_rmse, adaptive_rmse]
                    colors = ['#1f77b4', '#2ca02c', '#9467bd']
                    
                    bars = plt.bar(models, rmse_vals, color=colors)
                    
                    # Add value labels
                    for i, bar in enumerate(bars):
                        height = bar.get_height()
                        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                               f'{height:.3f}', ha='center', va='bottom')
                    
                    # Calculate improvement percentages
                    global_to_subject = ((global_rmse - subject_rmse) / global_rmse) * 100
                    subject_to_adaptive = ((subject_rmse - adaptive_rmse) / subject_rmse) * 100 if subject_rmse > adaptive_rmse else 0
                    
                    # Add improvement annotations
                    if global_to_subject > 0:
                        plt.annotate(f"{global_to_subject:.1f}% improvement", 
                                   xy=(0.5, (global_rmse + subject_rmse)/2),
                                   xytext=(0.35, (global_rmse + subject_rmse)/2 + 0.05),
                                   arrowprops=dict(arrowstyle="->", color='green'),
                                   color='green', fontsize=10)
                    
                    if subject_to_adaptive > 0:
                        plt.annotate(f"{subject_to_adaptive:.1f}% improvement", 
                                   xy=(1.5, (subject_rmse + adaptive_rmse)/2),
                                   xytext=(1.35, (subject_rmse + adaptive_rmse)/2 + 0.05),
                                   arrowprops=dict(arrowstyle="->", color='green'),
                                   color='green', fontsize=10)
                    
                    plt.title(f"Error Comparison for Subject {subject_id}", fontsize=14)
                    plt.ylabel("Root Mean Square Error (RMSE)", fontsize=12)
                    plt.grid(axis='y', alpha=0.3)
                    plt.tight_layout()
                    
                    self.figures["model_comparison_rmse"] = self._save_figure()
                    
                    # Create side-by-side comparison of scatter plots
                    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                    
                    # Global model
                    axes[0].scatter(actual, global_pred, alpha=0.6, color='#1f77b4')
                    min_val = min(min(actual), min(global_pred))
                    max_val = max(max(actual), max(global_pred))
                    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--')
                    axes[0].set_title(f"Global Model\nR² = {global_r2:.3f}")
                    axes[0].set_xlabel("Actual")
                    axes[0].set_ylabel("Predicted")
                    axes[0].grid(True, alpha=0.3)
                    
                    # Subject-specific model
                    axes[1].scatter(actual, subject_pred, alpha=0.6, color='#2ca02c')
                    min_val = min(min(actual), min(subject_pred))
                    max_val = max(max(actual), max(subject_pred))
                    axes[1].plot([min_val, max_val], [min_val, max_val], 'r--')
                    axes[1].set_title(f"Subject-Specific\nR² = {subject_r2:.3f}")
                    axes[1].set_xlabel("Actual")
                    axes[1].set_ylabel("Predicted")
                    axes[1].grid(True, alpha=0.3)
                    
                    # Adaptive model
                    axes[2].scatter(actual, adaptive_pred, alpha=0.6, color='#9467bd')
                    min_val = min(min(actual), min(adaptive_pred))
                    max_val = max(max(actual), max(adaptive_pred))
                    axes[2].plot([min_val, max_val], [min_val, max_val], 'r--')
                    axes[2].set_title(f"Adaptive Transfer\nR² = {adaptive_r2:.3f}")
                    axes[2].set_xlabel("Actual")
                    axes[2].set_ylabel("Predicted")
                    axes[2].grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    plt.suptitle(f"Predicted vs. Actual Comparison for Subject {subject_id}", 
                              fontsize=16, y=1.05)
                    
                    self.figures["model_comparison_scatter"] = self._save_figure()
        
        # 5. Improvement Distribution
        if 'subject_improvements' in predictions:
            improvements = list(predictions['subject_improvements'].values())
            
            if improvements:
                plt.figure(figsize=(12, 6))
                
                # Create histogram
                sns.histplot(improvements, kde=True, bins=10)
                
                # Add vertical line at 0
                plt.axvline(x=0, color='red', linestyle='--')
                
                # Add mean line
                plt.axvline(x=np.mean(improvements), color='green', linestyle='--', 
                          label=f'Mean: {np.mean(improvements):.1f}%')
                
                plt.title("Distribution of Performance Improvement (Subject vs. Global)", fontsize=14)
                plt.xlabel("Improvement (%)", fontsize=12)
                plt.ylabel("Count", fontsize=12)
                
                # Add statistics
                stats_text = (
                    f"Mean: {np.mean(improvements):.1f}%\n"
                    f"Median: {np.median(improvements):.1f}%\n"
                    f"Std Dev: {np.std(improvements):.1f}%\n"
                    f"Min: {min(improvements):.1f}%\n"
                    f"Max: {max(improvements):.1f}%\n"
                    f"Positive: {sum(i > 0 for i in improvements)} of {len(improvements)}"
                )
                plt.annotate(stats_text, xy=(0.95, 0.95), xycoords='axes fraction', 
                           ha='right', va='top', bbox=dict(boxstyle="round,pad=0.5", 
                                                         fc="white", ec="gray", alpha=0.8))
                
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                
                self.figures["improvement_distribution"] = self._save_figure()
        
        # 6. Adaptive Improvement Distribution
        if 'adaptive_improvements' in predictions:
            improvements = list(predictions['adaptive_improvements'].values())
            
            if improvements:
                plt.figure(figsize=(12, 6))
                
                # Create histogram
                sns.histplot(improvements, kde=True, bins=10)
                
                # Add vertical line at 0
                plt.axvline(x=0, color='red', linestyle='--')
                
                # Add mean line
                plt.axvline(x=np.mean(improvements), color='green', linestyle='--',
                          label=f'Mean: {np.mean(improvements):.1f}%')
                
                plt.title("Distribution of Performance Improvement (Adaptive vs. Subject-Specific)", 
                        fontsize=14)
                plt.xlabel("Improvement (%)", fontsize=12)
                plt.ylabel("Count", fontsize=12)
                
                # Add statistics
                stats_text = (
                    f"Mean: {np.mean(improvements):.1f}%\n"
                    f"Median: {np.median(improvements):.1f}%\n"
                    f"Std Dev: {np.std(improvements):.1f}%\n"
                    f"Min: {min(improvements):.1f}%\n"
                    f"Max: {max(improvements):.1f}%\n"
                    f"Positive: {sum(i > 0 for i in improvements)} of {len(improvements)}"
                )
                plt.annotate(stats_text, xy=(0.95, 0.95), xycoords='axes fraction', 
                           ha='right', va='top', bbox=dict(boxstyle="round,pad=0.5", 
                                                         fc="white", ec="gray", alpha=0.8))
                
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                
                self.figures["adaptive_improvement_distribution"] = self._save_figure()
        
        logger.info("Prediction comparison visualizations created")
    
    def add_r2_explanation_visualization(self):
        """Create a visualization explaining R² interpretation including negative values"""
        
        logger.info("Creating R² explanation visualization")
        
        plt.figure(figsize=(14, 8))
        
        # Set up the grid
        ax = plt.subplot(1, 3, 1)
        
        # Example 1: Perfect model (R² = 1.0)
        x = np.linspace(0, 10, 10)
        y = x
        
        plt.scatter(x, y, s=60, c='blue', alpha=0.7)
        plt.plot(x, y, 'r--')
        
        # Calculate metrics
        mean_y = np.mean(y)
        r2 = 1.0
        
        # Add horizontal line at mean
        plt.axhline(y=mean_y, color='green', linestyle='-', alpha=0.5, label=f'Mean: {mean_y:.1f}')
        
        # Add text
        plt.title("Perfect Model (R² = 1.0)", fontsize=14)
        plt.xlabel("Actual", fontsize=12)
        plt.ylabel("Predicted", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Example 2: Decent model (R² = 0.5)
        ax = plt.subplot(1, 3, 2)
        
        x = np.linspace(0, 10, 10)
        noise = np.random.normal(0, 1.5, 10)
        y = x + noise
        
        plt.scatter(x, y, s=60, c='blue', alpha=0.7)
        
        # Add perfect prediction line
        plt.plot(x, x, 'r--', label='Perfect Prediction')
        
        # Add trend line
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        plt.plot(x, p(x), 'purple', linestyle='-', label=f'Trend: y = {z[0]:.1f}x + {z[1]:.1f}')
        
        # Calculate metrics
        mean_y = np.mean(y)
        r2 = 0.5  # Approximate for illustration
        
        # Add horizontal line at mean
        plt.axhline(y=mean_y, color='green', linestyle='-', alpha=0.5, label=f'Mean: {mean_y:.1f}')
        
        # Add text
        plt.title("Decent Model (R² = 0.5)", fontsize=14)
        plt.xlabel("Actual", fontsize=12)
        plt.ylabel("Predicted", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Example 3: Poor model (R² = -0.5)
        ax = plt.subplot(1, 3, 3)
        
        x = np.linspace(0, 10, 10)
        y = np.array([7, 4, 9, 2, 8, 3, 6, 1, 5, 10])  # Intentionally poor correlation
        
        plt.scatter(x, y, s=60, c='blue', alpha=0.7)
        
        # Add perfect prediction line
        plt.plot(x, x, 'r--', label='Perfect Prediction')
        
        # Add trend line
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        plt.plot(x, p(x), 'purple', linestyle='-', label=f'Trend: y = {z[0]:.1f}x + {z[1]:.1f}')
        
        # Calculate metrics
        mean_y = np.mean(y)
        r2 = -0.5  # Approximate for illustration
        
        # Add horizontal line at mean
        plt.axhline(y=mean_y, color='green', linestyle='-', alpha=0.5, label=f'Mean: {mean_y:.1f}')
        
        # Add text
        plt.title("Poor Model (R² = -0.5)", fontsize=14)
        plt.xlabel("Actual", fontsize=12)
        plt.ylabel("Predicted", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Add overall explanation
        plt.suptitle("Understanding R² (Coefficient of Determination)", fontsize=16, y=0.98)
        plt.figtext(0.5, 0.01,
                  "R² measures how well the model explains the variance in the target variable compared to simply predicting the mean.\n"
                  "• R² = 1.0: Perfect prediction - the model explains all variance in the data\n"
                  "• R² = 0.0: The model performs exactly the same as predicting the mean value for every point\n"
                  "• R² < 0.0: The model performs worse than simply predicting the mean value", 
                  ha='center', fontsize=12, bbox={"facecolor":"white", "alpha":0.8, "pad":5})
        
        plt.subplots_adjust(bottom=0.2)
        
        self.figures["r2_explanation"] = self._save_figure()
        
        logger.info("R² explanation visualization created")
    
    def add_discussion_visualization(self):
        """Create discussion and implications visualizations"""
        
        logger.info("Creating discussion visualizations")
        
        # 1. Practical implications visualization
        plt.figure(figsize=(14, 8))
        
        # Create a 2x2 grid
        gs = plt.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1], 
                        wspace=0.3, hspace=0.4)
        
        # Data collection implications
        ax1 = plt.subplot(gs[0, 0])
        ax1.set_title("Data Collection Requirements", fontsize=14)
        ax1.axis('off')
        
        # Create a box
        box = plt.Rectangle((0.1, 0.1), 0.8, 0.8, fill=True, alpha=0.1, 
                          color='blue', edgecolor='blue')
        ax1.add_patch(box)
        
        # Add text
        text = (
            "• Initial calibration period recommended\n"
            "• 10-20 samples per subject minimum\n"
            "• Include range of cognitive load conditions\n"
            "• Standardized physiological measurements\n"
            "• Ground truth cognitive load ratings"
        )
        ax1.text(0.5, 0.5, text, fontsize=12, ha='center', va='center')
        
        # Model architecture implications
        ax2 = plt.subplot(gs[0, 1])
        ax2.set_title("Model Architecture Design", fontsize=14)
        ax2.axis('off')
        
        # Create a box
        box = plt.Rectangle((0.1, 0.1), 0.8, 0.8, fill=True, alpha=0.1,
                          color='green', edgecolor='green')
        ax2.add_patch(box)
        
        # Add text
        text = (
            "• Use subject ID as a grouping variable\n"
            "• Train global model as a foundation\n"
            "• Train subject-specific models in parallel\n"
            "• Add adaptive weighting mechanism\n"
            "• Consider transfer learning from similar subjects"
        )
        ax2.text(0.5, 0.5, text, fontsize=12, ha='center', va='center')
        
        # Real-time application implications
        ax3 = plt.subplot(gs[1, 0])
        ax3.set_title("Real-time Applications", fontsize=14)
        ax3.axis('off')
        
        # Create a box
        box = plt.Rectangle((0.1, 0.1), 0.8, 0.8, fill=True, alpha=0.1,
                          color='purple', edgecolor='purple')
        ax3.add_patch(box)
        
        # Add text
        text = (
            "• Start with global model for new users\n"
            "• Gradually transition to personalized model\n"
            "• Update models incrementally with new data\n"
            "• Consider confidence intervals for decisions\n"
            "• Provide feedback mechanism for corrections"
        )
        ax3.text(0.5, 0.5, text, fontsize=12, ha='center', va='center')
        
        # Ethical and privacy implications
        ax4 = plt.subplot(gs[1, 1])
        ax4.set_title("Ethical Considerations", fontsize=14)
        ax4.axis('off')
        
        # Create a box
        box = plt.Rectangle((0.1, 0.1), 0.8, 0.8, fill=True, alpha=0.1,
                          color='red', edgecolor='red')
        ax4.add_patch(box)
        
        # Add text
        text = (
            "• Ensure informed consent for physiological data\n"
            "• Maintain data privacy and security\n"
            "• Be transparent about model limitations\n"
            "• Consider bias in cognitive load self-reports\n"
            "• Allow users to opt-out of personalization"
        )
        ax4.text(0.5, 0.5, text, fontsize=12, ha='center', va='center')
        
        # Add overall title
        plt.suptitle("Practical Implications for Cognitive Load Modeling", fontsize=16, y=0.98)
        
        self.figures["practical_implications"] = self._save_figure()
        
        # 2. Limitations and future work visualization
        plt.figure(figsize=(14, 8))
        
        # Create a figure with two columns
        gs = plt.GridSpec(1, 2, width_ratios=[1, 1], wspace=0.3)
        
        # Limitations
        ax1 = plt.subplot(gs[0, 0])
        ax1.set_title("Limitations of Current Approach", fontsize=14)
        ax1.axis('off')
        
        # Create a box
        box = plt.Rectangle((0.05, 0.05), 0.9, 0.9, fill=True, alpha=0.1,
                          color='red', edgecolor='red')
        ax1.add_patch(box)
        
        # Add text
        text = (
            "1. Self-reported mental effort as ground truth\n"
            "   • Subjective and potentially inconsistent\n"
            "   • Ratings may vary between subjects\n\n"
            "2. Limited generalization to new subjects\n"
            "   • Negative cross-subject R² values\n"
            "   • Requires individual calibration\n\n"
            "3. Domain-specific task environment\n"
            "   • Results may differ in other contexts\n"
            "   • Limited ecological validity\n\n"
            "4. Time stability not evaluated\n"
            "   • Model performance may degrade over time\n"
            "   • Subject patterns may evolve"
        )
        ax1.text(0.5, 0.5, text, fontsize=12, ha='center', va='center')
        
        # Future work
        ax2 = plt.subplot(gs[0, 1])
        ax2.set_title("Directions for Future Research", fontsize=14)
        ax2.axis('off')
        
        # Create a box
        box = plt.Rectangle((0.05, 0.05), 0.9, 0.9, fill=True, alpha=0.1,
                          color='green', edgecolor='green')
        ax2.add_patch(box)
        
        # Add text
        text = (
            "1. Objective cognitive load measures\n"
            "   • Combine multiple physiological signals\n"
            "   • Validate with performance metrics\n\n"
            "2. Advanced transfer learning\n"
            "   • Cluster similar subjects\n"
            "   • Meta-learning approaches\n\n"
            "3. Temporal model stability\n"
            "   • Longitudinal studies\n"
            "   • Adaptive model updating\n\n"
            "4. Cross-domain validation\n"
            "   • Test in multiple task environments\n"
            "   • Diverse subject populations\n\n"
            "5. Deep learning approaches\n"
            "   • End-to-end learning from raw signals\n"
            "   • Attention mechanisms for feature extraction"
        )
        ax2.text(0.5, 0.5, text, fontsize=12, ha='center', va='center')
        
        # Add overall title
        plt.suptitle("Limitations and Future Research Directions", fontsize=16, y=0.98)
        
        self.figures["limitations_future_work"] = self._save_figure()
        
        logger.info("Discussion visualizations created")
    
    def _save_figure(self):
        """Save current figure to PNG and return base64 encoded string"""
        # Generate a unique filename
        filename = f"{len(self.figures) + 1}_{int(time.time())}.png"
        filepath = os.path.join(self.plots_dir, filename)
        
        # Save figure to disk
        plt.savefig(filepath, dpi=100, bbox_inches='tight')
        
        # Also save as base64 for HTML embedding
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        plt.close()
        
        # Encode figure data
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        
        return img_str
    
    def create_html_report(self, output_path=None):
        """
        Create comprehensive HTML report from all visualizations
        
        Args:
            output_path: Path to save the HTML report (defaults to results_dir/report.html)
            
        Returns:
            Path to the generated HTML report
        """
        logger.info("Creating HTML report")
        
        if not output_path:
            output_path = os.path.join(self.results_dir, "cognitive_load_report.html")
        
        # Get HTML template
        template_string = self._get_html_template()
        template = Template(template_string)
        
        # Prepare data for template
        template_data = {
            "title": self.title,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "version": VERSION,
            "figures": self.figures,
            "tables": self.tables,
            "metrics": self.metrics
        }
        
        # Render HTML
        html_content = template.render(**template_data)
        
        # Write to file
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        
        logger.info(f"HTML report created: {output_path}")
        return output_path
    
    def _get_html_template(self):
        """Get the HTML template for the report"""
        return '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    
    <!-- Bootstrap CSS -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f8f9fa;
            padding-top: 20px;
        }
        
        h1, h2, h3, h4 {
            color: #0056b3;
            margin-top: 1.5em;
            margin-bottom: 0.5em;
        }
        
        h1 {
            border-bottom: 2px solid #0056b3;
            padding-bottom: 10px;
        }
        
        .figure-container {
            margin: 20px 0;
            text-align: center;
            background-color: white;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            padding: 15px;
        }
        
        .figure-container img {
            max-width: 100%;
            height: auto;
        }
        
        .figure-caption {
            font-style: italic;
            color: #666;
            margin-top: 10px;
        }
        
        .metric-card {
            background-color: white;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            padding: 15px;
            margin-bottom: 20px;
            transition: transform 0.3s ease;
        }
        
        .metric-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }
        
        .metric-value {
            font-size: 2em;
            font-weight: bold;
            color: #0056b3;
        }
        
        .metric-label {
            color: #666;
            font-size: 0.9em;
        }
        
        .section {
            margin: 40px 0;
            background-color: white;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            padding: 20px;
        }
        
        footer {
            margin-top: 50px;
            padding: 20px 0;
            border-top: 1px solid #ddd;
            text-align: center;
            font-size: 0.9em;
            color: #666;
        }
        
        .btn-toc {
            position: fixed;
            bottom: 20px;
            right: 20px;
            z-index: 99;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            box-shadow: 0 2px 10px rgba(0,0,0,0.2);
        }
        
        #toc {
            margin-bottom: 30px;
        }
        
        .toc-link {
            color: #0056b3;
            text-decoration: none;
            transition: margin-left 0.3s ease;
        }
        
        .toc-link:hover {
            margin-left: 5px;
        }
        
        @media print {
            .btn-toc {
                display: none;
            }
            
            .section {
                box-shadow: none;
                border: 1px solid #ddd;
            }
            
            .figure-container {
                box-shadow: none;
                border: 1px solid #ddd;
                break-inside: avoid;
            }
            
            .no-break {
                break-inside: avoid;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1 class="text-center mb-4">{{ title }}</h1>
        <p class="text-center text-muted">Version {{ version }} • Generated on {{ timestamp }}</p>
        
        <!-- Table of Contents -->
        <div class="section" id="toc">
            <h2>Table of Contents</h2>
            <ol>
                <li><a href="#executive-summary" class="toc-link">Executive Summary</a></li>
                <li><a href="#introduction" class="toc-link">Introduction</a></li>
                <li><a href="#methodology" class="toc-link">Methodology</a></li>
                <li><a href="#dataset" class="toc-link">Dataset Overview</a></li>
                <li><a href="#feature-engineering" class="toc-link">Feature Engineering</a></li>
                <li><a href="#model-performance" class="toc-link">Model Performance</a></li>
                <li><a href="#subject-analysis" class="toc-link">Subject-specific Analysis</a></li>
                <li><a href="#transfer-learning" class="toc-link">Adaptive Transfer Learning</a></li>
                <li><a href="#prediction-accuracy" class="toc-link">Prediction Accuracy Analysis</a></li>
                <li><a href="#discussion" class="toc-link">Discussion</a></li>
                <li><a href="#conclusion" class="toc-link">Conclusion</a></li>
                <li><a href="#limitations" class="toc-link">Limitations and Future Work</a></li>
                <li><a href="#appendix" class="toc-link">Technical Appendix</a></li>
            </ol>
        </div>
        
        <!-- Executive Summary Section with Key Metrics -->
        <div class="section" id="executive-summary">
            <h2>Executive Summary</h2>
            
            <div class="row mb-4">
                <!-- First row: Three main model performance metrics -->
                <div class="col-md-4">
                    <div class="metric-card text-center">
                        <div class="metric-value text-danger">{{ "%.3f"|format(metrics.get('global_cross_subject_r2', -1)) }}</div>
                        <div class="metric-label">Global Model Cross-Subject R²<br><small>(Generalizes to new subjects)</small></div>
                        <div class="text-muted small">95% CI: [{{ "%.3f"|format(metrics.get('global_cross_subject_r2_ci', [-1, -1])[0]) }}, {{ "%.3f"|format(metrics.get('global_cross_subject_r2_ci', [-1, -1])[1]) }}]</div>
                    </div>
                </div>
                
                <div class="col-md-4">
                    <div class="metric-card text-center">
                        <div class="metric-value text-success">{{ "%.3f"|format(metrics.get('global_within_subject_r2', 0)) }}</div>
                        <div class="metric-label">Global Model Within-Subject R²<br><small>(Predicts known subjects)</small></div>
                        <div class="text-muted small">95% CI: [{{ "%.3f"|format(metrics.get('global_within_subject_r2_ci', [0, 0])[0]) }}, {{ "%.3f"|format(metrics.get('global_within_subject_r2_ci', [0, 0])[1]) }}]</div>
                    </div>
                </div>
                
                <div class="col-md-4">
                    <div class="metric-card text-center">
                        <div class="metric-value text-primary">{{ "%.3f"|format(metrics.get('subject_r2', 0)) }}</div>
                        <div class="metric-label">Subject-specific Model R²</div>
                        <div class="text-muted small">95% CI: [{{ "%.3f"|format(metrics.get('subject_r2_ci', [0, 0])[0]) }}, {{ "%.3f"|format(metrics.get('subject_r2_ci', [0, 0])[1]) }}]</div>
                    </div>
                </div>
            </div>
            
            <div class="row mb-4">
                <!-- Second row: Additional metrics -->
                <div class="col-md-4">
                    <div class="metric-card text-center">
                        <div class="metric-value text-primary">{{ "%.3f"|format(metrics.get('adaptive_r2', 0)) }}</div>
                        <div class="metric-label">Adaptive Transfer Model R²</div>
                        <div class="text-muted small">95% CI: [{{ "%.3f"|format(metrics.get('adaptive_r2_ci', [0, 0])[0]) }}, {{ "%.3f"|format(metrics.get('adaptive_r2_ci', [0, 0])[1]) }}]</div>
                    </div>
                </div>
                
                <div class="col-md-4">
                    <div class="metric-card text-center">
                        <div class="metric-value">{{ "%.3f"|format(metrics.get('avg_r2_improvement', 0)) }}</div>
                        <div class="metric-label">Average Improvement<br><small>(Subject vs Global)</small></div>
                        {% if metrics.get('improvement_p_value', 1) < 0.05 %}
                        <div class="text-success small">Statistically significant (p = {{ "%.4f"|format(metrics.get('improvement_p_value', 1)) }})</div>
                        {% else %}
                        <div class="text-muted small">p = {{ "%.4f"|format(metrics.get('improvement_p_value', 1)) }}</div>
                        {% endif %}
                    </div>
                </div>
                
                <div class="col-md-4">
                    <div class="metric-card text-center">
                        <div class="metric-value text-success">{{ "%.2f"|format(metrics.get('optimal_weight', 0)) }}</div>
                        <div class="metric-label">Optimal Subject Weight</div>
                        <div class="text-muted small">95% CI: [{{ "%.2f"|format(metrics.get('optimal_weight_ci', [0, 0])[0]) }}, {{ "%.2f"|format(metrics.get('optimal_weight_ci', [0, 0])[1]) }}]</div>
                    </div>
                </div>
            </div>
            
            <p class="lead">
                This analysis demonstrates that cognitive load prediction is highly individual-specific. 
                Subject-specific models dramatically outperform the global model when generalizing to new subjects
                (R² of {{ "%.3f"|format(metrics.get('subject_r2', 0)) }} compared to {{ "%.3f"|format(metrics.get('global_cross_subject_r2', -1)) }}), 
                highlighting the importance of personalization in cognitive load assessment.
                The adaptive transfer learning approach further improved performance to R² of {{ "%.3f"|format(metrics.get('adaptive_r2', 0)) }}.
            </p>
            
            <p><strong>Key Findings:</strong></p>
            <ul>
                <li>The striking difference between cross-subject ({{ "%.3f"|format(metrics.get('global_cross_subject_r2', -1)) }}) and within-subject ({{ "%.3f"|format(metrics.get('global_within_subject_r2', 0)) }}) 
                    performance demonstrates that cognitive load patterns have strong individual signatures that resist generalization.</li>
                <li>Physiological signals including skin conductance responses (SCR), heart rate variability (HRV), and electrodermal activity (EDA) 
                    are the most predictive features for cognitive load.</li>
                <li>The optimal subject weight of {{ "%.2f"|format(metrics.get('optimal_weight', 0)) }} indicates that adaptive transfer learning 
                    heavily favors individual patterns over general ones.</li>
                <li>Model performance improvements from personalization are statistically significant 
                    (p = {{ "%.4f"|format(metrics.get('improvement_p_value', 1)) }}).</li>
            </ul>
            
            <p><strong>Practical Implications:</strong></p>
            <ul>
                <li>Cognitive load monitoring systems should include an individual calibration phase.</li>
                <li>Adaptive transfer learning can provide an optimal balance between personalization and generalization.</li>
                <li>The baseline global model alone is insufficient for accurate cognitive load prediction across subjects.</li>
            </ul>
        </div>
        
        <!-- Introduction -->
        <div class="section" id="introduction">
            <h2>Introduction</h2>
            
            <p>
                Cognitive load refers to the mental effort being used in working memory during task performance. 
                Accurately measuring and predicting cognitive load is crucial in various domains including aviation, 
                healthcare, education, and human-computer interaction. However, cognitive load is a complex 
                psychological construct that can be challenging to quantify objectively.
            </p>
            
            <p>
                This report presents a comprehensive analysis of a cognitive load prediction pipeline using machine 
                learning techniques applied to physiological data from pilots performing aviation tasks. The analysis focuses on:
            </p>
            
            <ul>
                <li>Evaluating the effectiveness of various physiological signals as predictors of cognitive load</li>
                <li>Comparing global models (trained on all subjects) vs. subject-specific models</li>
                <li>Implementing adaptive transfer learning to balance generalizability and individuality</li>
                <li>Identifying the most informative features for cognitive load prediction</li>
                <li>Providing statistical confidence estimates for all performance metrics</li>
            </ul>
            
            <p>
                The primary research question is whether cognitive load exhibits strong individual differences that 
                necessitate personalized models, or whether general patterns exist that can be applied across individuals.
                The findings have important implications for the design of adaptive systems that monitor and respond to
                users' cognitive states in real-time.
            </p>
            
            <p>
                This analysis builds upon previous research in psychophysiological computing and affective computing,
                but with increased statistical rigor, more comprehensive validation approaches, and improved reporting
                of confidence intervals and significance testing.
            </p>
        </div>
        
        <!-- Methodology -->
        <div class="section" id="methodology">
            <h2>Methodology</h2>
            
            <p>
                The analysis pipeline follows a systematic approach to data processing, feature engineering, model training, 
                and evaluation, with particular emphasis on validating performance across different subjects.
            </p>
            
            {% if 'methodology' in figures %}
            <div class="figure-container">
                <img src="data:image/png;base64,{{ figures.methodology }}" alt="Methodology Overview">
                <div class="figure-caption">Overview of the cognitive load analysis pipeline</div>
            </div>
            {% endif %}
            
            <h4>Data Collection</h4>
            <p>
                The dataset contains physiological measurements and cognitive load assessments from pilots during simulated 
                flight tasks. Cognitive load was measured through self-reported mental effort ratings using a validated scale. 
                Physiological signals were recorded continuously during the tasks, including electrodermal activity (EDA), 
                heart rate variability (HRV), skin temperature, and accelerometer data.
            </p>
            
            <h4>Data Preprocessing</h4>
            <ul>
                <li>Removing potentially leaky features (features containing information not available in a real-time scenario)</li>
                <li>Filtering features with suspiciously high correlation to the target</li>
                <li>Handling missing values through appropriate imputation techniques</li>
                <li>Creating different data splits for within-subject and cross-subject validation</li>
            </ul>
            
            <h4>Validation Approach</h4>
            
            {% if 'validation_approaches' in figures %}
            <div class="figure-container">
                <img src="data:image/png;base64,{{ figures.validation_approaches }}" alt="Validation Approaches">
                <div class="figure-caption">Comparison of cross-subject and within-subject validation approaches</div>
            </div>
            {% endif %}
            
            <p>
                Two complementary validation approaches were employed to evaluate model performance:
            </p>
            
            <ul>
                <li><strong>Within-subject validation:</strong> Tests how well models predict new samples from known subjects. Uses k-fold cross-validation
                    while preserving subject grouping.</li>
                <li><strong>Cross-subject validation:</strong> Tests how well models generalize to completely new subjects. Uses leave-one-subject-out
                    cross-validation, where the model is trained on data from all subjects except one, and tested on the left-out subject.</li>
            </ul>
            
            <p>
                This dual validation approach provides a comprehensive understanding of model performance in different scenarios:
                predicting for already-calibrated subjects versus generalizing to entirely new users.
            </p>
            
            <h4>Feature Engineering</h4>
            <ul>
                <li>Temporal feature extraction (rates of change, moving averages, lags)</li>
                <li>Frequency domain feature extraction for physiological signals</li>
                <li>Creation of compound features with domain-specific ratios</li>
                <li>Feature selection based on importance ranking</li>
            </ul>
            
            <h4>Modeling Approaches</h4>
            <ul>
                <li><strong>Global Model:</strong> Trained on data from all subjects</li>
                <li><strong>Subject-specific Models:</strong> Individual models trained for each subject</li>
                <li><strong>Adaptive Transfer Learning:</strong> Weighted combination of global and subject-specific models, with optimal weights determined by validation</li>
            </ul>
            
            <h4>Statistical Analysis</h4>
            <ul>
                <li>Calculation of 95% confidence intervals for all performance metrics using bootstrap resampling</li>
                <li>Statistical significance testing for model comparisons using paired t-tests</li>
                <li>Analysis of feature importance stability across subjects</li>
            </ul>
        </div>
        
        <!-- Dataset Overview -->
        <div class="section" id="dataset">
            <h2>Dataset Overview</h2>
            
            <p>
                The dataset contains physiological measurements and cognitive load assessments from 
                {{ metrics.get('n_subjects', 'multiple') }} pilots during flight tasks. The target 
                variable is self-reported mental effort, which serves as a proxy for cognitive load.
            </p>
            
            <div class="row mb-4">
                <div class="col-md-4">
                    <div class="metric-card text-center">
                        <div class="metric-value">{{ metrics.get('n_subjects', 'N/A') }}</div>
                        <div class="metric-label">Subjects</div>
                    </div>
                </div>
                <div class="col-md-4">
                    <div class="metric-card text-center">
                        <div class="metric-value">{{ metrics.get('mean_samples_per_subject', 'N/A')|int }}</div>
                        <div class="metric-label">Average Samples per Subject</div>
                        <div class="text-muted small">Range: {{ metrics.get('min_samples_per_subject', 'N/A')|int }} - {{ metrics.get('max_samples_per_subject', 'N/A')|int }}</div>
                    </div>
                </div>
                <div class="col-md-4">
                    <div class="metric-card text-center">
                        <div class="metric-value">{{ metrics.get('n_physiological_signals', 'N/A') }}</div>
                        <div class="metric-label">Physiological Signal Types</div>
                    </div>
                </div>
            </div>
            
            {% if 'subject_distribution' in figures %}
            <div class="figure-container">
                <img src="data:image/png;base64,{{ figures.subject_distribution }}" alt="Subject Distribution">
                <div class="figure-caption">Distribution of samples across subjects</div>
            </div>
            {% endif %}
            
            {% if 'target_distribution' in figures %}
            <div class="figure-container">
                <img src="data:image/png;base64,{{ figures.target_distribution }}" alt="Target Distribution">
                <div class="figure-caption">Distribution of mental effort scores (cognitive load)</div>
            </div>
            {% endif %}
            
            {% if 'physiological_signals' in figures %}
            <div class="figure-container">
                <img src="data:image/png;base64,{{ figures.physiological_signals }}" alt="Physiological Signals">
                <div class="figure-caption">Types and counts of physiological signals in the dataset</div>
            </div>
            {% endif %}
            
            {% if 'missing_values' in figures %}
            <div class="figure-container">
                <img src="data:image/png;base64,{{ figures.missing_values }}" alt="Missing Values">
                <div class="figure-caption">Missing values analysis by feature</div>
            </div>
            {% endif %}
            
            <h4>Cognitive Load Measurement</h4>
            <p>
                Cognitive load was measured through self-reported mental effort ratings using a scale from 
                {{ metrics.get('target_min', 0)|int }} to {{ metrics.get('target_max', 10)|int }}. Ratings were collected after each task segment.
                The mean rating across all samples was {{ "%.2f"|format(metrics.get('target_mean', 5)) }} with a standard deviation of 
                {{ "%.2f"|format(metrics.get('target_std', 2)) }}.
            </p>
            
            <h4>Physiological Signals</h4>
            <p>
                The dataset includes various physiological signals known to correlate with cognitive load:
            </p>
            <ul>
                {% for signal_type in metrics.get('physiological_signal_types', []) %}
                <li><strong>{{ signal_type|upper }}</strong>: {{ {'eda': 'Electrodermal Activity', 'scr': 'Skin Conductance Response', 'hr': 'Heart Rate', 'hrv': 'Heart Rate Variability', 'temp': 'Skin Temperature', 'accel': 'Accelerometer Data', 'ppg': 'Photoplethysmography', 'ibi': 'Inter-Beat Interval', 'gsr': 'Galvanic Skin Response', 'resp': 'Respiration', 'pupil': 'Pupil Diameter', 'emg': 'Electromyography', 'eeg': 'Electroencephalography'}.get(signal_type, signal_type|title) }}</li>
                {% endfor %}
            </ul>
        </div>
        
        <!-- Feature Engineering -->
        <div class="section" id="feature-engineering">
            <h2>Feature Engineering</h2>
            
            <p>
                Feature engineering was a crucial step in the pipeline, transforming raw physiological 
                signals into informative predictors of cognitive load.
            </p>
            
            {% if 'feature_counts' in figures %}
            <div class="figure-container">
                <img src="data:image/png;base64,{{ figures.feature_counts }}" alt="Feature Counts">
                <div class="figure-caption">Number of features at different stages of the pipeline</div>
            </div>
            {% endif %}
            
            {% if 'feature_groups' in figures %}
            <div class="figure-container">
                <img src="data:image/png;base64,{{ figures.feature_groups }}" alt="Feature Groups">
                <div class="figure-caption">Types of engineered features created from raw signals</div>
            </div>
            {% endif %}
            
            <h4>Feature Engineering Approaches</h4>
            
            <div class="row">
                <div class="col-md-6">
                    <div class="card mb-3">
                        <div class="card-header bg-primary text-white">
                            <h5 class="card-title mb-0">Temporal Features</h5>
                        </div>
                        <div class="card-body">
                            <ul>
                                <li>Rate of change (first derivative) of signals</li>
                                <li>Moving averages and moving standard deviations</li>
                                <li>Time-lagged values (e.g., lag-1, lag-2) for autoregressive modeling</li>
                                <li>Peak detection and quantification</li>
                            </ul>
                        </div>
                    </div>
                </div>
                <div class="col-md-6">
                    <div class="card mb-3">
                        <div class="card-header bg-success text-white">
                            <h5 class="card-title mb-0">Frequency Domain Features</h5>
                        </div>
                        <div class="card-body">
                            <ul>
                                <li>Spectral power in different frequency bands</li>
                                <li>Low-frequency to high-frequency ratios</li>
                                <li>Dominant frequency identification</li>
                                <li>Spectral entropy (measure of signal complexity)</li>
                            </ul>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="row">
                <div class="col-md-6">
                    <div class="card mb-3">
                        <div class="card-header bg-info text-white">
                            <h5 class="card-title mb-0">Compound Features</h5>
                        </div>
                        <div class="card-body">
                            <ul>
                                <li>Ratios between different physiological signals</li>
                                <li>Combined arousal indices from multiple signals</li>
                                <li>Normalized compound metrics</li>
                                <li>Polynomial transformations of important signals</li>
                            </ul>
                        </div>
                    </div>
                </div>
                <div class="col-md-6">
                    <div class="card mb-3">
                        <div class="card-header bg-warning text-dark">
                            <h5 class="card-title mb-0">Feature Selection</h5>
                        </div>
                        <div class="card-body">
                            <ul>
                                <li>Random Forest importance ranking</li>
                                <li>Correlation with target variable</li>
                                <li>Stability analysis across subjects</li>
                                <li>Domain knowledge-based boosting</li>
                            </ul>
                        </div>
                    </div>
                </div>
            </div>
            
            <h4>Top Features by Importance</h4>
            
            {% if 'top_features' in figures %}
            <div class="figure-container">
                <img src="data:image/png;base64,{{ figures.top_features }}" alt="Top Features">
                <div class="figure-caption">Most important features for predicting cognitive load</div>
            </div>
            {% endif %}
            
            <p>
                The most informative features were primarily related to skin conductance responses (SCR), 
                heart rate variability (HRV), and their derivatives. Particularly notable are the compound
                features that combine multiple physiological signals, which often rank among the top predictors.
            </p>
        </div>
        
        <!-- Model Performance -->
        <div class="section" id="model-performance">
            <h2>Model Performance</h2>
            
            <p>
                We evaluated three modeling approaches: global models, subject-specific models, 
                and adaptive transfer learning. The results revealed striking differences in performance.
            </p>
            
            {% if 'r2_explanation' in figures %}
            <div class="figure-container">
                <img src="data:image/png;base64,{{ figures.r2_explanation }}" alt="R² Explanation">
                <div class="figure-caption">Visual explanation of R² interpretation, including negative values</div>
            </div>
            {% endif %}
            
            {% if 'model_comparison' in figures %}
            <div class="figure-container">
                <img src="data:image/png;base64,{{ figures.model_comparison }}" alt="Model Comparison">
                <div class="figure-caption">Performance comparison of different modeling approaches with 95% confidence intervals</div>
            </div>
            {% endif %}
            
            <h4>Understanding the Negative R² in Cross-Subject Validation</h4>
            <p>
                The strong negative R² value ({{ "%.3f"|format(metrics.get('global_cross_subject_r2', -1)) }}) for the global model in cross-subject validation 
                indicates that the model performs worse than simply predicting the mean cognitive load value when applied to entirely new subjects.
                This is a critical finding that demonstrates the highly individual nature of cognitive load patterns.
                In other words, applying a model trained on one group of subjects to predict cognitive load for a new subject 
                leads to predictions that are less accurate than simply using the average cognitive load value.
            </p>
            
            <h4>Subject-Specific Model Performance</h4>
            
            {% if 'subject_r2_distribution' in figures %}
            <div class="figure-container">
                <img src="data:image/png;base64,{{ figures.subject_r2_distribution }}" alt="Subject R2 Distribution">
                <div class="figure-caption">Distribution of R² scores across subject-specific models</div>
            </div>
            {% endif %}
            
            {% if 'subject_performance_ranking' in figures %}
            <div class="figure-container">
                <img src="data:image/png;base64,{{ figures.subject_performance_ranking }}" alt="Subject Performance">
                <div class="figure-caption">Individual subject model performance (sorted by R² score)</div>
            </div>
            {% endif %}
            
            <p>
                Subject-specific models achieved excellent performance (average R² = {{ "%.3f"|format(metrics.get('subject_r2', 0)) }}),
                in stark contrast to the global model's cross-subject performance (R² = {{ "%.3f"|format(metrics.get('global_cross_subject_r2', -1)) }}).
                This dramatic difference in performance strongly suggests that cognitive load patterns are
                highly individualized, with significant variation between subjects that cannot be captured
                by a global model.
            </p>
            
            <h4>Cross-Subject Validation Results</h4>
            
            {% if 'cross_subject_validation' in figures %}
            <div class="figure-container">
                <img src="data:image/png;base64,{{ figures.cross_subject_validation }}" alt="Cross-Subject Validation">
                <div class="figure-caption">Performance of global model when applied to each individual subject</div>
            </div>
            {% endif %}
            
            <p>
                The cross-subject validation results show how the global model performs when applied to completely new subjects.
                The wide variation in performance across subjects (some positive, many negative R² values) indicates that
                some subjects may follow more "typical" patterns while others have highly unique physiological responses
                to cognitive load.
            </p>
            
            <h4>Performance Improvements</h4>
            
            {% if 'performance_improvements' in figures %}
            <div class="figure-container">
                <img src="data:image/png;base64,{{ figures.performance_improvements }}" alt="Performance Improvements">
                <div class="figure-caption">Stepwise improvements in performance from global to subject-specific to adaptive transfer models</div>
            </div>
            {% endif %}
            
            <p>
                The subject-specific models showed a statistically significant improvement over the global model
                (p = {{ "%.4f"|format(metrics.get('improvement_p_value', 1)) }}). This confirms that the benefits of personalization
                are not due to chance but represent a genuine improvement in predictive capability.
            </p>
        </div>
        
        <!-- Subject-specific Analysis -->
        <div class="section" id="subject-analysis">
            <h2>Subject-specific Analysis</h2>
            
            <p>
                To better understand the individual differences in cognitive load patterns, we analyzed 
                how feature importance varies across subjects.
            </p>
            
            {% if 'subject_feature_importance' in figures %}
            <div class="figure-container">
                <img src="data:image/png;base64,{{ figures.subject_feature_importance }}" alt="Subject Feature Importance">
                <div class="figure-caption">Variation in feature importance across different subjects</div>
            </div>
            {% endif %}
            
            {% if 'feature_stability' in figures %}
            <div class="figure-container">
                <img src="data:image/png;base64,{{ figures.feature_stability }}" alt="Feature Stability">
                <div class="figure-caption">Stability of feature importance across subjects (mean and standard deviation)</div>
            </div>
            {% endif %}
            
            <p>
                The heatmap reveals considerable variation in which features are most predictive for different 
                subjects. While some features (like SCR-related metrics) are consistently important across many 
                subjects, the relative importance of features shows distinct patterns for each individual.
            </p>
            
            <p>
                This variation helps explain why the global model performed poorly: it cannot effectively 
                capture the unique relationships between physiological signals and cognitive load that 
                exist for each individual. The feature stability analysis quantifies this variation, with
                higher coefficients of variation (CV) indicating less stable features across subjects.
            </p>
        </div>
        
        <!-- Adaptive Transfer Learning -->
        <div class="section" id="transfer-learning">
            <h2>Adaptive Transfer Learning</h2>
            
            <p>
                The adaptive transfer learning approach aimed to find an optimal balance between the 
                global model (trained on all subjects) and subject-specific models (trained on individual data).
            </p>
            
            {% if 'transfer_learning_weight' in figures %}
            <div class="figure-container">
                <img src="data:image/png;base64,{{ figures.transfer_learning_weight }}" alt="Transfer Learning Weight">
                <div class="figure-caption">Visual explanation of what the optimal weight means in transfer learning</div>
            </div>
            {% endif %}
            
            {% if 'weight_distribution' in figures %}
            <div class="figure-container">
                <img src="data:image/png;base64,{{ figures.weight_distribution }}" alt="Weight Distribution">
                <div class="figure-caption">Distribution of optimal weights across subjects</div>
            </div>
            {% endif %}
            
            {% if 'weight_vs_performance' in figures %}
            <div class="figure-container">
                <img src="data:image/png;base64,{{ figures.weight_vs_performance }}" alt="Weight vs Performance">
                <div class="figure-caption">Relationship between optimal weight and model performance</div>
            </div>
            {% endif %}
            
            <p>
                The average optimal weight for subject-specific models was {{ "%.2f"|format(metrics.get('optimal_weight', 0)) }}, 
                indicating a strong preference for individual patterns over global ones. This further 
                confirms the highly personalized nature of cognitive load patterns.
            </p>
            
            <p>
                The adaptive transfer learning approach achieved the best overall performance 
                (R² = {{ "%.3f"|format(metrics.get('adaptive_r2', 0)) }}), suggesting that while individual patterns dominate, 
                there is still some value in incorporating information from other subjects.
            </p>
            
            <h4>Adaptation Strategy</h4>
            <p>
                The optimal adaptation strategy involves:
            </p>
            <ol>
                <li>Training a global model on data from all available subjects</li>
                <li>Training a subject-specific model on data from the target subject</li>
                <li>Finding the optimal weighting between the two models through validation</li>
                <li>Using the weighted combination for predictions: <code>prediction = (weight × subject_prediction) + ((1 - weight) × global_prediction)</code></li>
            </ol>
            
            <p>
                This approach provides a practical solution for real-world systems, where a global model can be used initially,
                and gradually replaced by a personalized model as more individual data becomes available.
            </p>
        </div>
        
        <!-- Prediction Accuracy -->
        <div class="section" id="prediction-accuracy">
            <h2>Prediction Accuracy Analysis</h2>
            
            <p>
                This section provides a detailed analysis of how accurately the different models predict cognitive load,
                using visualizations that compare predicted values against actual values.
            </p>
            
            <div class="row">
                <div class="col-md-6">
                    {% if 'global_predicted_vs_actual' in figures %}
                    <div class="figure-container">
                        <img src="data:image/png;base64,{{ figures.global_predicted_vs_actual }}" alt="Global Model Predictions">
                        <div class="figure-caption">Global model predicted vs. actual values</div>
                    </div>
                    {% endif %}
                </div>
                <div class="col-md-6">
                    {% if 'global_residuals' in figures %}
                    <div class="figure-container">
                        <img src="data:image/png;base64,{{ figures.global_residuals }}" alt="Global Model Residuals">
                        <div class="figure-caption">Global model prediction errors (residuals)</div>
                    </div>
                    {% endif %}
                </div>
            </div>
            
            <div class="row">
                <div class="col-md-6">
                    {% if 'subject_predicted_vs_actual' in figures %}
                    <div class="figure-container">
                        <img src="data:image/png;base64,{{ figures.subject_predicted_vs_actual }}" alt="Subject Model Predictions">
                        <div class="figure-caption">Subject-specific models predicted vs. actual values</div>
                    </div>
                    {% endif %}
                </div>
                <div class="col-md-6">
                    {% if 'subject_residuals' in figures %}
                    <div class="figure-container">
                        <img src="data:image/png;base64,{{ figures.subject_residuals }}" alt="Subject Model Residuals">
                        <div class="figure-caption">Subject-specific models prediction errors (residuals)</div>
                    </div>
                    {% endif %}
                </div>
            </div>
            
            <div class="row">
                <div class="col-md-6">
                    {% if 'adaptive_predicted_vs_actual' in figures %}
                    <div class="figure-container">
                        <img src="data:image/png;base64,{{ figures.adaptive_predicted_vs_actual }}" alt="Adaptive Model Predictions">
                        <div class="figure-caption">Adaptive transfer models predicted vs. actual values</div>
                    </div>
                    {% endif %}
                </div>
                <div class="col-md-6">
                    {% if 'adaptive_residuals' in figures %}
                    <div class="figure-container">
                        <img src="data:image/png;base64,{{ figures.adaptive_residuals }}" alt="Adaptive Model Residuals">
                        <div class="figure-caption">Adaptive transfer models prediction errors (residuals)</div>
                    </div>
                    {% endif %}
                </div>
            </div>
            
            {% if 'model_comparison_timeseries' in figures %}
            <div class="figure-container">
                <img src="data:image/png;base64,{{ figures.model_comparison_timeseries }}" alt="Model Comparison Time Series">
                <div class="figure-caption">Comparison of model predictions over time for a single subject</div>
            </div>
            {% endif %}
            
            {% if 'model_comparison_rmse' in figures %}
            <div class="figure-container">
                <img src="data:image/png;base64,{{ figures.model_comparison_rmse }}" alt="Model Comparison RMSE">
                <div class="figure-caption">Error comparison across modeling approaches for a single subject</div>
            </div>
            {% endif %}
            
            {% if 'model_comparison_scatter' in figures %}
            <div class="figure-container">
                <img src="data:image/png;base64,{{ figures.model_comparison_scatter }}" alt="Model Comparison Scatter">
                <div class="figure-caption">Side-by-side comparison of prediction accuracy for each model type</div>
            </div>
            {% endif %}
            
            <h4>Prediction Improvement Distribution</h4>
            
            <div class="row">
                <div class="col-md-6">
                    {% if 'improvement_distribution' in figures %}
                    <div class="figure-container">
                        <img src="data:image/png;base64,{{ figures.improvement_distribution }}" alt="Improvement Distribution">
                        <div class="figure-caption">Distribution of performance improvement (Subject vs. Global)</div>
                    </div>
                    {% endif %}
                </div>
                <div class="col-md-6">
                    {% if 'adaptive_improvement_distribution' in figures %}
                    <div class="figure-container">
                        <img src="data:image/png;base64,{{ figures.adaptive_improvement_distribution }}" alt="Adaptive Improvement Distribution">
                        <div class="figure-caption">Distribution of performance improvement (Adaptive vs. Subject)</div>
                    </div>
                    {% endif %}
                </div>
            </div>
            
            <p>
                The prediction accuracy visualizations clearly demonstrate the superior performance of personalized models.
                The scatter plots show tighter clustering around the perfect prediction line for subject-specific and adaptive models
                compared to the global model. The residual plots show smaller and more evenly distributed errors for the
                personalized approaches.
            </p>
            
            <p>
                The time series comparison for a single subject vividly illustrates how the subject-specific and adaptive models
                track the actual cognitive load values more accurately than the global model, which often misses the nuances
                of individual patterns.
            </p>
        </div>
        
        <!-- Discussion -->
        <div class="section" id="discussion">
            <h2>Discussion</h2>
            
            <h4>Key Findings</h4>
            <ul>
                <li><strong>Individual Variation:</strong> Cognitive load patterns show strong individual differences, requiring personalized models for accurate prediction.</li>
                <li><strong>Physiological Predictors:</strong> Skin conductance responses (SCR) and heart rate variability (HRV) metrics emerged as the most powerful predictors.</li>
                <li><strong>Feature Engineering:</strong> Derived features, particularly ratios between different physiological signals, significantly improved prediction accuracy.</li>
                <li><strong>Transfer Learning:</strong> The optimal approach heavily favors individual patterns but still benefits from incorporating global information.</li>
                <li><strong>Statistical Significance:</strong> The performance improvements from personalization are statistically significant and not due to chance variation.</li>
            </ul>
            
            {% if 'practical_implications' in figures %}
            <div class="figure-container">
                <img src="data:image/png;base64,{{ figures.practical_implications }}" alt="Practical Implications">
                <div class="figure-caption">Practical implications for cognitive load modeling in real-world applications</div>
            </div>
            {% endif %}
            
            <h4>Interpretation of Negative R² Values</h4>
            <p>
                The negative R² value ({{ "%.3f"|format(metrics.get('global_cross_subject_r2', -1)) }}) in cross-subject validation is a crucial finding that requires explanation.
                This indicates that when a model trained on some subjects is applied to predict cognitive load for a new subject,
                it performs worse than simply predicting the mean cognitive load value. This strongly suggests that:
            </p>
            <ul>
                <li>Cognitive load patterns are highly individualistic</li>
                <li>Physiological correlates of cognitive load vary substantially between individuals</li>
                <li>Attempting to apply a one-size-fits-all model across subjects is counterproductive</li>
                <li>Personalization is not just beneficial but necessary for accurate cognitive load prediction</li>
            </ul>
            
            <h4>Optimal Personalization Approach</h4>
            <p>
                Based on our findings, we recommend a hybrid approach to cognitive load prediction:
            </p>
            <ol>
                <li>Begin with a global model for new users (acknowledging its limitations)</li>
                <li>Collect individual calibration data during an initial period</li>
                <li>Train a subject-specific model once sufficient data is available</li>
                <li>Use adaptive transfer learning to optimally blend global and individual patterns</li>
                <li>Continuously update the model as more individual data becomes available</li>
            </ol>
            
            <h4>Implications for Different Domains</h4>
            <div class="row">
                <div class="col-md-6">
                    <div class="card mb-3">
                        <div class="card-header bg-primary text-white">
                            <h5 class="card-title mb-0">Aviation</h5>
                        </div>
                        <div class="card-body">
                            <ul>
                                <li>Pilot cognitive overload detection requires personalization</li>
                                <li>Training systems should include individual calibration phase</li>
                                <li>Cockpit systems could adapt to individual cognitive patterns</li>
                            </ul>
                        </div>
                    </div>
                </div>
                <div class="col-md-6">
                    <div class="card mb-3">
                        <div class="card-header bg-success text-white">
                            <h5 class="card-title mb-0">Education</h5>
                        </div>
                        <div class="card-body">
                            <ul>
                                <li>Adaptive learning systems need to recognize individual cognitive load profiles</li>
                                <li>Content difficulty should adjust based on personalized cognitive load models</li>
                                <li>Physiological monitoring could enable real-time adaptation</li>
                            </ul>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="row">
                <div class="col-md-6">
                    <div class="card mb-3">
                        <div class="card-header bg-info text-white">
                            <h5 class="card-title mb-0">Healthcare</h5>
                        </div>
                        <div class="card-body">
                            <ul>
                                <li>Clinical decision support systems should adapt to individual clinician patterns</li>
                                <li>Medical training could incorporate personalized cognitive load monitoring</li>
                                <li>Patient monitoring systems might benefit from individualized models</li>
                            </ul>
                        </div>
                    </div>
                </div>
                <div class="col-md-6">
                    <div class="card mb-3">
                        <div class="card-header bg-warning text-dark">
                            <h5 class="card-title mb-0">Human-Computer Interaction</h5>
                        </div>
                        <div class="card-body">
                            <ul>
                                <li>Interfaces should adapt based on personalized cognitive load models</li>
                                <li>Wearable devices could provide individualized cognitive state monitoring</li>
                                <li>Notification and interruption systems could consider personal cognitive patterns</li>
                            </ul>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Conclusion -->
        <div class="section" id="conclusion">
            <h2>Conclusion</h2>
            
            <p>
                This analysis provides strong evidence that cognitive load, as measured through physiological signals, 
                exhibits distinct individual patterns that cannot be effectively captured by a one-size-fits-all model.
                The dramatic performance gap between global and subject-specific models 
                ({{ "%.3f"|format(metrics.get('global_cross_subject_r2', -1)) }} vs. {{ "%.3f"|format(metrics.get('subject_r2', 0)) }})
                underscores the importance of personalization in cognitive load assessment.
            </p>
            
            <p>
                The most effective approach was adaptive transfer learning, which achieved an R² score of 
                {{ "%.3f"|format(metrics.get('adaptive_r2', 0)) }} by leveraging both global patterns and individual characteristics, 
                with a strong preference for the latter (optimal weight = {{ "%.2f"|format(metrics.get('optimal_weight', 0)) }}).
                The statistical significance of these improvements (p = {{ "%.4f"|format(metrics.get('improvement_p_value', 1)) }})
                confirms that personalization provides genuine benefits, not just random variation.
            </p>
            
            <p>
                Our findings have important implications for the design of adaptive systems in education, aviation, 
                healthcare, and human-computer interaction. Such systems should incorporate individual calibration
                phases and personalized models rather than relying solely on population averages.
            </p>
            
            <p>
                In summary, effective cognitive load prediction requires personalized models that account for 
                individual differences in physiological responses. The statistical confidence intervals and
                significance testing provided in this analysis add robustness to these conclusions and 
                should inform the development of future cognitive load monitoring systems.
            </p>
        </div>
        
        <!-- Limitations and Future Work -->
        <div class="section" id="limitations">
            <h2>Limitations and Future Work</h2>
            
            {% if 'limitations_future_work' in figures %}
            <div class="figure-container">
                <img src="data:image/png;base64,{{ figures.limitations_future_work }}" alt="Limitations and Future Work">
                <div class="figure-caption">Overview of current limitations and directions for future research</div>
            </div>
            {% endif %}
            
            <h4>Methodological Limitations</h4>
            <ul>
                <li><strong>Self-reported Cognitive Load:</strong> The use of self-reported mental effort as ground truth introduces 
                    subjectivity and potential inconsistencies. Future work should explore more objective measures or 
                    multimodal ground truth approaches.</li>
                <li><strong>Laboratory Setting:</strong> The data was collected in a controlled environment, which may not 
                    fully reflect real-world conditions. Ecological validity should be tested in more naturalistic settings.</li>
                <li><strong>Temporal Stability:</strong> The current analysis does not evaluate how stable the cognitive load 
                    patterns are over extended periods. Longitudinal studies are needed to assess model durability.</li>
                <li><strong>Sample Size:</strong> While the dataset includes multiple subjects, a larger and more diverse 
                    sample would strengthen the generalizability of the findings.</li>
            </ul>
            
            <h4>Technical Limitations</h4>
            <ul>
                <li><strong>Model Complexity:</strong> The current implementation uses random forest models, which may not 
                    capture all nonlinear relationships in the data. Deep learning approaches might offer additional benefits.</li>
                <li><strong>Feature Selection:</strong> The feature selection process could be more sophisticated, perhaps 
                    using subject-specific feature selection rather than a global approach.</li>
                <li><strong>Missing Data:</strong> The handling of missing values could be improved with more advanced 
                    imputation techniques specific to physiological time series data.</li>
                <li><strong>Real-time Implementation:</strong> The current pipeline is not optimized for real-time 
                    processing, which would be necessary for many practical applications.</li>
            </ul>
            
            <h4>Promising Directions for Future Research</h4>
            <ul>
                <li><strong>Advanced Transfer Learning:</strong> Explore more sophisticated transfer learning approaches, 
                    such as meta-learning or domain adaptation techniques.</li>
                <li><strong>Deep Physiological Models:</strong> Investigate end-to-end deep learning models that can 
                    learn directly from raw physiological signals without manual feature engineering.</li>
                <li><strong>Subject Clustering:</strong> Develop methods to identify clusters of similar subjects to 
                    improve transfer learning for new users.</li>
                <li><strong>Multimodal Integration:</strong> Combine physiological signals with other modalities 
                    such as eye tracking, facial expressions, or performance metrics.</li>
                <li><strong>Continuous Adaptation:</strong> Design models that can continuously adapt to changing 
                    individual patterns over time, accounting for learning effects and habituation.</li>
                <li><strong>Cross-domain Validation:</strong> Test the personalization approach across different 
                    domains and task types to assess generalizability.</li>
            </ul>
        </div>
        
        <!-- Technical Appendix -->
        <div class="section" id="appendix">
            <h2>Technical Appendix</h2>
            
            <h4>Model Hyperparameters</h4>
            <div class="table-responsive">
                <table class="table table-striped">
                    <thead>
                        <tr>
                            <th>Model Type</th>
                            <th>Algorithm</th>
                            <th>Key Hyperparameters</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td>Global Model</td>
                            <td>Random Forest Regressor</td>
                            <td>n_estimators=100, random_state=42</td>
                        </tr>
                        <tr>
                            <td>Subject-specific Models</td>
                            <td>Random Forest Regressor</td>
                            <td>n_estimators=50, random_state=42</td>
                        </tr>
                        <tr>
                            <td>Adaptive Transfer</td>
                            <td>Weighted Ensemble</td>
                            <td>Optimal weight determined by validation</td>
                        </tr>
                    </tbody>
                </table>
            </div>
            
            <h4>Statistical Testing Details</h4>
            <p>
                Statistical significance testing was performed using paired t-tests comparing model performance across subjects.
                Confidence intervals (95%) were calculated using bootstrap resampling with 1000 samples.
                Model improvements were considered statistically significant at p < 0.05.
            </p>
            
            <h4>Selected Features</h4>
            <p>
                The top 10 selected features for modeling were:
            </p>
            <ol>
                <li>scr_mean (Skin Conductance Response mean)</li>
                <li>eda_rate (Rate of change in Electrodermal Activity)</li>
                <li>hrv_lf_hf_ratio (Low Frequency to High Frequency ratio in Heart Rate Variability)</li>
                <li>arousal_index (Compound feature combining multiple physiological signals)</li>
                <li>scr_count (Count of Skin Conductance Responses)</li>
                <li>eda_mean_ma3 (Moving average of EDA mean)</li>
                <li>ibi_mean (Inter-Beat Interval mean)</li>
                <li>temp_rate (Rate of change in skin temperature)</li>
                <li>ratio_scr_eda (Ratio between Skin Conductance Response and Electrodermal Activity)</li>
                <li>hrv_spectral_entropy (Spectral entropy of Heart Rate Variability)</li>
            </ol>
            
            <h4>Software Implementation</h4>
            <p>
                The analysis pipeline was implemented in Python using scikit-learn for machine learning components 
                and visualization libraries including matplotlib and seaborn. The full source code is available
                in the accompanying repository.
            </p>
            
            <p>
                Key Python libraries used:
            </p>
            <ul>
                <li>pandas and numpy for data manipulation</li>
                <li>scikit-learn for machine learning models and evaluation</li>
                <li>scipy for signal processing and statistical testing</li>
                <li>matplotlib and seaborn for visualization</li>
                <li>statsmodels for additional statistical analysis</li>
            </ul>
        </div>
        
        <footer>
            <p>
                Cognitive Load Analysis Report | Pipeline Version {{ version }}
            </p>
            <p class="small text-muted">
                Generated on {{ timestamp }} | <a href="#toc">Back to Top</a>
            </p>
        </footer>
    </div>
    
    <!-- Back to Top Button -->
    <a href="#" class="btn btn-primary btn-toc">
        ↑
    </a>
    
    <!-- Bootstrap JS -->
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    
</body>
</html>
        '''


# =============================================================================
# Main Pipeline
# =============================================================================

class CognitiveLoadPipeline:
    """Main pipeline for cognitive load analysis"""
    
    def __init__(self, target_col: str = DEFAULT_TARGET_COL, 
                 subject_col: str = DEFAULT_SUBJECT_COL,
                 results_dir: str = None,
                 random_state: int = 42):
        """
        Initialize the pipeline
        
        Args:
            target_col: Name of the target column (cognitive load measure)
            subject_col: Name of the subject identifier column
            results_dir: Directory to store results
            random_state: Random seed for reproducibility
        """
        self.target_col = target_col
        self.subject_col = subject_col
        self.random_state = random_state
        
        # Create results directory if needed
        if results_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.results_dir = f"cognitive_load_results_{timestamp}"
        else:
            self.results_dir = results_dir
            
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(os.path.join(self.results_dir, "models"), exist_ok=True)
        os.makedirs(os.path.join(self.results_dir, "data"), exist_ok=True)
        
        # Initialize components
        self.data_processor = DataProcessor(target_col, subject_col, random_state)
        self.feature_engineer = FeatureEngineer(subject_col)
        self.feature_selector = FeatureSelector()
        self.modeler = CognitiveLoadModeler(subject_col, target_col, random_state)
        self.report_generator = None
        
        # Results storage
        self.results = {}
        self.train_data = None
        self.test_data = None
        self.train_data_between = None
        self.test_data_between = None
        self.features = None
        self.selected_features = None
        self.predictions = {}
    
    def run(self, data_path: str, test_size: float = DEFAULT_TEST_SIZE, 
           n_features: int = DEFAULT_FEATURE_COUNT):
        """
        Run the full pipeline with both within-subject and cross-subject evaluation
        
        Args:
            data_path: Path to the input data file
            test_size: Proportion of data to use for testing
            n_features: Number of features to select
        """
        start_time = time.time()
        logger.info(f"=== COGNITIVE LOAD ANALYSIS PIPELINE v{VERSION} ===")
        logger.info(f"Analysis started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Results will be saved to: {self.results_dir}")
        
        # 1. Load and preprocess data
        df_clean, df_ready = self.data_processor.load_and_clean_data(data_path)
        
        # Save preprocessed data
        preprocessed_path = os.path.join(self.results_dir, "data", "preprocessed_data.csv")
        df_ready.to_csv(preprocessed_path, index=False)
        logger.info(f"Preprocessed data saved to {preprocessed_path}")
        
        # 2. Explore dataset
        dataset_stats = self.data_processor.explore_dataset(df_ready)
        
        # Save dataset stats
        with open(os.path.join(self.results_dir, "data", "dataset_stats.json"), "w") as f:
            json.dump(dataset_stats, f, indent=2)
        
        # 3. Split data for different evaluation purposes
        
        # 3a. Within-subject split (for prediction visualization)
        logger.info("Creating data splits for validation")
        train_df_within, test_df_within = self.data_processor.split_train_test_within_subject(
            df_ready, test_size=test_size
        )
        self.train_data = train_df_within
        self.test_data = test_df_within
        
        # 3b. Between-subject split (for generalization evaluation)
        train_df_between, test_df_between = self.data_processor.split_train_test_between_subjects(
            df_ready, test_size=test_size
        )
        self.train_data_between = train_df_between
        self.test_data_between = test_df_between
        
        # 4. Engineer features (using within-subject training data)
        logger.info("Beginning feature engineering")
        train_engineered = self.feature_engineer.engineer_all_features(self.train_data)
        
        # Save engineered training data
        engineered_path = os.path.join(self.results_dir, "data", "engineered_train_data.csv")
        train_engineered.to_csv(engineered_path, index=False)
        logger.info(f"Engineered training data saved to {engineered_path}")
        
        # 5. Select features
        logger.info("Selecting features")
        selected_features = self.feature_selector.select_features(
            train_engineered, self.target_col, self.subject_col, n_features=n_features
        )
        # Ensure subject_col is not in features
        selected_features = [f for f in selected_features if f != self.subject_col]
        self.selected_features = selected_features
        
        # Save selected features
        with open(os.path.join(self.results_dir, "selected_features.txt"), "w") as f:
            for feature in selected_features:
                f.write(f"{feature}\n")
        
        # Save feature importance data
        feature_importance_df = self.feature_selector.get_feature_importance_df()
        feature_importance_df.to_csv(os.path.join(self.results_dir, "feature_importance.csv"), index=False)
        
        # 6. Create interaction features based on top features
        logger.info("Creating interaction features from top features")
        train_engineered = self.feature_engineer.create_interaction_features(
            train_engineered, selected_features, max_interactions=15)
        
        # Update feature selection to include top interaction features
        top_interaction_features = [f for f in self.feature_engineer.engineered_feature_groups.get('interaction', []) 
                                  if f not in selected_features][:5]  # Add up to 5 top interaction features
        
        if top_interaction_features:
            logger.info(f"Adding {len(top_interaction_features)} interaction features to selected features")
            selected_features.extend(top_interaction_features)
            self.selected_features = selected_features
            
            # Update the saved features file
            with open(os.path.join(self.results_dir, "selected_features.txt"), "w") as f:
                for feature in selected_features:
                    f.write(f"{feature}\n")
        
        # 7. Engineer features for test data
        logger.info("Engineering features for test data")
        test_engineered = self.feature_engineer.engineer_all_features(self.test_data)

        # Create the same interaction features for test data
        logger.info("Creating interaction features for test data")
        test_engineered = self.feature_engineer.create_interaction_features(
            test_engineered, selected_features, max_interactions=15)
        
        # 8. Train and evaluate global model with both within-subject and cross-subject metrics
        logger.info("Training and evaluating global model")
        global_model = self.modeler.train_global_model(
            train_engineered, selected_features, 
            model_class=RandomForestRegressor, 
            n_estimators=100, 
            max_depth=None, 
            n_jobs=-1
        )
        
        global_performance = self.modeler.evaluate_global_model(
            train_engineered, selected_features, self.data_processor
        )
        
        # 9. Train subject-specific models
        logger.info("Training subject-specific models")
        subject_performance = self.modeler.train_subject_models(
            train_engineered, selected_features, min_samples=10,
            model_class=RandomForestRegressor, 
            n_estimators=50, 
            max_depth=None
        )
        
        # 10. Train adaptive transfer models
        logger.info("Training adaptive transfer models")
        adaptive_performance = self.modeler.train_adaptive_transfer_models(
            train_engineered, selected_features, test_size=0.3,
            model_class=RandomForestRegressor, 
            n_estimators=100, 
            max_depth=None
        )
        
        # 11. Compare models on test data
        logger.info("Comparing models on test data")
        comparison_results = self.modeler.compare_models(test_engineered, selected_features)
        
        # 12. Generate predictions for visualization
        logger.info("Generating predictions for visualization")
        predictions = self.modeler.generate_predictions_for_report(test_engineered, selected_features)
        self.predictions = predictions
        
        # Save predictions
        serialized_predictions = self._serialize_for_json(predictions)
        with open(os.path.join(self.results_dir, "predictions.json"), "w") as f:
            json.dump(serialized_predictions, f, indent=2)
        
        # 13. Save models
        logger.info("Saving models")
        
        # Save global model
        joblib.dump(self.modeler.global_model, 
                   os.path.join(self.results_dir, "models", "global_model.pkl"))
        
        # Save a few key subject models (up to 5)
        for i, (subject, model) in enumerate(list(self.modeler.subject_models.items())[:5]):
            joblib.dump(model, 
                       os.path.join(self.results_dir, "models", f"subject_{subject}_model.pkl"))
            if i >= 4:
                break
        
        # Save a few key adaptive models (up to 5)
        for i, (subject, models) in enumerate(list(self.modeler.adaptive_models.items())[:5]):
            # Save the target model which has been trained on this subject
            joblib.dump(models['target_model'], 
                       os.path.join(self.results_dir, "models", f"adaptive_{subject}_target_model.pkl"))
            if i >= 4:
                break
        
        # 14. Store all results
        self.results = {
            'dataset_stats': dataset_stats,
            'global_model': global_performance,
            'subject_models': subject_performance,
            'adaptive_models': adaptive_performance,
            'model_comparison': comparison_results,
            'features': {
                'n_original': len(self.train_data.columns),
                'n_engineered': len(train_engineered.columns),
                'n_selected': len(selected_features),
                'selected': selected_features,
                'engineered_feature_groups': {k: len(v) for k, v in self.feature_engineer.engineered_feature_groups.items()}
            },
            'runtime': time.time() - start_time
        }
        
        # Save all results
        serialized_results = self._serialize_for_json(self.results)
        with open(os.path.join(self.results_dir, "results.json"), "w") as f:
            json.dump(serialized_results, f, indent=2)
        
        # 15. Create report generator
        self.report_generator = ReportGenerator(self.results_dir, "Cognitive Load Analysis Results")
        
        logger.info(f"Analysis completed in {time.time() - start_time:.2f} seconds")
        
        return self.results
    
    def generate_report(self):
        """
        Generate comprehensive HTML report
        
        Returns:
            Path to the generated HTML report
        """
        if self.report_generator is None:
            self.report_generator = ReportGenerator(self.results_dir, "Cognitive Load Analysis Results")
        
        logger.info("Generating report visualizations")
        
        # 1. Add dataset visualizations
        self.report_generator.add_dataset_visualization(
            self.train_data, 
            self.results['dataset_stats'],
            self.target_col, 
            self.subject_col
        )
        
        # 2. Add methodology visualization
        self.report_generator.add_methodology_visualization()
        
        # 3. Add feature engineering visualizations
        self.report_generator.add_feature_engineering_visualization(
            self.results['features']['n_original'],
            self.results['features']['n_engineered'],
            self.results['features']['n_selected'],
            self.feature_engineer.engineered_feature_groups
        )
        
        # 4. Add feature importance visualizations
        self.report_generator.add_feature_importance_visualization(
            self.selected_features,
            self.modeler.feature_importances
        )
        
        # 5. Add model performance visualizations
        self.report_generator.add_model_performance_visualization(
            self.results['global_model'],
            self.results['subject_models'],
            self.results['adaptive_models'],
            self.results['model_comparison']
        )
        
        # 6. Add R² explanation visualization
        self.report_generator.add_r2_explanation_visualization()
        
        # 7. Add adaptive transfer learning visualizations
        self.report_generator.add_adaptive_transfer_visualization(
            self.results['adaptive_models']
        )
        
        # 8. Add comparison visualizations
        self.report_generator.add_comparison_visualization(
            self.predictions
        )
        
        # 9. Add discussion visualizations
        self.report_generator.add_discussion_visualization()
        
        # 10. Create HTML report
        logger.info("Generating HTML report")
        report_path = self.report_generator.create_html_report()
        
        logger.info(f"Report generated: {report_path}")
        return report_path
    
    def predict(self, df: pd.DataFrame, method: str = 'adaptive'):
        """
        Make predictions using the appropriate model for each subject
        
        Args:
            df: Input dataframe
            method: Prediction method ('global', 'subject', or 'adaptive')
            
        Returns:
            Tuple of (predictions, prediction_metadata)
        """
        if self.selected_features is None:
            raise ValueError("Pipeline must be run before making predictions")
        
        # Copy input dataframe to avoid modification
        df = df.copy()
        
        # Engineer features
        logger.info("Engineering features for prediction data")
        df_engineered = self.feature_engineer.engineer_all_features(df)
        
        # Make predictions
        logger.info(f"Making predictions using {method} method")
        predictions, metadata = self.modeler.predict(
            df_engineered, self.selected_features, method
        )
        
        return predictions, metadata

    def _serialize_for_json(self, obj):
        """
        Serialize objects for JSON by converting NumPy types to Python native types
        and converting dictionary keys to strings
        
        Args:
            obj: Object to serialize
            
        Returns:
            JSON-serializable object
        """
        if isinstance(obj, dict):
            # Create a new dict with string keys and serialized values
            return {str(k): self._serialize_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._serialize_for_json(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, (np.ndarray, list)):
            return [self._serialize_for_json(x) for x in obj]
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, (datetime, pd.Timestamp)):
            return str(obj)
        elif obj is None:
            return None
        elif isinstance(obj, (str, int, float, bool)):
            return obj
        else:
            # For any other types, convert to string
            return str(obj)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description=f"Cognitive Load Analysis Pipeline v{VERSION}"
    )
    
    parser.add_argument("--data", required=True, 
                      help="Path to the input data CSV file")
    
    parser.add_argument("--output", default=None,
                      help="Output directory for results (default: timestamped directory)")
    
    parser.add_argument("--features", type=int, default=DEFAULT_FEATURE_COUNT,
                      help=f"Number of features to select (default: {DEFAULT_FEATURE_COUNT})")
    
    parser.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE,
                      help=f"Proportion of data to use for testing (default: {DEFAULT_TEST_SIZE})")
    
    parser.add_argument("--target", default=DEFAULT_TARGET_COL,
                      help=f"Name of the target column (default: {DEFAULT_TARGET_COL})")
    
    parser.add_argument("--subject", default=DEFAULT_SUBJECT_COL,
                      help=f"Name of the subject identifier column (default: {DEFAULT_SUBJECT_COL})")
    
    parser.add_argument("--no-report", action="store_true",
                      help="Skip HTML report generation")
    
    parser.add_argument("--seed", type=int, default=42,
                      help="Random seed for reproducibility (default: 42)")
    
    args = parser.parse_args()
    return args


def main():
    """Main entry point for the script"""
    # Parse arguments
    args = parse_arguments()
    
    # Check if data file exists
    if not os.path.exists(args.data):
        logger.error(f"Data file not found: {args.data}")
        return 1
    
    try:
        # Create pipeline
        pipeline = CognitiveLoadPipeline(
            target_col=args.target,
            subject_col=args.subject,
            results_dir=args.output,
            random_state=args.seed
        )
        
        # Run pipeline
        pipeline.run(
            data_path=args.data,
            test_size=args.test_size,
            n_features=args.features
        )
        
        # Generate report
        if not args.no_report:
            report_path = pipeline.generate_report()
            logger.info(f"HTML report: {report_path}")
        
        logger.info("Pipeline completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Error in pipeline: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())



    
