import os
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
from sklearn.model_selection import train_test_split, GroupKFold, StratifiedGroupKFold
import matplotlib.pyplot as plt
import seaborn as sns


class DataSplitter:
    """
    Handles data splitting for various validation approaches,
    with special attention to maintaining subject independence.
    """
    
    def __init__(self, id_column: str = 'pilot_id',
                 random_seed: int = 42,
                 output_dir: str = None):
        """
        Initialize the DataSplitter object.
        
        Args:
            id_column: Column name for subject identifier
            random_seed: Random seed for reproducibility
            output_dir: Directory to save validation files
        """
        self.id_column = id_column
        self.random_seed = random_seed
        self.output_dir = output_dir
        
        # Split containers
        self.train_indices = None
        self.test_indices = None
        self.validation_indices = None
        
        # Split metadata
        self.split_metadata = {}
        
        # Logger
        self.logger = logging.getLogger(__name__)
        
        # Set random seed
        np.random.seed(self.random_seed)
    
    def train_test_split(self, data: pd.DataFrame, test_size: float = 0.2,
                        stratify_by: str = None, validation_size: float = 0.0) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into train and test sets, ensuring subject independence.
        
        Args:
            data: Input DataFrame
            test_size: Proportion of data to use for testing
            stratify_by: Column to use for stratified splitting
            validation_size: Proportion of training data to use for validation
            
        Returns:
            Tuple of (train_data, test_data) or 
            (train_data, test_data, validation_data) if validation_size > 0
        """
        self.logger.info(f"Splitting data with test_size={test_size}")
        
        # Check if ID column exists
        if self.id_column not in data.columns:
            self.logger.error(f"ID column '{self.id_column}' not found in data")
            raise ValueError(f"ID column '{self.id_column}' not found in data")
        
        # Get unique subject IDs
        subject_ids = data[self.id_column].unique()
        n_subjects = len(subject_ids)
        
        self.logger.info(f"Found {n_subjects} unique subjects")
        
        # Determine if we can perform stratified split
        can_stratify = stratify_by is not None and stratify_by in data.columns
        
        if can_stratify:
            # Get stratification values for each subject
            # We need one value per subject, so we'll take the most common value
            subject_strata = {}
            for subject in subject_ids:
                subject_data = data[data[self.id_column] == subject]
                most_common = subject_data[stratify_by].value_counts().idxmax()
                subject_strata[subject] = most_common
            
            strata = [subject_strata[subject] for subject in subject_ids]
            self.logger.info(f"Performing stratified split by '{stratify_by}'")
        else:
            strata = None
            if stratify_by is not None:
                self.logger.warning(f"Column '{stratify_by}' not found for stratification, using random split")
        
        # Split subjects into train and test
        train_subjects, test_subjects = train_test_split(
            subject_ids, 
            test_size=test_size,
            random_state=self.random_seed,
            stratify=strata
        )
        
        # Create train and test masks
        train_mask = data[self.id_column].isin(train_subjects)
        test_mask = data[self.id_column].isin(test_subjects)
        
        # Get train and test data
        train_data = data[train_mask].copy().reset_index(drop=True)
        test_data = data[test_mask].copy().reset_index(drop=True)
        
        # Store indices
        self.train_indices = train_mask
        self.test_indices = test_mask
        
        # Log split statistics
        self.logger.info(f"Train set: {len(train_data)} samples, {len(train_subjects)} subjects")
        self.logger.info(f"Test set: {len(test_data)} samples, {len(test_subjects)} subjects")
        
        # Store split metadata
        self.split_metadata = {
            'n_subjects': n_subjects,
            'train_subjects': len(train_subjects),
            'test_subjects': len(test_subjects),
            'train_samples': len(train_data),
            'test_samples': len(test_data),
            'stratified': can_stratify,
            'stratify_by': stratify_by if can_stratify else None
        }
        
        # Verify that subjects are not split between train and test
        train_test_intersection = set(train_data[self.id_column]).intersection(set(test_data[self.id_column]))
        if train_test_intersection:
            self.logger.error(f"Subject independence violation: {len(train_test_intersection)} subjects appear in both train and test sets")
            raise ValueError("Subject independence violation in train/test split")
        
        # Check if validation set is requested
        if validation_size > 0:
            # Split validation set from train set
            train_data, validation_data = self._split_validation(train_data, validation_size)
            
            # Update metadata
            self.split_metadata['validation_samples'] = len(validation_data)
            self.split_metadata['validation_subjects'] = len(validation_data[self.id_column].unique())
            
            # Log validation statistics
            self.logger.info(f"Validation set: {len(validation_data)} samples, {len(validation_data[self.id_column].unique())} subjects")
            
            return train_data, test_data, validation_data
        
        # Analyze the split balance
        self._analyze_split_balance(train_data, test_data)
        
        # Save split visualization if output directory is provided
        if self.output_dir:
            self._visualize_split(train_data, test_data)
        
        return train_data, test_data
    
    def _split_validation(self, train_data: pd.DataFrame, validation_size: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split validation set from training data, maintaining subject independence.
        
        Args:
            train_data: Training data
            validation_size: Proportion of training data to use for validation
            
        Returns:
            Tuple of (train_data, validation_data)
        """
        # Get unique subject IDs from training data
        train_subjects = train_data[self.id_column].unique()
        
        # Split subjects into train and validation
        final_train_subjects, validation_subjects = train_test_split(
            train_subjects, 
            test_size=validation_size,
            random_state=self.random_seed
        )
        
        # Create train and validation masks
        final_train_mask = train_data[self.id_column].isin(final_train_subjects)
        validation_mask = train_data[self.id_column].isin(validation_subjects)
        
        # Get final train and validation data
        final_train_data = train_data[final_train_mask].copy().reset_index(drop=True)
        validation_data = train_data[validation_mask].copy().reset_index(drop=True)
        
        # Store validation indices
        self.validation_indices = validation_mask
        
        # Verify subject independence
        train_val_intersection = set(final_train_data[self.id_column]).intersection(set(validation_data[self.id_column]))
        if train_val_intersection:
            self.logger.error(f"Subject independence violation: {len(train_val_intersection)} subjects appear in both train and validation sets")
            raise ValueError("Subject independence violation in train/validation split")
        
        return final_train_data, validation_data
    
    def create_cross_validation_folds(self, data: pd.DataFrame, n_splits: int = 5, 
                                     stratify: bool = False) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Create cross-validation folds with subject independence.
        
        Args:
            data: Input DataFrame
            n_splits: Number of CV folds
            stratify: Whether to use stratified folds
            
        Returns:
            List of (train_indices, test_indices) tuples
        """
        self.logger.info(f"Creating {n_splits}-fold cross-validation split")
        
        # Check if ID column exists
        if self.id_column not in data.columns:
            self.logger.error(f"ID column '{self.id_column}' not found in data")
            raise ValueError(f"ID column '{self.id_column}' not found in data")
        
        # Get groups for GroupKFold
        groups = data[self.id_column].values
        
        # Choose appropriate CV splitter
        if stratify and 'turbulence' in data.columns:
            self.logger.info("Using StratifiedGroupKFold with turbulence as stratification")
            cv_splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=self.random_seed)
            y = data['turbulence'].values
        else:
            self.logger.info("Using GroupKFold for subject independence")
            cv_splitter = GroupKFold(n_splits=n_splits)
            y = None
        
        # Create folds
        fold_indices = []
        X = np.arange(len(data))
        
        for train_idx, test_idx in cv_splitter.split(X, y, groups):
            fold_indices.append((train_idx, test_idx))
        
        # Store metadata about folds
        fold_stats = []
        for i, (train_idx, test_idx) in enumerate(fold_indices):
            # Count unique subjects in train and test
            train_subjects = len(set(data.iloc[train_idx][self.id_column]))
            test_subjects = len(set(data.iloc[test_idx][self.id_column]))
            
            fold_stats.append({
                'fold': i + 1,
                'train_samples': len(train_idx),
                'test_samples': len(test_idx),
                'train_subjects': train_subjects,
                'test_subjects': test_subjects
            })
            
            self.logger.info(f"Fold {i+1}: {len(train_idx)} train samples, {len(test_idx)} test samples, "
                         f"{train_subjects} train subjects, {test_subjects} test subjects")
        
        # Update metadata
        self.split_metadata['cv_folds'] = n_splits
        self.split_metadata['cv_stats'] = fold_stats
        self.split_metadata['cv_stratified'] = stratify
        
        return fold_indices
    
    def create_leave_one_subject_out_folds(self, data: pd.DataFrame) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Create leave-one-subject-out cross-validation folds.
        
        Args:
            data: Input DataFrame
            
        Returns:
            List of (train_indices, test_indices) tuples
        """
        self.logger.info("Creating leave-one-subject-out cross-validation split")
        
        # Check if ID column exists
        if self.id_column not in data.columns:
            self.logger.error(f"ID column '{self.id_column}' not found in data")
            raise ValueError(f"ID column '{self.id_column}' not found in data")
        
        # Get unique subject IDs
        subject_ids = data[self.id_column].unique()
        n_subjects = len(subject_ids)
        
        self.logger.info(f"Found {n_subjects} unique subjects")
        
        # Create folds - one fold per subject
        fold_indices = []
        
        for subject in subject_ids:
            # Test indices are for the current subject
            test_mask = data[self.id_column] == subject
            train_mask = ~test_mask
            
            test_indices = np.where(test_mask)[0]
            train_indices = np.where(train_mask)[0]
            
            fold_indices.append((train_indices, test_indices))
        
        # Store metadata
        self.split_metadata['loso_folds'] = n_subjects
        
        fold_stats = []
        for i, (train_idx, test_idx) in enumerate(fold_indices):
            fold_stats.append({
                'fold': i + 1,
                'subject': subject_ids[i],
                'train_samples': len(train_idx),
                'test_samples': len(test_idx)
            })
        
        self.split_metadata['loso_stats'] = fold_stats
        
        return fold_indices
    
    def create_learning_curve_splits(self, data: pd.DataFrame, 
                                    train_sizes: List[float] = None) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Create splits for learning curve analysis.
        
        Args:
            data: Input DataFrame
            train_sizes: List of training set proportions
            
        Returns:
            List of (train_data, test_data) tuples
        """
        self.logger.info("Creating splits for learning curve analysis")
        
        # Default train sizes if not provided
        if train_sizes is None:
            train_sizes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        
        # First create a fixed test set (20% of data)
        train_data, test_data = self.train_test_split(data, test_size=0.2)
        
        # Get unique subject IDs from training data
        train_subjects = train_data[self.id_column].unique()
        n_train_subjects = len(train_subjects)
        
        # Shuffle subjects
        np.random.shuffle(train_subjects)
        
        learning_curve_splits = []
        
        for size in train_sizes:
            # Calculate number of subjects to include
            n_subjects = max(1, int(size * n_train_subjects))
            
            # Select subjects for this split
            selected_subjects = train_subjects[:n_subjects]
            
            # Create subset of training data
            subset_mask = train_data[self.id_column].isin(selected_subjects)
            subset_data = train_data[subset_mask].copy().reset_index(drop=True)
            
            # Store split
            learning_curve_splits.append((subset_data, test_data))
            
            self.logger.info(f"Train size {size:.1f}: {len(subset_data)} samples, {n_subjects} subjects")
        
        # Store metadata
        self.split_metadata['learning_curve_sizes'] = train_sizes
        self.split_metadata['learning_curve_test_size'] = len(test_data)
        self.split_metadata['learning_curve_test_subjects'] = len(test_data[self.id_column].unique())
        
        return learning_curve_splits
    
    def _analyze_split_balance(self, train_data: pd.DataFrame, test_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze the balance of the train/test split.
        
        Args:
            train_data: Training data
            test_data: Test data
            
        Returns:
            Dictionary with balance statistics
        """
        balance_stats = {}
        
        # Check turbulence level balance if available
        if 'turbulence' in train_data.columns and 'turbulence' in test_data.columns:
            train_turb = train_data['turbulence'].value_counts(normalize=True)
            test_turb = test_data['turbulence'].value_counts(normalize=True)
            
            # Calculate distribution difference
            turb_levels = sorted(set(train_turb.index) | set(test_turb.index))
            train_dist = [train_turb.get(level, 0) for level in turb_levels]
            test_dist = [test_turb.get(level, 0) for level in turb_levels]
            
            # Calculate Jensen-Shannon divergence
            from scipy.spatial.distance import jensenshannon
            js_divergence = jensenshannon(train_dist, test_dist)
            
            balance_stats['turbulence_js_divergence'] = js_divergence
            balance_stats['turbulence_train'] = train_turb.to_dict()
            balance_stats['turbulence_test'] = test_turb.to_dict()
            
            # Log turbulence balance
            self.logger.info(f"Turbulence distribution JS divergence: {js_divergence:.4f}")
            if js_divergence > 0.1:
                self.logger.warning("Train and test sets have different turbulence distributions")
                
            # Update metadata
            self.split_metadata.update(balance_stats)
        
        # Check pilot category balance if available
        if 'pilot_category' in train_data.columns and 'pilot_category' in test_data.columns:
            train_cat = train_data['pilot_category'].value_counts(normalize=True)
            test_cat = test_data['pilot_category'].value_counts(normalize=True)
            
            # Calculate distribution difference
            categories = sorted(set(train_cat.index) | set(test_cat.index))
            train_dist = [train_cat.get(cat, 0) for cat in categories]
            test_dist = [test_cat.get(cat, 0) for cat in categories]
            
            # Calculate Jensen-Shannon divergence
            from scipy.spatial.distance import jensenshannon
            js_divergence = jensenshannon(train_dist, test_dist)
            
            balance_stats['pilot_category_js_divergence'] = js_divergence
            balance_stats['pilot_category_train'] = train_cat.to_dict()
            balance_stats['pilot_category_test'] = test_cat.to_dict()
            
            # Log category balance
            self.logger.info(f"Pilot category distribution JS divergence: {js_divergence:.4f}")
            if js_divergence > 0.1:
                self.logger.warning("Train and test sets have different pilot category distributions")
                
            # Update metadata
            self.split_metadata.update(balance_stats)
            
        return balance_stats
    
    def _visualize_split(self, train_data: pd.DataFrame, test_data: pd.DataFrame) -> str:
        """
        Create visualizations of the train/test split.
        
        Args:
            train_data: Training data
            test_data: Test data
            
        Returns:
            Path to the saved visualization
        """
        if not self.output_dir:
            return None
            
        # Create figure with subplots
        fig, axs = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot 1: Turbulence distribution
        if 'turbulence' in train_data.columns and 'turbulence' in test_data.columns:
            train_turb = train_data['turbulence'].value_counts().sort_index()
            test_turb = test_data['turbulence'].value_counts().sort_index()
            
            # Combine for plotting
            combined = pd.DataFrame({
                'Train': train_turb,
                'Test': test_turb
            })
            
            # Plot as percentage
            combined_pct = combined.div(combined.sum(axis=0), axis=1) * 100
            combined_pct.plot(kind='bar', ax=axs[0])
            
            axs[0].set_title('Turbulence Level Distribution')
            axs[0].set_ylabel('Percentage (%)')
            axs[0].set_xlabel('Turbulence Level')
        
        # Plot 2: Pilot category distribution
        if 'pilot_category' in train_data.columns and 'pilot_category' in test_data.columns:
            train_cat = train_data['pilot_category'].value_counts()
            test_cat = test_data['pilot_category'].value_counts()
            
            # Combine for plotting
            combined = pd.DataFrame({
                'Train': train_cat,
                'Test': test_cat
            })
            
            # Plot as percentage
            combined_pct = combined.div(combined.sum(axis=0), axis=1) * 100
            combined_pct.plot(kind='bar', ax=axs[1])
            
            axs[1].set_title('Pilot Category Distribution')
            axs[1].set_ylabel('Percentage (%)')
            axs[1].set_xlabel('Pilot Category')
        
        plt.tight_layout()
        
        # Save figure
        save_path = os.path.join(self.output_dir, 'train_test_split_distribution.png')
        plt.savefig(save_path, dpi=300)
        plt.close()
        
        # Additionally, create a visualization of target distribution
        if 'avg_tlx_quantile' in train_data.columns and 'avg_tlx_quantile' in test_data.columns:
            plt.figure(figsize=(10, 6))
            
            # Create KDE plots
            sns.kdeplot(train_data['avg_tlx_quantile'], label='Train', alpha=0.7)
            sns.kdeplot(test_data['avg_tlx_quantile'], label='Test', alpha=0.7)
            
            plt.title('Target Distribution in Train and Test Sets')
            plt.xlabel('Target Value (avg_tlx_quantile)')
            plt.ylabel('Density')
            plt.legend()
            
            # Save figure
            target_path = os.path.join(self.output_dir, 'target_distribution_split.png')
            plt.savefig(target_path, dpi=300)
            plt.close()
        
        return save_path
    
    def save_split_metadata(self) -> str:
        """
        Save split metadata to a file.
        
        Returns:
            Path to the saved file
        """
        if not self.output_dir:
            self.logger.warning("No output directory specified, skipping saving split metadata")
            return None
            
        import json
        
        # Create a JSON-serializable version of the metadata
        serializable_metadata = {}
        for key, value in self.split_metadata.items():
            if isinstance(value, np.ndarray):
                serializable_metadata[key] = value.tolist()
            elif isinstance(value, pd.Series):
                serializable_metadata[key] = value.to_dict()
            else:
                serializable_metadata[key] = value
        
        # Save to file
        metadata_path = os.path.join(self.output_dir, 'split_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(serializable_metadata, f, indent=2)
            
        self.logger.info(f"Split metadata saved to: {metadata_path}")
        
        return metadata_path
