import os
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Tuple, Optional, Union
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import KNNImputer
import joblib


class DataLoader:
    """
    Handles data loading, preprocessing, and basic analysis for cognitive load datasets.
    Improved version with enhanced dataset statistics and diagnostic capabilities.
    """
    
    def __init__(self, 
                 id_column: str = 'pilot_id', 
                 target_column: str = 'avg_tlx_quantile',
                 excluded_columns: List[str] = None,
                 output_dir: str = None):
        """
        Initialize the DataLoader object.
        
        Args:
            id_column: Column name for subject identifier
            target_column: Column name for the target variable
            excluded_columns: List of columns to exclude from analysis
            output_dir: Directory to save output files
        """
        self.id_column = id_column
        self.target_column = target_column
        self.excluded_columns = excluded_columns or []
        self.output_dir = output_dir
        
        # Data containers
        self.data = None
        self.preprocessed_data = None
        
        # Metadata
        self.dataset_stats = {}
        self.pilot_info = {}
        self.data_quality_report = {}
        
        # Processing objects
        self.scaler = None
        self.imputer = None
        
        # Logger
        self.logger = logging.getLogger(__name__)
    
    def load_data(self, file_path: str, **kwargs) -> pd.DataFrame:
        """
        Load data from file and perform initial preprocessing.
        
        Args:
            file_path: Path to the data file (parquet, csv, etc.)
            **kwargs: Additional arguments to pass to pandas read function
            
        Returns:
            Loaded DataFrame
        """
        # Validate file existence
        if not os.path.exists(file_path):
            self.logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"The file {file_path} does not exist")
        
        # Determine file type and load accordingly
        if file_path.endswith('.parquet'):
            self.data = pd.read_parquet(file_path, **kwargs)
        elif file_path.endswith('.csv'):
            self.data = pd.read_csv(file_path, **kwargs)
        elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            self.data = pd.read_excel(file_path, **kwargs)
        else:
            self.logger.error(f"Unsupported file type: {file_path}")
            raise ValueError(f"Unsupported file type: {file_path}")
        
        self.logger.info(f"Loaded data with {len(self.data)} rows and {len(self.data.columns)} columns")
        
        # Perform data quality checks
        self._check_data_quality()
        
        # Gather basic dataset statistics
        self._analyze_dataset()
        
        return self.data
    
    def _check_data_quality(self) -> None:
        """
        Perform basic data quality checks and report issues.
        """
        # Check for missing values
        missing_values = self.data.isnull().sum()
        columns_with_missing = missing_values[missing_values > 0]
        
        # Check for duplicates
        duplicates = self.data.duplicated().sum()
        
        # Check for constant columns
        constant_columns = [col for col in self.data.columns 
                           if self.data[col].nunique() == 1]
        
        # Check for highly correlated features
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        corr_matrix = self.data[numeric_cols].corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        high_corr_pairs = [(upper_tri.index[i], upper_tri.columns[j], upper_tri.iloc[i, j])
                          for i, j in zip(*np.where(upper_tri > 0.95))]
        
        # Check for extreme outliers
        outlier_columns = {}
        for col in numeric_cols:
            if col != self.target_column and col != self.id_column:
                q1 = self.data[col].quantile(0.25)
                q3 = self.data[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 3 * iqr
                upper_bound = q3 + 3 * iqr
                outliers = self.data[(self.data[col] < lower_bound) | (self.data[col] > upper_bound)]
                if len(outliers) > 0:
                    outlier_columns[col] = len(outliers)
        
        # Store data quality report
        self.data_quality_report = {
            'missing_values': columns_with_missing.to_dict(),
            'duplicate_rows': duplicates,
            'constant_columns': constant_columns,
            'high_correlation_pairs': high_corr_pairs,
            'outlier_columns': outlier_columns
        }
        
        # Log data quality issues
        if columns_with_missing.sum() > 0:
            self.logger.warning(f"Found {columns_with_missing.sum()} missing values in {len(columns_with_missing)} columns")
        
        if duplicates > 0:
            self.logger.warning(f"Found {duplicates} duplicate rows")
        
        if constant_columns:
            self.logger.warning(f"Found {len(constant_columns)} constant columns: {constant_columns}")
        
        if high_corr_pairs:
            self.logger.warning(f"Found {len(high_corr_pairs)} highly correlated feature pairs (>0.95)")
        
        if outlier_columns:
            self.logger.warning(f"Found extreme outliers in {len(outlier_columns)} columns")
    
    def _analyze_dataset(self) -> None:
        """
        Analyze the dataset and gather detailed statistics.
        """
        self.logger.info("\nAnalyzing dataset statistics...")
        
        # Subject info
        n_subjects = self.data[self.id_column].nunique()
        self.logger.info(f"Number of subjects: {n_subjects}")
        
        # Trial info
        n_trials = len(self.data)
        trials_per_subject = self.data.groupby(self.id_column).size()
        self.logger.info(f"Number of trials: {n_trials}")
        self.logger.info(f"Trials per subject: min={trials_per_subject.min()}, max={trials_per_subject.max()}, "
              f"mean={trials_per_subject.mean():.1f}")
        
        # Check for unbalanced subjects
        if trials_per_subject.max() > 2 * trials_per_subject.min():
            self.logger.warning("Unbalanced number of trials per subject detected")
        
        # Turbulence levels
        if 'turbulence' in self.data.columns:
            turb_levels = sorted(self.data['turbulence'].unique())
            self.logger.info(f"Turbulence levels: {turb_levels}")
            
            # Check distribution of turbulence levels
            turb_counts = self.data['turbulence'].value_counts()
            self.logger.info(f"Turbulence level distribution: {turb_counts.to_dict()}")
        
        # Categorize pilots
        if self.id_column == 'pilot_id' and 'pilot_category' not in self.data.columns:
            self.data['pilot_category'] = self.data[self.id_column].apply(self._categorize_pilot)
            
        if 'pilot_category' in self.data.columns:
            pilot_categories = self.data['pilot_category'].value_counts().to_dict()
            self.logger.info(f"Pilot categories: {pilot_categories}")
            self.pilot_info['categories'] = pilot_categories
            
            # Calculate trials per category
            trials_by_category = self.data.groupby('pilot_category').size()
            self.logger.info(f"Trials per category: {trials_by_category.to_dict()}")
            
            # Check if categories are balanced
            if trials_by_category.max() > 2 * trials_by_category.min():
                self.logger.warning("Unbalanced number of trials per pilot category detected")
        
        # Target variable statistics
        if self.target_column in self.data.columns:
            target_stats = self.data[self.target_column].describe()
            self.logger.info(f"\nTarget ({self.target_column}) statistics:")
            self.logger.info(f"Mean: {target_stats['mean']:.4f}")
            self.logger.info(f"Std: {target_stats['std']:.4f}")
            self.logger.info(f"Min: {target_stats['min']:.4f}")
            self.logger.info(f"Max: {target_stats['max']:.4f}")
            
            # Check for target distribution by category if available
            if 'pilot_category' in self.data.columns:
                target_by_category = self.data.groupby('pilot_category')[self.target_column].agg(['mean', 'std'])
                self.logger.info(f"Target by category:\n{target_by_category}")
            
            # Check for target distribution by turbulence if available
            if 'turbulence' in self.data.columns:
                target_by_turb = self.data.groupby('turbulence')[self.target_column].agg(['mean', 'std'])
                self.logger.info(f"Target by turbulence:\n{target_by_turb}")
            
            # Store dataset statistics
            self.dataset_stats = {
                'n_subjects': n_subjects,
                'n_trials': n_trials,
                'trials_per_subject': {
                    'min': trials_per_subject.min(),
                    'max': trials_per_subject.max(),
                    'mean': trials_per_subject.mean(),
                    'std': trials_per_subject.std()
                },
                'target_stats': target_stats.to_dict()
            }
            
            if 'turbulence' in self.data.columns:
                self.dataset_stats['turbulence_levels'] = turb_levels
                self.dataset_stats['turbulence_distribution'] = turb_counts.to_dict()
            
            if 'pilot_category' in self.data.columns:
                self.dataset_stats['pilot_categories'] = pilot_categories
                self.dataset_stats['target_by_category'] = target_by_category.to_dict()
                
                # Store pilot IDs by category
                pilot_ids_by_category = {}
                for category in pilot_categories.keys():
                    pilot_ids_by_category[category] = self.data[self.data['pilot_category'] == category][self.id_column].unique().tolist()
                self.dataset_stats['pilot_ids_by_category'] = pilot_ids_by_category
    
    def preprocess_data(self, handle_missing: bool = True, 
                        scale_method: str = 'robust',
                        outlier_handling: str = 'robust') -> pd.DataFrame:
        """
        Preprocess the loaded data.
        
        Args:
            handle_missing: Whether to handle missing values
            scale_method: Method for scaling features ('standard', 'robust', 'none')
            outlier_handling: Method for handling outliers ('none', 'robust', 'clip', 'winsorize')
            
        Returns:
            Preprocessed DataFrame
        """
        self.logger.info("\nPreprocessing data...")
        
        if self.data is None:
            self.logger.error("No data loaded. Call load_data() first.")
            raise ValueError("No data loaded. Call load_data() first.")
        
        # Make a copy to avoid modifying the original data
        self.preprocessed_data = self.data.copy()
        
        # Identify numerical columns (excluding target and ID)
        exclude_cols = [self.id_column, self.target_column] + self.excluded_columns
        if 'pilot_category' in self.preprocessed_data.columns:
            exclude_cols.append('pilot_category')
            
        num_cols = self.preprocessed_data.select_dtypes(include=[np.number]).columns.tolist()
        num_cols = [col for col in num_cols if col not in exclude_cols]
        
        # Handle missing values
        if handle_missing:
            missing_counts = self.preprocessed_data[num_cols].isnull().sum()
            cols_with_missing = missing_counts[missing_counts > 0]
            
            if len(cols_with_missing) > 0:
                self.logger.info(f"Handling missing values in {len(cols_with_missing)} columns")
                # Use KNN imputer for better handling of missing values
                self.imputer = KNNImputer(n_neighbors=5)
                self.preprocessed_data[num_cols] = self.imputer.fit_transform(self.preprocessed_data[num_cols])
                
                # Save imputer
                if self.output_dir:
                    joblib.dump(self.imputer, os.path.join(self.output_dir, 'knn_imputer.joblib'))
        
        # Handle outliers
        if outlier_handling != 'none':
            self.logger.info(f"Handling outliers using {outlier_handling} method")
            for col in num_cols:
                if outlier_handling == 'robust':
                    # Use median and IQR
                    Q1 = self.preprocessed_data[col].quantile(0.25)
                    Q3 = self.preprocessed_data[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    # Count outliers before clipping
                    outliers = self.preprocessed_data[(self.preprocessed_data[col] < lower_bound) | 
                                                     (self.preprocessed_data[col] > upper_bound)]
                    if len(outliers) > 0:
                        self.logger.info(f"  Column {col}: clipping {len(outliers)} outliers")
                    
                    self.preprocessed_data[col] = self.preprocessed_data[col].clip(lower_bound, upper_bound)
                    
                elif outlier_handling == 'clip':
                    # Use mean and standard deviation
                    mean = self.preprocessed_data[col].mean()
                    std = self.preprocessed_data[col].std()
                    
                    # Count outliers before clipping
                    outliers = self.preprocessed_data[(self.preprocessed_data[col] < mean - 3*std) | 
                                                     (self.preprocessed_data[col] > mean + 3*std)]
                    if len(outliers) > 0:
                        self.logger.info(f"  Column {col}: clipping {len(outliers)} outliers")
                    
                    self.preprocessed_data[col] = self.preprocessed_data[col].clip(mean - 3*std, mean + 3*std)
                    
                elif outlier_handling == 'winsorize':
                    # Winsorize at 5th and 95th percentiles
                    from scipy import stats
                    
                    # Count outliers before winsorizing
                    lower = self.preprocessed_data[col].quantile(0.05)
                    upper = self.preprocessed_data[col].quantile(0.95)
                    outliers = self.preprocessed_data[(self.preprocessed_data[col] < lower) | 
                                                     (self.preprocessed_data[col] > upper)]
                    if len(outliers) > 0:
                        self.logger.info(f"  Column {col}: winsorizing {len(outliers)} outliers")
                    
                    self.preprocessed_data[col] = stats.mstats.winsorize(self.preprocessed_data[col], limits=[0.05, 0.05])
        
        # Scale features if specified
        if scale_method != 'none':
            self.logger.info(f"Scaling features using {scale_method} scaler")
            if scale_method == 'standard':
                self.scaler = StandardScaler()
            elif scale_method == 'robust':
                self.scaler = RobustScaler()
            else:
                raise ValueError(f"Unknown scaling method: {scale_method}")
                
            # Store original column names
            scaled_cols = num_cols
            
            # Fit the scaler and transform the data
            scaled_data = self.scaler.fit_transform(self.preprocessed_data[scaled_cols])
            
            # Update DataFrame with scaled values
            for i, col in enumerate(scaled_cols):
                self.preprocessed_data[col] = scaled_data[:, i]
            
            # Store the scaler for later use
            if self.output_dir:
                joblib.dump(self.scaler, os.path.join(self.output_dir, 'feature_scaler.joblib'))
        
        # Check for target distribution after preprocessing
        if self.target_column in self.preprocessed_data.columns:
            target_stats_after = self.preprocessed_data[self.target_column].describe()
            self.logger.info(f"\nTarget ({self.target_column}) statistics after preprocessing:")
            self.logger.info(f"Mean: {target_stats_after['mean']:.4f}")
            self.logger.info(f"Std: {target_stats_after['std']:.4f}")
            self.logger.info(f"Min: {target_stats_after['min']:.4f}")
            self.logger.info(f"Max: {target_stats_after['max']:.4f}")
        
        self.logger.info(f"Preprocessing complete. Data shape: {self.preprocessed_data.shape}")
        return self.preprocessed_data
    
    @staticmethod
    def _categorize_pilot(pilot_id: str) -> str:
        """
        Categorize pilots based on their ID.
        
        Args:
            pilot_id: The pilot's identifier
            
        Returns:
            The pilot category as a string
        """
        pid = str(pilot_id)
        if pid.startswith('8'):
            return 'minimal_exp'
        elif pid.startswith('9'):
            return 'commercial'
        else:
            return 'air_force'
    
    def get_feature_columns(self, include_target: bool = False) -> List[str]:
        """
        Get list of feature columns (excluding ID and other non-feature columns).
        
        Args:
            include_target: Whether to include the target column
            
        Returns:
            List of feature column names
        """
        if self.preprocessed_data is None:
            self.logger.error("No preprocessed data available. Call preprocess_data() first.")
            raise ValueError("No preprocessed data available. Call preprocess_data() first.")
            
        exclude_cols = [self.id_column] + self.excluded_columns
        if not include_target:
            exclude_cols.append(self.target_column)
            
        if 'pilot_category' in self.preprocessed_data.columns:
            exclude_cols.append('pilot_category')
            
        all_cols = self.preprocessed_data.columns.tolist()
        feature_cols = [col for col in all_cols if col not in exclude_cols]
        
        return feature_cols
    
    def save_dataset_statistics(self) -> str:
        """
        Save dataset statistics to a file.
        
        Returns:
            Path to the saved file
        """
        if not self.output_dir:
            self.logger.warning("No output directory specified, skipping saving dataset statistics")
            return None
            
        import json
        
        # Create a JSON-serializable version of the stats
        serializable_stats = {}
        for key, value in self.dataset_stats.items():
            if isinstance(value, np.ndarray):
                serializable_stats[key] = value.tolist()
            elif isinstance(value, pd.Series):
                serializable_stats[key] = value.to_dict()
            else:
                serializable_stats[key] = value
        
        # Save to file
        stats_path = os.path.join(self.output_dir, 'dataset_statistics.json')
        with open(stats_path, 'w') as f:
            json.dump(serializable_stats, f, indent=2)
            
        # Also save data quality report
        quality_path = os.path.join(self.output_dir, 'data_quality_report.json')
        serializable_quality = {}
        for key, value in self.data_quality_report.items():
            if isinstance(value, np.ndarray):
                serializable_quality[key] = value.tolist()
            elif isinstance(value, pd.Series):
                serializable_quality[key] = value.to_dict()
            else:
                serializable_quality[key] = value
                
        with open(quality_path, 'w') as f:
            json.dump(serializable_quality, f, indent=2)
            
        self.logger.info(f"Dataset statistics saved to: {stats_path}")
        self.logger.info(f"Data quality report saved to: {quality_path}")
        
        return stats_path
