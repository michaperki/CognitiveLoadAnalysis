
"""
Standalone script to fix pilot ID formats and merge datasets.

This script can be run directly to debug the merge issues between
tabular data and windowed features.
"""

import os
import sys
import pandas as pd
import numpy as np
import json
import logging
import re
import argparse
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Fix and merge pilot biometrics data')
    
    parser.add_argument(
        '--data_path',
        type=str,
        default='../pre_proccessed_table_data.parquet',
        help='Path to the tabular data file'
    )
    
    parser.add_argument(
        '--windowed_data_dir',
        type=str,
        default='../',
        help='Directory containing windowed physiological data'
    )
    
    parser.add_argument(
        '--output_path',
        type=str,
        default='merged_data.csv',
        help='Path to save the merged dataset'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode with more detailed logging'
    )
    
    return parser.parse_args()

def load_tabular_data(path):
    """Load tabular data from Parquet file."""
    logger.info(f"Loading tabular data from {path}")
    try:
        df = pd.read_parquet(path)
        logger.info(f"Loaded tabular data with shape: {df.shape}")
        logger.info(f"Columns: {', '.join(df.columns[:10])}...")
        return df
    except Exception as e:
        logger.error(f"Error loading tabular data: {str(e)}")
        raise

def load_windowed_data(directory):
    """Load windowed data from JSON files."""
    logger.info(f"Loading windowed data from {directory}")
    
    windowed_data = {}
    
    try:
        # Get all JSON files in the directory
        json_files = [f for f in os.listdir(directory) if f.endswith('.json')]
        logger.info(f"Found {len(json_files)} JSON files")
        
        for filename in json_files:
            file_path = os.path.join(directory, filename)
            
            try:
                # Extract pilot ID from filename (assuming format ID001.json)
                pilot_id = filename.strip('.json')
                
                # Load the JSON data
                with open(file_path, 'r') as f:
                    pilot_data = json.load(f)
                
                # Store the pilot data
                windowed_data[pilot_id] = pilot_data
                
                # Log basic info
                logger.info(f"Loaded data for pilot {pilot_id}: {len(pilot_data)} trials")
                
            except Exception as e:
                logger.warning(f"Error loading file {filename}: {str(e)}")
                continue
        
        return windowed_data
        
    except Exception as e:
        logger.error(f"Error in load_windowed_data: {str(e)}")
        raise

def extract_basic_features(pilot_id, trial_data):
    """Extract basic features from trial data."""
    features = {}
    
    # Extract meta data
    meta_data = trial_data.get('meta_data', {})
    trial_id = meta_data.get('trial', 'unknown')
    turbulence = meta_data.get('turbulence', 0)
    
    # Include meta data in features
    features['pilot_id'] = pilot_id
    features['trial'] = trial_id
    features['turbulence'] = turbulence
    
    # Get windowed features
    windows = trial_data.get('windowed_features', [])
    if not windows:
        logger.warning(f"No windowed features found for pilot {pilot_id}, trial {trial_id}")
        return features
    
    # Track all signal types
    signal_types = set()
    for window in windows:
        signal_types.update(window.keys())
    
    # Remove timestamp from signal types
    if 'timestamp' in signal_types:
        signal_types.remove('timestamp')
    
    # Process each signal type
    for signal_type in signal_types:
        # Skip eng_features for now
        if 'eng_features' in signal_type:
            continue
            
        # Collect all values for this signal type across windows
        all_values = []
        for window in windows:
            if signal_type in window:
                values = window[signal_type]
                if isinstance(values, list):
                    all_values.extend(values)
        
        # Skip if no values
        if not all_values:
            continue
            
        # Convert to numpy array for efficient computation
        values_array = np.array(all_values)
        
        # Calculate basic statistics
        features[f"{signal_type}_mean"] = float(np.mean(values_array))
        features[f"{signal_type}_std"] = float(np.std(values_array))
        features[f"{signal_type}_min"] = float(np.min(values_array))
        features[f"{signal_type}_max"] = float(np.max(values_array))
        features[f"{signal_type}_median"] = float(np.median(values_array))
        
        # Count number of windows
        features[f"{signal_type}_window_count"] = len(windows)
        
        # Calculate rate of change (simple version)
        if len(values_array) > 1:
            diff = np.diff(values_array)
            features[f"{signal_type}_diff_mean"] = float(np.mean(diff))
            features[f"{signal_type}_diff_std"] = float(np.std(diff))
    
    return features

def extract_all_features(windowed_data):
    """Extract features from all trials for all pilots."""
    logger.info("Extracting features from all trials")
    
    all_features = []
    
    for pilot_id, pilot_trials in windowed_data.items():
        # Clean pilot ID (remove ID prefix)
        clean_pilot_id = pilot_id.replace("ID", "") if pilot_id.startswith("ID") else pilot_id
        
        for trial_data in pilot_trials:
            # Extract features
            trial_features = extract_basic_features(clean_pilot_id, trial_data)
            all_features.append(trial_features)
    
    # Convert to DataFrame
    features_df = pd.DataFrame(all_features)
    
    logger.info(f"Extracted features for {len(features_df)} trials")
    logger.info(f"Feature columns: {', '.join(features_df.columns[:10])}...")
    
    return features_df

def normalize_pilot_ids(df, column='pilot_id'):
    """Normalize pilot IDs for consistent comparison."""
    # Make a copy to avoid modifying the original
    df_copy = df.copy()
    
    # Convert to string
    df_copy[column] = df_copy[column].astype(str)
    
    # Remove ID prefix if present
    df_copy[column] = df_copy[column].str.replace('^ID', '', regex=True)
    
    # Remove trailing decimal point and zeros
    df_copy[column] = df_copy[column].str.replace(r'\.0$', '', regex=True)
    
    # Remove leading zeros
    df_copy[column] = df_copy[column].str.replace(r'^0+', '', regex=True)
    
    return df_copy

def standardize_trial_names(df, column='trial'):
    """Standardize trial names for consistent comparison."""
    # Make a copy to avoid modifying the original
    df_copy = df.copy()
    
    # Convert to string
    df_copy[column] = df_copy[column].astype(str)
    
    # Create a standardized column
    df_copy['trial_std'] = (
        df_copy[column]
        .str.lower()
        .str.replace(r'\s+', '', regex=True)  # Remove spaces
        .str.replace(r'[_-]', '', regex=True)  # Remove underscores and hyphens
    )
    
    # Extract phase numbers if present
    df_copy['phase_num'] = df_copy['trial_std'].str.extract(r'phase(\d+)', expand=False)
    
    return df_copy

def merge_datasets(tabular_df, features_df):
    """Merge tabular data with features using robust matching."""
    logger.info("Merging tabular data with extracted features")
    
    # Make copies to avoid modifying the originals
    tabular_copy = tabular_df.copy()
    features_copy = features_df.copy()
    
    # Log data types and sample values
    logger.info(f"Tabular pilot_id type: {tabular_copy['pilot_id'].dtype}")
    logger.info(f"Features pilot_id type: {features_copy['pilot_id'].dtype}")
    logger.info(f"Tabular trial type: {tabular_copy['trial'].dtype}")
    logger.info(f"Features trial type: {features_copy['trial'].dtype}")
    
    logger.info(f"Tabular pilot_id sample: {tabular_copy['pilot_id'].head().tolist()}")
    logger.info(f"Features pilot_id sample: {features_copy['pilot_id'].head().tolist()}")
    logger.info(f"Tabular trial sample: {tabular_copy['trial'].head().tolist()}")
    logger.info(f"Features trial sample: {features_copy['trial'].head().tolist()}")
    
    # Step 1: Normalize pilot IDs in both DataFrames
    tabular_norm = normalize_pilot_ids(tabular_copy)
    features_norm = normalize_pilot_ids(features_copy)
    
    # Step 2: Standardize trial names
    tabular_norm = standardize_trial_names(tabular_norm)
    features_norm = standardize_trial_names(features_norm)
    
    # Step 3: Try merging on normalized pilot ID and standardized trial
    merged_df = pd.merge(
        tabular_norm,
        features_norm,
        on=['pilot_id', 'trial_std'],
        how='inner',
        suffixes=('_tab', '_feat')
    )
    
    # Remove helper columns
    if 'trial_std' in merged_df.columns:
        merged_df = merged_df.drop(columns=['trial_std'])
    if 'phase_num' in merged_df.columns:
        merged_df = merged_df.drop(columns=['phase_num'])
    
    # If no matches found, try more lenient approach
    if len(merged_df) == 0:
        logger.warning("No matches found. Trying more lenient matching...")
        
        # Try merging on pilot_id and phase_num
        if 'phase_num' in tabular_norm.columns and 'phase_num' in features_norm.columns:
            merged_df = pd.merge(
                tabular_norm,
                features_norm,
                on=['pilot_id', 'phase_num'],
                how='inner',
                suffixes=('_tab', '_feat')
            )
            
            # Remove helper columns
            if 'trial_std' in merged_df.columns:
                merged_df = merged_df.drop(columns=['trial_std'])
            if 'phase_num' in merged_df.columns:
                merged_df = merged_df.drop(columns=['phase_num'])
    
    # If still no matches, try pilot_id only
    if len(merged_df) == 0:
        logger.warning("No matches found. Trying pilot_id only...")
        
        # Try merging on pilot_id only
        merged_df = pd.merge(
            tabular_norm[['pilot_id', 'trial', 'trial_std']],
            features_norm,
            on='pilot_id',
            how='inner',
            suffixes=('_tab', '_feat')
        )
        
        if len(merged_df) > 0:
            logger.info(f"Found {len(merged_df)} rows with matching pilot_id but mismatched trial")
            
            # Print a sample of the mismatches
            mismatch_df = merged_df[['pilot_id', 'trial_tab', 'trial_feat']].head(10)
            logger.info(f"Sample mismatches:\n{mismatch_df}")
            
            # Try to create a complete merged dataset by matching each tabular row
            # with the corresponding feature row that has the same phase number
            
            # First, extract and standardize phase numbers
            merged_df['phase_tab'] = merged_df['trial_std'].str.extract(r'phase(\d+)', expand=False)
            merged_df['phase_feat'] = merged_df['trial_std_feat'].str.extract(r'phase(\d+)', expand=False)
            
            # Filter to rows where phases match
            phase_matched = merged_df[merged_df['phase_tab'] == merged_df['phase_feat']]
            
            if len(phase_matched) > 0:
                logger.info(f"Found {len(phase_matched)} rows with matching phase numbers")
                
                # Get the unique (pilot_id, trial_tab) combinations
                unique_pilot_trials = phase_matched[['pilot_id', 'trial_tab']].drop_duplicates()
                
                # Create a new complete dataset
                complete_rows = []
                
                for _, row in unique_pilot_trials.iterrows():
                    pilot_id = row['pilot_id']
                    trial = row['trial_tab']
                    
                    # Get the full tabular data for this pilot and trial
                    tab_data = tabular_norm[(tabular_norm['pilot_id'] == pilot_id) & 
                                           (tabular_norm['trial'] == trial)]
                    
                    # Get the matching feature data
                    matching_features = phase_matched[(phase_matched['pilot_id'] == pilot_id) & 
                                                     (phase_matched['trial_tab'] == trial)]
                    
                    if not tab_data.empty and not matching_features.empty:
                        # Combine the data
                        tab_row = tab_data.iloc[0].to_dict()
                        feat_row = matching_features.iloc[0].to_dict()
                        
                        # Remove duplicate columns
                        for col in ['pilot_id', 'trial', 'trial_std', 'phase_tab', 'phase_feat']:
                            if col + '_feat' in feat_row:
                                feat_row.pop(col + '_feat')
                        
                        # Merge dictionaries
                        combined_row = {**tab_row, **feat_row}
                        complete_rows.append(combined_row)
                
                if complete_rows:
                    # Create a new DataFrame with the combined data
                    merged_df = pd.DataFrame(complete_rows)
                    
                    # Clean up helper columns
                    for col in ['trial_std', 'phase_tab', 'phase_feat']:
                        if col in merged_df.columns:
                            merged_df = merged_df.drop(columns=[col])
    
    # Log results
    if len(merged_df) > 0:
        logger.info(f"Successfully merged {len(merged_df)} rows")
        logger.info(f"Merged data has {merged_df['pilot_id'].nunique()} unique pilots")
        
        # Check which trial column to use
        if 'trial_feat' in merged_df.columns:
            logger.info(f"Merged data has {merged_df['trial_feat'].nunique()} unique trials")
        else:
            logger.info(f"Merged data has {merged_df['trial'].nunique()} unique trials")
    else:
        logger.warning("Failed to merge datasets. No matches found.")
    
    return merged_df

def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Set log level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Load tabular data
        tabular_df = load_tabular_data(args.data_path)
        
        # Load windowed data
        windowed_data = load_windowed_data(args.windowed_data_dir)
        
        # Extract features
        features_df = extract_all_features(windowed_data)
        
        # Merge datasets
        merged_df = merge_datasets(tabular_df, features_df)
        
        # Save merged dataset
        if len(merged_df) > 0:
            merged_df.to_csv(args.output_path, index=False)
            logger.info(f"Saved merged dataset to {args.output_path}")
        else:
            logger.error("No data merged. Unable to save output file.")
        
    except Exception as e:
        logger.error(f"Error during processing: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()
