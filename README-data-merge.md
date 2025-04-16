# Pilot Data Merge Tool

This utility script prepares and merges pilot biometric data for the cognitive load analysis pipeline.

## Overview

The data merge tool is a critical preprocessing step for the cognitive load analysis pipeline. It performs the following essential functions:

- Normalizes pilot IDs across different data formats
- Standardizes trial names for consistent comparison
- Extracts and processes windowed physiological features
- Merges tabular data with windowed physiological features
- Computes statistical metrics from raw signal data

## Why This Is Necessary

The cognitive load analysis pipeline requires preprocessed data that combines multiple data sources:

1. **Tabular Data**: Contains demographic information, subjective ratings, and performance metrics
2. **Windowed Physiological Data**: Contains time-series biometric signals (EDA, HR, HRV, etc.)

These datasets often have inconsistent ID formats and trial naming conventions, making direct merging impossible. This script handles all the normalization and feature extraction needed before running the main pipeline.

## Usage

### Basic Usage

```bash
python merge_pilot_data.py --data_path path/to/tabular_data.parquet --windowed_data_dir path/to/json_files/
```

### Advanced Options

```bash
python merge_pilot_data.py \
  --data_path path/to/tabular_data.parquet \
  --windowed_data_dir path/to/json_files/ \
  --output_path merged_data.csv \
  --debug
```

### Command Line Arguments

- `--data_path`: Path to the tabular data file (Parquet format)
- `--windowed_data_dir`: Directory containing windowed physiological data (JSON files)
- `--output_path`: Path to save the merged dataset (default: merged_data.csv)
- `--debug`: Enable debug mode with more detailed logging

## Input Data Format

The script expects:

1. **Tabular Data**: A Parquet file containing pilot information and ratings
2. **Windowed Data**: JSON files containing physiological signals with each file named by pilot ID (e.g., ID001.json)

## Output

The script produces a merged CSV file that:

- Contains normalized pilot IDs and standardized trial names
- Includes all tabular data columns
- Adds statistical features derived from windowed physiological signals
- Is ready for input into the cognitive load analysis pipeline

## Key Features

- Robust ID normalization to handle prefixes and different numeric formats
- Smart trial name matching using standardized text and phase numbers
- Advanced feature extraction from windowed physiological data
- Comprehensive error handling and detailed logging
- Fallback matching strategies when exact matches aren't found

## Troubleshooting

The most common issues relate to:

- **ID Format Mismatches**: The script handles most common formats, but custom formats may require code modification
- **Trial Name Inconsistencies**: Check the logs for trial name matching issues
- **Missing Data**: Ensure all required JSON files exist in the windowed data directory

Check the log output for detailed information about any issues encountered during processing.
