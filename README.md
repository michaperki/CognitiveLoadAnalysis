# Cognitive Load Analysis Pipeline v30

A comprehensive implementation for predicting and analyzing cognitive load from physiological data, with advanced modeling approaches and detailed reporting.

## Overview

This pipeline analyzes physiological data to predict cognitive load (mental effort) using three modeling approaches:

1. **Global Model**: Trained on data from all subjects
2. **Subject-specific Models**: Individual models trained for each subject
3. **Adaptive Transfer Learning**: Optimally weighted combination of global and subject-specific models

The pipeline implements complete feature engineering, model training, evaluation with statistical significance testing, and comprehensive HTML reporting.

## Key Features

- Robust data preprocessing with leak prevention
- Advanced feature engineering for physiological signals
- Multiple modeling approaches with statistical comparison
- Confidence intervals for all performance metrics
- Comprehensive HTML report with interactive visualizations
- Detailed methodological explanation and limitations discussion

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/cognitive-load-pipeline.git
cd cognitive-load-pipeline

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
python cognitive_load_pipeline_v30.py --data path/to/your/data.csv
```

### Advanced Options

```bash
python cognitive_load_pipeline_v30.py \
  --data path/to/your/data.csv \
  --output results_directory \
  --features 30 \
  --test-size 0.2 \
  --target mental_effort \
  --subject pilot_id \
  --seed 42
```

### Command Line Arguments

- `--data`: Path to the input CSV file (required)
- `--output`: Directory to store results (default: auto-generated timestamped directory)
- `--features`: Number of features to select (default: 30)
- `--test-size`: Proportion of data for testing (default: 0.2)
- `--target`: Name of the target column (default: 'mental_effort')
- `--subject`: Name of the subject identifier column (default: 'pilot_id')
- `--no-report`: Skip HTML report generation
- `--seed`: Random seed for reproducibility (default: 42)

## Input Data Format

The input CSV file should contain:

- A subject identifier column (e.g., `pilot_id`)
- A cognitive load measure column (e.g., `mental_effort`)
- Physiological signal data (e.g., EDA, HR, HRV, temperature, etc.)

## Output

The pipeline generates:

1. **Preprocessed Data**: Cleaned and ready for analysis
2. **Engineered Features**: Temporal, frequency domain, and compound features
3. **Trained Models**: Global, subject-specific, and adaptive transfer models
4. **Performance Metrics**: R², RMSE, confidence intervals, significance tests
5. **HTML Report**: Comprehensive visualization and analysis

## HTML Report

The generated HTML report includes:

- Executive summary with key findings
- Dataset overview and visualization
- Feature engineering explanation
- Model performance comparison
- Subject-specific analysis
- Adaptive transfer learning results
- Prediction accuracy analysis
- Discussion of implications
- Limitations and future work

## Example Results

Subject-specific models significantly outperform global models when generalizing to new subjects, with adaptive transfer learning providing the best overall performance:

| Model Type | Cross-Subject R² | Within-Subject R² |
|------------|-----------------|-------------------|
| Global     | -0.984          | 0.920             |
| Subject-specific | 0.892      | -                |
| Adaptive Transfer | 0.926     | -                |

The negative cross-subject R² indicates that cognitive load patterns are highly individual and resist generalization.

## Requirements

- Python 3.8+
- NumPy
- Pandas
- Scikit-learn
- SciPy
- Matplotlib
- Seaborn
- Jinja2
- Joblib

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
