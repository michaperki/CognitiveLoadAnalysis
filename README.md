# CognitiveLoadAnalysis

CognitiveLoadAnalysis is an end-to-end machine learning solution for predicting cognitive load from physiological signals. The system automates data preprocessing, feature engineering/selection, model training, extensive validation (including leave-one-pilot-out cross‑validation), visualization, and report generation.

---

## Features

- **Data Handling:**  
  Automated data loading from CSV, Parquet, or Excel files with built‑in quality checks.
  
- **Feature Engineering:**  
  Generates hundreds of features using methods such as core feature extraction, turbulence interactions, pilot normalization, polynomial transformations, signal ratios, and experience‑based features.
  
- **Feature Selection:**  
  Combines several strategies (correlation, mutual information, random forest importance, SHAP, and recursive feature elimination) to select an optimal set of features.
  
- **Model Training & Optimization:**  
  Supports gradient boosting and other models along with hyperparameter optimization using optuna.
  
- **Validation:**  
  Implements robust techniques including leave‑one‑pilot‑out cross‑validation and learning curve analysis for evaluating generalization.
  
- **Visualization & Reporting:**  
  Automatically generates plots and a comprehensive HTML report summarizing performance and insights.

---

## Requirements

- Python 3.8+
- Dependencies (see `requirements.txt`):  
  numpy, pandas, scikit‑learn, matplotlib, seaborn, joblib, PyYAML, optuna, xgboost, lightgbm, etc.

---

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://your-repo-url.git
   cd CognitiveLoadAnalysis
   ```
2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate   # For Windows: venv\Scripts\activate
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

To run the full pipeline, execute:
```bash
python experiment.py --config config/default_config.yaml --data_path path/to/data.parquet --output_dir output_folder
```
Additional command line arguments let you override configuration settings (e.g., random seed, skipping validation).

---

## Repository Structure

```
CognitiveLoadAnalysis/
├── config/                    # YAML configuration files
├── data/                      # Data loading and splitting utilities
│   ├── data_loader.py
│   └── data_splitter.py
├── features/                  # Feature engineering and selection scripts
│   ├── basic_features.py
│   ├── feature_engineer.py
│   ├── feature_selector.py
│   └── pilot_features.py
├── models/                    # Model implementations and training scripts
│   ├── base_model.py
│   ├── gradient_boosting.py
│   └── model_trainer.py
├── validation/                # Validation and cross‑validation routines
│   ├── cross_validation.py
│   ├── validation_manager.py
│   └── learning_curves.py
├── visualization/             # Visualization tools (error analysis, feature plots, etc.)
│   ├── visualizer.py
│   ├── error_analysis.py
│   └── feature_plots.py
├── reporting/                 # Report and paper generation tools
│   └── report_generator.py
├── utils/                     # Logging and statistical utilities
│   ├── logging_utils.py
│   └── statistical_utils.py
└── experiment.py              # Experiment runner that integrates the entire pipeline
```

---

## Contact

For questions or feedback, please open an issue or contact [michaelperkins@smu.edu].

