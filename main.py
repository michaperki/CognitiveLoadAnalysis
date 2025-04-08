#!/usr/bin/env python
"""
Cognitive Load Analysis (v9)

A comprehensive pipeline for predicting cognitive load from physiological signals,
with enhanced validation frameworks designed to detect overfitting.
"""

import os
import argparse
import yaml
import logging
from datetime import datetime
from pathlib import Path

from data.data_loader import DataLoader
from data.data_splitter import DataSplitter
from features.feature_engineer import FeatureEngineer
from features.feature_selector import FeatureSelector
from models.model_trainer import ModelTrainer
from validation.validation_manager import ValidationManager
from visualization.visualizer import Visualizer
from reporting.report_generator import ReportGenerator
from utils.logging_utils import setup_logging


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Cognitive Load Analysis Pipeline (v9)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--config', 
        type=str, 
        default='config/default_config.yaml',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--data_path', 
        type=str, 
        help='Path to the tabular data file (overrides config)'
    )
    
    parser.add_argument(
        '--output_dir', 
        type=str, 
        help='Directory to save output files (overrides config)'
    )
    
    parser.add_argument(
        '--seed', 
        type=int,
        help='Random seed for reproducibility (overrides config)'
    )
    
    parser.add_argument(
        '--skip_validation',
        action='store_true',
        help='Skip extensive validation (useful for quick tests)'
    )
    
    return parser.parse_args()


def load_config(config_path):
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
    
    Returns:
        Dictionary with configuration
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def update_config_with_args(config, args):
    """
    Update configuration with command line arguments.
    
    Args:
        config: Configuration dictionary
        args: Command line arguments
    
    Returns:
        Updated configuration dictionary
    """
    # Override data path if specified
    if args.data_path:
        config['data']['file_path'] = args.data_path
    
    # Override output directory if specified
    if args.output_dir:
        config['output']['base_dir'] = args.output_dir
    
    # Override random seed if specified
    if args.seed:
        config['validation']['random_seed'] = args.seed
    
    # Skip validation if specified
    if args.skip_validation:
        config['validation']['leave_one_pilot_out'] = False
        config['validation']['generate_learning_curves'] = False
        config['validation']['permutation_test'] = False
        config['validation']['significance_testing'] = False
    
    return config


def create_output_directory(base_dir):
    """
    Create a timestamped output directory.
    
    Args:
        base_dir: Base directory name
    
    Returns:
        Path to the created directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{base_dir}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create subdirectories
    subdirs = [
        'models', 
        'features', 
        'validation', 
        'visualizations', 
        'report'
    ]
    
    for subdir in subdirs:
        os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)
    
    return output_dir


def main():
    """Main entry point for the Cognitive Load Analysis pipeline."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Load and update configuration
    config = load_config(args.config)
    config = update_config_with_args(config, args)
    
    # Create output directory
    output_dir = create_output_directory(config['output']['base_dir'])
    
    # Set up logging
    log_file = os.path.join(output_dir, 'pipeline.log')
    setup_logging(log_file)
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting Cognitive Load Analysis Pipeline (v9) - {datetime.now()}")
    logger.info(f"Output directory: {output_dir}")
    
    # Save configuration to output directory
    config_path = os.path.join(output_dir, 'config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    try:
        # 1. Load and preprocess data
        logger.info("Step 1: Loading data")
        data_loader = DataLoader(
            target_column=config['data']['target_column'],
            id_column=config['data']['id_column'],
            excluded_columns=config['data']['excluded_columns'],
            output_dir=output_dir
        )
        data = data_loader.load_data(config['data']['file_path'])
        preprocessed_data = data_loader.preprocess_data()

        # 2. Split data for validation
        logger.info("Step 2: Splitting data for validation")
        splitter = DataSplitter(
            id_column=config['data']['id_column'],
            random_seed=config['validation']['random_seed'],
            output_dir=output_dir
        )
        
        # Standard train/test split
        train_data, test_data = splitter.train_test_split(
            preprocessed_data, 
            test_size=config['validation']['test_size']
        )
        
        # 3. Engineer features
        logger.info("Step 3: Engineering features")
        feature_engineer = FeatureEngineer(
            id_column=config['data']['id_column'],
            target_column=config['data']['target_column'],
            seed=config['validation']['random_seed'],
            output_dir=output_dir,
            config=config['features']
        )
        
        engineered_train = feature_engineer.engineer_features(train_data)
        engineered_test = feature_engineer.engineer_features(test_data)
        
        # 4. Select features
        logger.info("Step 4: Selecting features")
        feature_selector = FeatureSelector(
            method=config['feature_selection']['method'],
            n_features=config['feature_selection']['n_features'],
            use_shap=config['feature_selection']['use_shap'],
            target_column=config['data']['target_column'],
            seed=config['validation']['random_seed'],
            output_dir=output_dir
        )
        
        selected_features = feature_selector.select_features(
            engineered_train, 
            sample_weight=None
        )

        # 5. Create feature sets
        logger.info("Step 5: Creating feature sets")
        feature_sets = feature_selector.create_feature_sets(engineered_train, selected_features)
        
        # 6. Train and evaluate models
        logger.info("Step 6: Training and evaluating models")
        model_trainer = ModelTrainer(
            id_column=config['data']['id_column'],
            target_column=config['data']['target_column'],
            seed=config['validation']['random_seed'],
            output_dir=output_dir,
            model_config=config['model']
        )
        
        # Train models on different feature sets
        model_results = model_trainer.train_models(
            engineered_train,
            engineered_test,
            feature_sets
        )
        
        # 7. Comprehensive validation
        logger.info("Step 7: Performing comprehensive validation")
        validation_manager = ValidationManager(
            id_column=config['data']['id_column'],
            target_column=config['data']['target_column'],
            seed=config['validation']['random_seed'],
            output_dir=output_dir,
            config=config['validation']
        )
        
        # Perform leave-one-pilot-out validation
        if config['validation']['leave_one_pilot_out']:
            logger.info("Performing leave-one-pilot-out validation")
            lopo_results = validation_manager.leave_one_pilot_out_validation(
                preprocessed_data,
                feature_engineer,
                feature_selector,
                model_trainer
            )
        
        # Generate learning curves
        if config['validation']['generate_learning_curves']:
            logger.info("Generating learning curves")
            learning_curve_results = validation_manager.generate_learning_curves(
                preprocessed_data,
                feature_engineer,
                feature_selector,
                model_trainer
            )
        
        # 8. Perform additional analyses
        if config['analysis']['analyze_by_pilot_category']:
            logger.info("Analyzing by pilot category")
            category_results = model_trainer.analyze_results_by_category(
                engineered_test,
                model_trainer.best_model
            )
        
        if config['analysis']['analyze_by_turbulence']:
            logger.info("Analyzing by turbulence level")
            turbulence_results = model_trainer.analyze_turbulence_performance(
                engineered_test,
                model_trainer.best_model
            )

        # 9. Create visualizations
        logger.info("Step 8: Creating visualizations")
        visualizer = Visualizer(output_dir=output_dir)
        
        # Generate all visualizations based on config
        visualizer.generate_visualizations(
            data=preprocessed_data,
            train_data=engineered_train,
            test_data=engineered_test,
            model_results=model_results,
            validation_results=validation_manager.get_results(),
            target_column=config['data']['target_column'],
            config=config['visualization']
        )
        
        # 10. Generate final report
        if config['output']['generate_report']:
            logger.info("Step 9: Generating report")
            report_generator = ReportGenerator(output_dir=output_dir)
            
            report_path = report_generator.generate_report(
                data_stats=data_loader.dataset_stats,
                feature_engineering_stats=feature_engineer.feature_engineering_stats,
                model_results=model_results,
                validation_results=validation_manager.get_results(),
                analysis_results={
                    'category_metrics': category_results if config['analysis']['analyze_by_pilot_category'] else None,
                    'turbulence_metrics': turbulence_results if config['analysis']['analyze_by_turbulence'] else None,
                    'feature_importance': feature_selector.feature_importance
                }
            )
            
            logger.info(f"Report generated at: {report_path}")
        
        # Generate final summary
        logger.info("Step 10: Generating final summary")
        visualizer.final_summary({
            'best_model': model_trainer.best_model,
            'feature_importance': feature_selector.feature_importance,
            'validation_results': validation_manager.get_results(),
            'feature_engineering_stats': feature_engineer.feature_engineering_stats,
            'category_metrics': category_results if config['analysis']['analyze_by_pilot_category'] else None,
            'turbulence_metrics': turbulence_results if config['analysis']['analyze_by_turbulence'] else None
        })
        
        logger.info(f"Analysis complete - {datetime.now()}")
        logger.info(f"Results saved in: {output_dir}")
        
    except Exception as e:
        logger.exception(f"Error in pipeline: {str(e)}")
        raise
    
    return 0


if __name__ == "__main__":
    exit(main())
