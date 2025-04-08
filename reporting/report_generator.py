import os
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import shutil


class ReportGenerator:
    """
    Generates comprehensive reports for cognitive load prediction analysis
    with a focus on validation results and overfitting detection.
    """
    
    def __init__(self, output_dir: str = None):
        """
        Initialize the ReportGenerator.
        
        Args:
            output_dir: Directory to save the report
        """
        self.output_dir = output_dir
        
        # Create report directory
        if self.output_dir:
            self.report_dir = os.path.join(self.output_dir, 'report')
            os.makedirs(self.report_dir, exist_ok=True)
        else:
            self.report_dir = None
        
        # Logger
        self.logger = logging.getLogger(__name__)
    
    def generate_report(self, data_stats: Dict[str, Any] = None,
                     feature_engineering_stats: Dict[str, int] = None,
                     model_results: Dict[str, Dict[str, Any]] = None,
                     validation_results: Dict[str, Any] = None,
                     analysis_results: Dict[str, Any] = None) -> str:
        """
        Generate a comprehensive HTML report for the analysis.
        
        Args:
            data_stats: Dataset statistics
            feature_engineering_stats: Feature engineering statistics
            model_results: Model training results
            validation_results: Validation results
            analysis_results: Additional analysis results
            
        Returns:
            Path to the generated report
        """
        self.logger.info("Generating comprehensive report...")
        
        if not self.report_dir:
            self.logger.warning("No report directory specified, skipping report generation")
            return None
            
        # Copy all visualization files to report directory
        if self.output_dir:
            self._copy_visualizations()
        
        # Create HTML content
        html_content = self._generate_html_content(
            data_stats, 
            feature_engineering_stats, 
            model_results, 
            validation_results, 
            analysis_results
        )
        
        # Save HTML report
        report_path = os.path.join(self.report_dir, 'report.html')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
            
        self.logger.info(f"Report generated at: {report_path}")
        
        return report_path
    
    def _copy_visualizations(self) -> None:
        """Copy visualization files to report directory."""
        for file in os.listdir(self.output_dir):
            if file.endswith(('.png', '.jpg', '.svg')):
                src = os.path.join(self.output_dir, file)
                dst = os.path.join(self.report_dir, file)
                shutil.copy2(src, dst)
    
    def _generate_html_content(self, data_stats: Dict[str, Any],
                            feature_engineering_stats: Dict[str, int],
                            model_results: Dict[str, Dict[str, Any]],
                            validation_results: Dict[str, Any],
                            analysis_results: Dict[str, Any]) -> str:
        """
        Generate HTML content for the report.
        
        Args:
            data_stats: Dataset statistics
            feature_engineering_stats: Feature engineering statistics
            model_results: Model training results
            validation_results: Validation results
            analysis_results: Additional analysis results
            
        Returns:
            HTML content as a string
        """
        # Create HTML header
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Cognitive Load Prediction Analysis Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }}
        h1, h2, h3, h4 {{
            color: #2c3e50;
        }}
        h1 {{
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            margin-top: 30px;
            border-bottom: 1px solid #ddd;
            padding-bottom: 5px;
        }}
        .section {{
            margin: 30px 0;
            padding: 20px;
            background-color: #f9f9f9;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        .summary {{
            font-weight: bold;
            background-color: #e8f4f8;
            padding: 15px;
            border-left: 4px solid #3498db;
            margin: 20px 0;
        }}
        .warning {{
            background-color: #ffebee;
            padding: 15px;
            border-left: 4px solid #f44336;
            margin: 20px 0;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }}
        th {{
            background-color: #f2f2f2;
        }}
        tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        .img-container {{
            margin: 20px 0;
            text-align: center;
        }}
        .img-container img {{
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 5px;
        }}
        .caption {{
            font-style: italic;
            margin-top: 5px;
            color: #666;
        }}
        .footer {{
            margin-top: 50px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            color: #777;
            font-size: 0.9em;
            text-align: center;
        }}
    </style>
</head>
<body>
    <h1>Cognitive Load Prediction Analysis Report</h1>
    <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    
    <div class="section">
        <h2>Executive Summary</h2>
"""
        
        # Add executive summary based on available results
        if model_results and any('r2' in result for result in model_results.values()):
            # Calculate average performance across models
            r2_values = [result['r2'] for result in model_results.values() if 'r2' in result]
            avg_r2 = np.mean(r2_values)
            max_r2 = np.max(r2_values)
            
            # Get best model info
            best_model_key = max(model_results.keys(), key=lambda k: model_results[k].get('r2', 0))
            best_model = model_results[best_model_key]
            
            html += f"""
        <div class="summary">
            <p>Model Performance Summary:</p>
            <ul>
                <li>Best model: {best_model_key} (R² = {best_model['r2']:.4f}, RMSE = {best_model['rmse']:.4f})</li>
                <li>Average model performance: R² = {avg_r2:.4f}</li>
                <li>Number of models evaluated: {len(model_results)}</li>
            </ul>
        </div>
"""
            
            # Add overfitting risk assessment if available
            if validation_results and 'learning_curves' in validation_results:
                learning_curves = validation_results['learning_curves']
                if 'overfitting_analysis' in learning_curves:
                    overfitting = learning_curves['overfitting_analysis']
                    score = overfitting.get('overfitting_score', 0)
                    
                    if score > 0.5:
                        html += f"""
        <div class="warning">
            <p>⚠️ High Risk of Overfitting Detected (Score: {score:.2f})</p>
            <ul>
                <li>Large gap between train and test performance: {overfitting.get('max_gap', 0):.4f}</li>
                <li>Test performance slope: {overfitting.get('test_slope', 0):.4f}</li>
                <li>Recommended action: Reduce model complexity or increase regularization</li>
            </ul>
        </div>
"""
                    elif score > 0.2:
                        html += f"""
        <div class="warning" style="background-color: #fff9c4; border-left-color: #ffc107;">
            <p>⚠️ Moderate Risk of Overfitting Detected (Score: {score:.2f})</p>
            <ul>
                <li>Gap between train and test performance: {overfitting.get('max_gap', 0):.4f}</li>
                <li>Test performance slope: {overfitting.get('test_slope', 0):.4f}</li>
                <li>Recommended action: Consider additional cross-validation strategies</li>
            </ul>
        </div>
"""
                    else:
                        html += f"""
        <div class="summary" style="background-color: #e8f5e9; border-left-color: #4caf50;">
            <p>✅ Low Risk of Overfitting (Score: {score:.2f})</p>
            <p>The model appears to generalize well based on learning curve analysis.</p>
        </div>
"""
            
            # Add leave-one-pilot-out results if available
            if validation_results and 'lopo' in validation_results:
                lopo = validation_results['lopo']
                html += f"""
        <div class="summary">
            <p>Leave-One-Pilot-Out Validation Results:</p>
            <ul>
                <li>Mean R²: {lopo.get('mean_r2', 0):.4f} ± {lopo.get('std_r2', 0):.4f}</li>
                <li>Mean RMSE: {lopo.get('mean_rmse', 0):.4f} ± {lopo.get('std_rmse', 0):.4f}</li>
                <li>Pilots with negative R²: {len(lopo.get('negative_r2_pilots', []))}</li>
            </ul>
        </div>
"""
        
        html += """
    </div>
    
    <div class="section">
        <h2>Dataset Overview</h2>
"""
        
        # Add dataset statistics
        if data_stats:
            html += f"""
        <p>The dataset consists of {data_stats.get('n_trials', 'unknown')} trials from {data_stats.get('n_subjects', 'unknown')} pilots.</p>
        
        <h3>Target Variable Statistics</h3>
        <table>
            <tr>
                <th>Statistic</th>
                <th>Value</th>
            </tr>
"""
            
            if 'target_stats' in data_stats:
                for stat, value in data_stats['target_stats'].items():
                    if stat in ['mean', 'std', 'min', '25%', '50%', '75%', 'max']:
                        html += f"""
            <tr>
                <td>{stat}</td>
                <td>{value:.4f}</td>
            </tr>
"""
            
            html += """
        </table>
"""
            
            # Add pilot categories if available
            if 'pilot_categories' in data_stats:
                html += """
        <h3>Pilot Categories</h3>
        <table>
            <tr>
                <th>Category</th>
                <th>Count</th>
            </tr>
"""
                
                for category, count in data_stats['pilot_categories'].items():
                    html += f"""
            <tr>
                <td>{category}</td>
                <td>{count}</td>
            </tr>
"""
                
                html += """
        </table>
"""
            
            # Add turbulence levels if available
            if 'turbulence_levels' in data_stats:
                html += """
        <h3>Turbulence Levels</h3>
        <ul>
"""
                
                for level in data_stats['turbulence_levels']:
                    html += f"""
            <li>{level}</li>
"""
                
                html += """
        </ul>
"""
        
        # Add data distribution visualizations
        if os.path.exists(os.path.join(self.report_dir, 'target_distribution.png')):
            html += """
        <div class="img-container">
            <img src="target_distribution.png" alt="Target Distribution">
            <div class="caption">Distribution of Cognitive Load Scores</div>
        </div>
"""
        
        if os.path.exists(os.path.join(self.report_dir, 'turbulence_relationship.png')):
            html += """
        <div class="img-container">
            <img src="turbulence_relationship.png" alt="Turbulence Relationship">
            <div class="caption">Relationship between Turbulence Level and Cognitive Load</div>
        </div>
"""
        
        html += """
    </div>
    
    <div class="section">
        <h2>Feature Engineering</h2>
"""
        
        # Add feature engineering statistics
        if feature_engineering_stats:
            html += """
        <h3>Engineered Feature Categories</h3>
        <table>
            <tr>
                <th>Category</th>
                <th>Number of Features</th>
            </tr>
"""
            
            for category, count in feature_engineering_stats.items():
                html += f"""
            <tr>
                <td>{category.replace('_', ' ').title()}</td>
                <td>{count}</td>
            </tr>
"""
            
            html += """
        </table>
"""
        
        # Add feature importance visualization
        if os.path.exists(os.path.join(self.report_dir, 'feature_importance.png')):
            html += """
        <div class="img-container">
            <img src="feature_importance.png" alt="Feature Importance">
            <div class="caption">Feature Importance</div>
        </div>
"""
        
        # Add feature correlation visualization if available
        if os.path.exists(os.path.join(self.report_dir, 'feature_correlation_heatmap.png')):
            html += """
        <div class="img-container">
            <img src="feature_correlation_heatmap.png" alt="Feature Correlation">
            <div class="caption">Feature Correlation Heatmap</div>
        </div>
"""
        
        # Add feature target correlation visualization if available
        if os.path.exists(os.path.join(self.report_dir, 'feature_target_correlation.png')):
            html += """
        <div class="img-container">
            <img src="feature_target_correlation.png" alt="Feature-Target Correlation">
            <div class="caption">Feature Correlation with Target Variable</div>
        </div>
"""
        
        html += """
    </div>
    
    <div class="section">
        <h2>Model Performance</h2>
"""
        
        # Add model comparison visualization if available
        if os.path.exists(os.path.join(self.report_dir, 'model_comparison.png')):
            html += """
        <div class="img-container">
            <img src="model_comparison.png" alt="Model Comparison">
            <div class="caption">Model Performance Comparison</div>
        </div>
"""
        
        # Add model results table
        if model_results:
            html += """
        <h3>Model Performance Metrics</h3>
        <table>
            <tr>
                <th>Model</th>
                <th>R²</th>
                <th>RMSE</th>
                <th>MAE</th>
                <th>Train R²</th>
                <th>R² Gap</th>
            </tr>
"""
            
            # Sort models by R² score
            sorted_models = sorted(model_results.items(), key=lambda x: x[1].get('r2', 0), reverse=True)
            
            for model_key, result in sorted_models:
                r2 = result.get('r2', 0)
                rmse = result.get('rmse', 0)
                mae = result.get('mae', 0)
                train_r2 = result.get('train_r2', 0)
                r2_gap = result.get('r2_gap', 0)
                
                # Highlight potential overfitting
                gap_style = ""
                if r2_gap > 0.2:
                    gap_style = 'style="background-color: #ffebee;"'
                    
                html += f"""
            <tr>
                <td>{model_key}</td>
                <td>{r2:.4f}</td>
                <td>{rmse:.4f}</td>
                <td>{mae:.4f}</td>
                <td>{train_r2:.4f}</td>
                <td {gap_style}>{r2_gap:.4f}</td>
            </tr>
"""
            
            html += """
        </table>
"""
        
        # Add best model performance visualizations
        best_predictions_viz = None
        best_residuals_viz = None
        
        if model_results:
            best_model_key = max(model_results.keys(), key=lambda k: model_results[k].get('r2', 0))
            best_predictions_viz = f"actual_vs_pred_{best_model_key}.png"
            best_residuals_viz = f"residuals_{best_model_key}.png"
        
        if best_predictions_viz and os.path.exists(os.path.join(self.report_dir, best_predictions_viz)):
            html += f"""
        <div class="img-container">
            <img src="{best_predictions_viz}" alt="Best Model Predictions">
            <div class="caption">Actual vs. Predicted Values for Best Model</div>
        </div>
"""
        
        if best_residuals_viz and os.path.exists(os.path.join(self.report_dir, best_residuals_viz)):
            html += f"""
        <div class="img-container">
            <img src="{best_residuals_viz}" alt="Best Model Residuals">
            <div class="caption">Residual Analysis for Best Model</div>
        </div>
"""
        
        html += """
    </div>
    
    <div class="section">
        <h2>Validation Results</h2>
"""
        
        # Add learning curves if available
        if os.path.exists(os.path.join(self.report_dir, 'learning_curves.png')):
            html += """
        <h3>Learning Curves Analysis</h3>
        <div class="img-container">
            <img src="learning_curves.png" alt="Learning Curves">
            <div class="caption">Learning Curves (Training vs. Validation Performance)</div>
        </div>
"""
            
            # Add train vs test curve if available
            if os.path.exists(os.path.join(self.report_dir, 'train_vs_test_curve.png')):
                html += """
        <div class="img-container">
            <img src="train_vs_test_curve.png" alt="Train vs Test Curve">
            <div class="caption">Training vs. Test Performance Curve</div>
        </div>
"""
        
        # Add leave-one-pilot-out results if available
        if validation_results and 'lopo' in validation_results:
            html += """
        <h3>Leave-One-Pilot-Out Validation</h3>
"""
            
            if os.path.exists(os.path.join(self.report_dir, 'lopo_r2_by_pilot.png')):
                html += """
        <div class="img-container">
            <img src="lopo_r2_by_pilot.png" alt="LOPO R² by Pilot">
            <div class="caption">Leave-One-Pilot-Out R² Score by Pilot</div>
        </div>
"""
            
            if os.path.exists(os.path.join(self.report_dir, 'lopo_r2_by_category.png')):
                html += """
        <div class="img-container">
            <img src="lopo_r2_by_category.png" alt="LOPO R² by Category">
            <div class="caption">Leave-One-Pilot-Out R² Score by Pilot Category</div>
        </div>
"""
            
            lopo = validation_results['lopo']
            if 'negative_r2_pilots' in lopo and lopo['negative_r2_pilots']:
                html += """
        <div class="warning">
            <p>⚠️ Some pilots show poor generalization (negative R²):</p>
            <ul>
"""
                
                for pilot in lopo['negative_r2_pilots']:
                    html += f"""
                <li>Pilot {pilot}</li>
"""
                
                html += """
            </ul>
            <p>This indicates that the model may not generalize well to these pilots.</p>
        </div>
"""
        
        # Add permutation test results if available
        if validation_results and 'permutation' in validation_results:
            perm = validation_results['permutation']
            
            html += """
        <h3>Permutation Test Results</h3>
"""
            
            if os.path.exists(os.path.join(self.report_dir, 'permutation_test.png')):
                html += """
        <div class="img-container">
            <img src="permutation_test.png" alt="Permutation Test">
            <div class="caption">Permutation Test Results</div>
        </div>
"""
            
            if 'p_value_r2' in perm and 'p_value_rmse' in perm:
                html += f"""
        <p>The permutation test assesses whether model performance is significantly better than chance.</p>
        <ul>
            <li>R² p-value: {perm.get('p_value_r2', 0):.4f} ({'statistically significant' if perm.get('significant_r2', False) else 'not statistically significant'})</li>
            <li>RMSE p-value: {perm.get('p_value_rmse', 0):.4f} ({'statistically significant' if perm.get('significant_rmse', False) else 'not statistically significant'})</li>
        </ul>
"""
                
                if perm.get('significant_r2', False) and perm.get('significant_rmse', False):
                    html += """
        <div class="summary" style="background-color: #e8f5e9; border-left-color: #4caf50;">
            <p>✅ The model performs significantly better than chance.</p>
        </div>
"""
                elif not perm.get('significant_r2', False) or not perm.get('significant_rmse', False):
                    html += """
        <div class="warning">
            <p>⚠️ The model may not perform significantly better than chance on some metrics.</p>
            <p>This could indicate overfitting or a lack of predictive power.</p>
        </div>
"""
        
        html += """
    </div>
    
    <div class="section">
        <h2>Performance Analysis</h2>
"""
        
        # Add performance by category if available
        if analysis_results and 'category_metrics' in analysis_results:
            html += """
        <h3>Performance by Pilot Category</h3>
"""
            
            if os.path.exists(os.path.join(self.report_dir, 'performance_by_category.png')):
                html += """
        <div class="img-container">
            <img src="performance_by_category.png" alt="Performance by Category">
            <div class="caption">Model Performance by Pilot Category</div>
        </div>
"""
            
            if os.path.exists(os.path.join(self.report_dir, 'predictions_by_category.png')):
                html += """
        <div class="img-container">
            <img src="predictions_by_category.png" alt="Predictions by Category">
            <div class="caption">Predictions vs. Actual by Pilot Category</div>
        </div>
"""
            
            category_metrics = analysis_results['category_metrics']
            html += """
        <table>
            <tr>
                <th>Category</th>
                <th>Count</th>
                <th>R²</th>
                <th>RMSE</th>
                <th>MAE</th>
            </tr>
"""
            
            for category, metrics in category_metrics.items():
                html += f"""
            <tr>
                <td>{category}</td>
                <td>{metrics.get('count', 0)}</td>
                <td>{metrics.get('r2', 0):.4f}</td>
                <td>{metrics.get('rmse', 0):.4f}</td>
                <td>{metrics.get('mae', 0):.4f}</td>
            </tr>
"""
            
            html += """
        </table>
"""
        
        # Add performance by turbulence level if available
        if analysis_results and 'turbulence_metrics' in analysis_results:
            html += """
        <h3>Performance by Turbulence Level</h3>
"""
            
            if os.path.exists(os.path.join(self.report_dir, 'performance_by_turbulence.png')):
                html += """
        <div class="img-container">
            <img src="performance_by_turbulence.png" alt="Performance by Turbulence">
            <div class="caption">Model Performance by Turbulence Level</div>
        </div>
"""
            
            if os.path.exists(os.path.join(self.report_dir, 'predictions_by_turbulence.png')):
                html += """
        <div class="img-container">
            <img src="predictions_by_turbulence.png" alt="Predictions by Turbulence">
            <div class="caption">Predictions vs. Actual by Turbulence Level</div>
        </div>
"""
            
            turbulence_metrics = analysis_results['turbulence_metrics']
            html += """
        <table>
            <tr>
                <th>Turbulence Level</th>
                <th>Count</th>
                <th>R²</th>
                <th>RMSE</th>
                <th>MAE</th>
            </tr>
"""
            
            for level, metrics in sorted(turbulence_metrics.items()):
                html += f"""
            <tr>
                <td>{level}</td>
                <td>{metrics.get('count', 0)}</td>
                <td>{metrics.get('r2', 0):.4f}</td>
                <td>{metrics.get('rmse', 0):.4f}</td>
                <td>{metrics.get('mae', 0):.4f}</td>
            </tr>
"""
            
            html += """
        </table>
"""
        
        html += """
    </div>
    
    <div class="section">
        <h2>Conclusions and Recommendations</h2>
"""
        
        # Add conclusions based on results
        has_high_r2 = False
        has_overfitting = False
        has_category_differences = False
        has_turbulence_differences = False
        
        if model_results:
            best_model_key = max(model_results.keys(), key=lambda k: model_results[k].get('r2', 0))
            best_r2 = model_results[best_model_key].get('r2', 0)
            has_high_r2 = best_r2 > 0.7
            
            # Check for overfitting
            if 'r2_gap' in model_results[best_model_key]:
                r2_gap = model_results[best_model_key]['r2_gap']
                has_overfitting = r2_gap > 0.2
        
        # Check for differences between categories
        if analysis_results and 'category_metrics' in analysis_results:
            category_metrics = analysis_results['category_metrics']
            r2_values = [metrics.get('r2', 0) for metrics in category_metrics.values()]
            if len(r2_values) > 1:
                has_category_differences = max(r2_values) - min(r2_values) > 0.1
        
        # Check for differences between turbulence levels
        if analysis_results and 'turbulence_metrics' in analysis_results:
            turbulence_metrics = analysis_results['turbulence_metrics']
            r2_values = [metrics.get('r2', 0) for metrics in turbulence_metrics.values()]
            if len(r2_values) > 1:
                has_turbulence_differences = max(r2_values) - min(r2_values) > 0.1
        
        # Generate conclusions
        if has_high_r2:
            html += """
        <div class="summary" style="background-color: #e8f5e9; border-left-color: #4caf50;">
            <p>✅ The model demonstrates strong predictive performance for cognitive load.</p>
        </div>
"""
        else:
            html += """
        <div class="warning" style="background-color: #fff9c4; border-left-color: #ffc107;">
            <p>⚠️ The model shows moderate predictive performance and may benefit from further refinement.</p>
        </div>
"""
        
        if has_overfitting:
            html += """
        <div class="warning">
            <p>⚠️ Overfitting Detected: The model performs significantly better on training data than on testing data.</p>
            <p>Recommended actions:</p>
            <ul>
                <li>Reduce model complexity</li>
                <li>Increase regularization</li>
                <li>Collect more diverse training data</li>
                <li>Use simpler feature sets</li>
            </ul>
        </div>
"""
        
        if has_category_differences:
            html += """
        <div class="warning" style="background-color: #fff9c4; border-left-color: #ffc107;">
            <p>⚠️ Performance varies significantly across pilot categories.</p>
            <p>Consider:</p>
            <ul>
                <li>Developing separate models for different pilot categories</li>
                <li>Adding more features related to pilot experience</li>
                <li>Collecting more data for underrepresented categories</li>
            </ul>
        </div>
"""
        
        if has_turbulence_differences:
            html += """
        <div class="warning" style="background-color: #fff9c4; border-left-color: #ffc107;">
            <p>⚠️ Performance varies across turbulence levels.</p>
            <p>Consider:</p>
            <ul>
                <li>Creating specialized models for different turbulence levels</li>
                <li>Engineering additional features that capture turbulence-specific responses</li>
                <li>Balancing the dataset across turbulence levels</li>
            </ul>
        </div>
"""
        
        # Add general recommendations
        html += """
        <h3>General Recommendations</h3>
        <ul>
            <li>Continue validation with new pilots to ensure generalizability</li>
            <li>Consider incorporating temporal dynamics in future models (windowed features)</li>
            <li>Explore more sophisticated ensemble techniques to improve performance</li>
            <li>Investigate additional physiological signals and their interactions</li>
            <li>Develop a real-time monitoring system based on the validated model</li>
        </ul>
"""
        
        html += """
    </div>
    
    <div class="footer">
        <p>Generated by Cognitive Load Analysis Pipeline v9</p>
        <p>© {}</p>
    </div>
</body>
</html>
""".format(datetime.now().year)
        
        return html
