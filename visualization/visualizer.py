
# visualization/visualizer.py
import os
import logging
import matplotlib.pyplot as plt
from .error_analysis import ErrorAnalysisVisualizer
from .feature_plots import FeatureVisualizer

# If you have a separate performance_plots module, import it here.
# from .performance_plots import PerformancePlotsVisualizer

class Visualizer:
    """
    Aggregates visualization functions for the cognitive load analysis pipeline.
    This includes error/residual analysis, feature importance and correlation plots,
    and overall performance visualizations.
    """
    def __init__(self, output_dir=None):
        """
        Initialize the Visualizer by setting up sub-visualizers.
        
        Args:
            output_dir (str): Directory where visualizations will be saved.
        """
        self.output_dir = output_dir
        # Ensure the output directory exists.
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        self.error_vis = ErrorAnalysisVisualizer(output_dir)
        self.feature_vis = FeatureVisualizer(output_dir)
        # Uncomment if you implement a PerformancePlotsVisualizer.
        # self.performance_vis = PerformancePlotsVisualizer(output_dir)

    def generate_visualizations(self, data, train_data, test_data, model_results,
                                validation_results, target_column, config):
        """
        Generate various visualizations based on configuration.
        
        Args:
            data (pd.DataFrame): The complete preprocessed dataset.
            train_data (pd.DataFrame): The train dataset (engineered).
            test_data (pd.DataFrame): The test dataset (engineered).
            model_results (dict): Results from model training.
            validation_results (dict): Results from validation routines.
            target_column (str): Name of the target column.
            config (dict): Visualization configuration.
        """
        # Plot feature importance (if available in model_results)
        if config.get("plot_feature_importance", False):
            # Attempt to generate the feature importance plot using FeatureVisualizer.
            # Assuming model_results holds a key 'feature_importance' as a DataFrame.
            if "feature_importance" in model_results:
                self.feature_vis.plot_feature_importance(model_results["feature_importance"],
                                                         title="Model Feature Importance")
            else:
                self.logger.warning("No feature importance data available for plotting.")

        # Generate error distribution plots for the best model if available.
        if config.get("plot_error_distribution", False) and model_results:
            # Assume model_results has a 'best_model' key with an associated 'residuals' array.
            best_model_key = model_results.get("best_model", {}).get("key")
            if best_model_key and "residuals" in model_results.get(best_model_key, {}):
                residuals = model_results[best_model_key]["residuals"]
                self.error_vis.plot_residual_distribution(residuals, best_model_key)
                self.error_vis.plot_residual_qq(residuals, best_model_key)
            else:
                self.logger.warning("No residual data available for error analysis plots.")

        # Additional plots can be added here.
        # For instance, if you have performance plots:
        # if config.get("plot_model_comparison", False):
        #     self.performance_vis.plot_model_comparison(model_results)

        # Save any common plots (if needed). This is a placeholder.
        self.logger.info("Visualizations generated.")

    def final_summary(self, summary_data):
        """
        Generate a final summary visualization.
        
        Args:
            summary_data (dict): Aggregated metrics and summaries to visualize.
        """
        # Example: Create a bar chart summarizing key metrics of the best model.
        best_model = summary_data.get("best_model", {})
        if not best_model:
            self.logger.warning("No best model data provided for summary visualization.")
            return

        labels = ["R²", "RMSE", "MAE", "Train R²", "R² Gap"]
        values = [
            best_model.get("r2", 0),
            best_model.get("rmse", 0),
            best_model.get("mae", 0),
            best_model.get("train_r2", 0),
            best_model.get("r2_gap", 0)
        ]

        plt.figure(figsize=(8, 6))
        bars = plt.bar(labels, values, color=plt.cm.viridis(range(len(labels))))
        plt.title("Best Model Performance Summary")
        plt.ylabel("Value")
        # Add value labels on the bars.
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, height, f'{height:.4f}',
                     ha='center', va='bottom')
        plt.tight_layout()

        if self.output_dir:
            summary_path = os.path.join(self.output_dir, "final_summary.png")
            plt.savefig(summary_path, dpi=300)
            self.logger.info(f"Final summary visualization saved to {summary_path}")
        plt.close()
