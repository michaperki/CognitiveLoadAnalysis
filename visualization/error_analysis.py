
import os
import matplotlib.pyplot as plt
import seaborn as sns
import logging

class ErrorAnalysisVisualizer:
    """
    Visualization tools for error and residual analysis.
    """
    def __init__(self, output_dir: str = None):
        self.output_dir = output_dir
        self.logger = logging.getLogger(__name__)
    
    def plot_residual_distribution(self, residuals, model_key: str) -> None:
        plt.figure(figsize=(10, 6))
        sns.histplot(residuals, kde=True, color='steelblue', bins=30)
        plt.axvline(0, color='red', linestyle='--', linewidth=2)
        plt.title(f"Residual Distribution for {model_key}")
        plt.xlabel("Residuals")
        plt.ylabel("Frequency")
        plt.tight_layout()
        if self.output_dir:
            save_path = os.path.join(self.output_dir, f"error_distribution_{model_key}.png")
            plt.savefig(save_path, dpi=300)
            self.logger.info(f"Saved error distribution plot to {save_path}")
        plt.close()
    
    def plot_residual_qq(self, residuals, model_key: str) -> None:
        import scipy.stats as stats
        plt.figure(figsize=(10, 6))
        stats.probplot(residuals, dist="norm", plot=plt)
        plt.title(f"Residual Q-Q Plot for {model_key}")
        plt.tight_layout()
        if self.output_dir:
            save_path = os.path.join(self.output_dir, f"residuals_qq_{model_key}.png")
            plt.savefig(save_path, dpi=300)
            self.logger.info(f"Saved residual Q-Q plot to {save_path}")
        plt.close()
