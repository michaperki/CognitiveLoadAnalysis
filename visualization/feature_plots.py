
import os
import matplotlib.pyplot as plt
import seaborn as sns
import logging

class FeatureVisualizer:
    """
    Visualization routines for features.
    """
    def __init__(self, output_dir: str = None):
        self.output_dir = output_dir
        self.logger = logging.getLogger(__name__)
    
    def plot_feature_importance(self, importance_df, title="Feature Importance") -> None:
        plt.figure(figsize=(12, 8))
        importance_df = importance_df.sort_values("importance", ascending=False).head(20)
        plt.barh(importance_df["feature"], importance_df["importance"], color="skyblue")
        plt.xlabel("Importance")
        plt.title(title)
        plt.tight_layout()
        if self.output_dir:
            path = os.path.join(self.output_dir, "feature_importance.png")
            plt.savefig(path, dpi=300)
            self.logger.info(f"Saved feature importance plot to {path}")
        plt.close()
    
    def plot_feature_correlation(self, data, features) -> None:
        plt.figure(figsize=(16, 14))
        corr = data[features].corr()
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
        plt.title("Feature Correlation Heatmap")
        plt.tight_layout()
        if self.output_dir:
            path = os.path.join(self.output_dir, "feature_correlation_heatmap.png")
            plt.savefig(path, dpi=300)
            self.logger.info(f"Saved feature correlation heatmap to {path}")
        plt.close()
