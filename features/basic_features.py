
import numpy as np
import pandas as pd
import logging

class BasicFeatures:
    """
    Creates basic core features for the cognitive load analysis.
    """
    def __init__(self, id_column: str = 'pilot_id', target_column: str = 'avg_tlx_quantile'):
        self.id_column = id_column
        self.target_column = target_column
        self.logger = logging.getLogger(__name__)
    
    def create_basic_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add basic normalized and aggregated features.
        """
        df = data.copy()
        self.logger.info("Creating basic features...")

        # Example: simple z-score normalization for a few known signals if they exist
        for col in ['scr_mean', 'hr_mean']:
            if col in df.columns:
                df[f"{col}_z"] = (df[col] - df[col].mean()) / (df[col].std() or 1)
                self.logger.info(f"Created z-score for {col}")
        
        # Add an example ratio feature: if both mean and std exist for a signal
        if 'scr_mean' in df.columns and 'scr_std' in df.columns:
            df["scr_cv"] = df["scr_std"] / (df["scr_mean"] + 1e-6)
            self.logger.info("Created coefficient of variation for scr signals")
        
        return df
