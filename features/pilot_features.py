
import pandas as pd
import numpy as np
import logging

class PilotFeatures:
    """
    Creates features based on pilot-specific characteristics.
    """
    def __init__(self, id_column: str = 'pilot_id'):
        self.id_column = id_column
        self.logger = logging.getLogger(__name__)
    
    def add_pilot_group(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Categorize pilots if the column 'pilot_category' is missing.
        """
        df = data.copy()
        if 'pilot_category' not in df.columns:
            self.logger.info("Adding pilot_category based on pilot_id prefix")
            df['pilot_category'] = df[self.id_column].astype(str).apply(
                lambda pid: 'minimal_exp' if pid.startswith('8') 
                            else 'commercial' if pid.startswith('9')
                            else 'air_force'
            )
        return df
    
    def pilot_statistics(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Compute summary statistics per pilot.
        """
        df = data.copy()
        stats = df.groupby(self.id_column).agg(['mean', 'std', 'min', 'max'])
        self.logger.info("Calculated pilot-specific summary statistics")
        return stats
