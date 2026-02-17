import numpy as np
import scipy.stats as stats
import pandas as pd

class Analyzer:
    def __init__(self, dataframe):
        self.df = dataframe
        # Calculate Magnitude of Forces
        self.df['magnitude'] = np.linalg.norm(self.df[['fx', 'fy', 'fz']].values, axis=1)

    def force_stats(self):
        return self.df['magnitude'].describe()

    def pressure_stats(self):
        if 'pressure' in self.df.columns:
            # Pressure is one value per step, so we drop duplicates to analyze trajectory
            step_data = self.df.drop_duplicates(subset=['step', 'file_source'])
            return step_data['pressure'].describe()
        return "No Pressure data found."

    def check_convergence(self, force_thresh=0.01):
        last_step = self.df['step'].max()
        final_forces = self.df[self.df['step'] == last_step]['magnitude']
        max_force = final_forces.max()
        return {
                "converged": max_force < force_thresh,
                "max_force": max_force
                }

