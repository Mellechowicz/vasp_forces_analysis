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
        """Checks if the maximum force at the last step is below a specified threshold."""
        last_step = self.df['step'].max()
        final_forces = self.df[self.df['step'] == last_step]['magnitude']
        max_force = final_forces.max()
        return {
                "converged": max_force < force_thresh,
                "max_force": max_force
                }

    def energy_stats(self):
        """Energy is also one value per step, so we analyze it similarly to pressure."""
        if 'energy' in self.df.columns:
            # Drop duplicates because energy is constant per step (same for all atoms in that step)
            step_data = self.df.drop_duplicates(subset=['step', 'file_source'])
            return step_data['energy'].describe()
        return "No Energy data found."

    def drift_stats(self):
        """Calculates Total Drift (Sum of forces on all atoms) per step."""
        # Sum forces for each step
        drift_df = self.df.groupby(['step', 'file_source'])[['fx', 'fy', 'fz']].sum()
        # Calculate magnitude of the drift vector
        drift_mags = np.linalg.norm(drift_df.values, axis=1)
        return pd.Series(drift_mags, name="Drift Magnitude").describe()

    def stress_stats(self):
        """Extended stress statistics."""
        if 'stress_xx' in self.df.columns:
            step_data = self.df.drop_duplicates(subset=['step', 'file_source'])
            return step_data[['stress_xx', 'stress_yy', 'stress_zz']].describe()
        return "No Stress data found."

