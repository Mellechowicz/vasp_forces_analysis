# force_analysis.py
import numpy as np
import scipy.stats as stats
import pandas as pd

class ForceAnalyzer:
    def __init__(self, dataframe):
        self.df = dataframe
        # Calculate Magnitude
        self.df['magnitude'] = np.linalg.norm(self.df[['fx', 'fy', 'fz']].values, axis=1)

    def basic_stats(self):
        """Returns basic description of force magnitudes."""
        return self.df['magnitude'].describe()

    def check_convergence(self, threshold=0.01):
        """
        Logical analysis: Checks if the maximum force in the final step 
        is below a convergence threshold.
        """
        last_step = self.df['step'].max()
        final_forces = self.df[self.df['step'] == last_step]['magnitude']
        max_force = final_forces.max()
        converged = max_force < threshold
        return {
                "converged": converged,
                "max_force_final_step": max_force,
                "threshold": threshold,
                "atoms_above_threshold": final_forces[final_forces > threshold].count()
                }

    def component_normality_test(self):
        """
        Statistical analysis: Performs Shapiro-Wilk test to see if 
        force components are normally distributed.
        """
        results = {}
        for comp in ['fx', 'fy', 'fz']:
            # Sampling simple if dataset is too large (>5000), otherwise Shapiro is valid
            data = self.df[comp].sample(min(len(self.df), 4000))
            stat, p = stats.shapiro(data)
            results[comp] = {'statistic': stat, 'p_value': p, 'is_normal': p > 0.05}
        return results

    def atom_type_analysis(self):
        """Grouped statistics by atom element."""
        return self.df.groupby('element')['magnitude'].agg(['mean', 'max', 'std'])
