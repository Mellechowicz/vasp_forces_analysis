import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class Visualizer:
    def __init__(self, dataframe):
        self.df = dataframe
        sns.set_theme(style="whitegrid")

    def plot_trajectory(self):
        """Plots Mean Force and Pressure evolution."""
        # Aggregate by step
        stats = self.df.groupby('step').agg({
            'magnitude': 'mean',
            'pressure': 'first' # Pressure is constant per step
            })

        fig, ax1 = plt.subplots(figsize=(10, 5))

        color = 'tab:blue'
        ax1.set_xlabel('Ionic Step')
        ax1.set_ylabel('Mean Force (eV/A)', color=color)
        ax1.plot(stats.index, stats['magnitude'], color=color, marker='o', markersize=4)
        ax1.tick_params(axis='y', labelcolor=color)

        if 'pressure' in stats.columns:
            ax2 = ax1.twinx()
            color = 'tab:red'
            ax2.set_ylabel('Pressure (kB)', color=color)
            ax2.plot(stats.index, stats['pressure'], color=color, linestyle='--')
            ax2.tick_params(axis='y', labelcolor=color)

        plt.title('Force and Pressure Trajectory')
        plt.tight_layout()
        plt.show()

    def plot_stress_distribution(self):
        if 'stress_xx' not in self.df.columns: return

        # Unique steps only
        step_df = self.df.drop_duplicates(subset=['step', 'file_source'])

        plt.figure(figsize=(8, 6))
        sns.kdeplot(data=step_df[['stress_xx', 'stress_yy', 'stress_zz']], fill=True)
        plt.title("Distribution of Diagonal Stress Components")
        plt.xlabel("Stress (kB)")
        plt.show()


    def plot_extended_diagnostics(self):
        """Plots Energy, Drift, and Stress components."""

        # Aggregate data per step
        step_df = self.df.drop_duplicates(subset=['step', 'file_source']).sort_values('step')

        # Calculate Drift per step
        drift_df = self.df.groupby('step')[['fx', 'fy', 'fz']].sum()
        drift_mags = np.linalg.norm(drift_df.values, axis=1)

        fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

        # 1. Energy Plot
        if 'energy' in step_df.columns:
            axes[0].plot(step_df['step'], step_df['energy'], color='tab:green', marker='.')
            axes[0].set_ylabel('Total Energy (eV)')
            axes[0].set_title('Energy Evolution')
            axes[0].grid(True)

        # 2. Drift Plot
        axes[1].plot(drift_df.index, drift_mags, color='tab:orange', marker='.')
        axes[1].set_ylabel('Total Drift (eV/A)')
        axes[1].set_title('Force Drift (Sum of Forces)')
        axes[1].grid(True)

        # 3. Stress Components Plot
        if 'stress_xx' in step_df.columns:
            axes[2].plot(step_df['step'], step_df['stress_xx'], label='XX', linestyle='--')
            axes[2].plot(step_df['step'], step_df['stress_yy'], label='YY', linestyle='--')
            axes[2].plot(step_df['step'], step_df['stress_zz'], label='ZZ', linestyle='--')
            axes[2].set_ylabel('Stress (kB)')
            axes[2].set_xlabel('Ionic Step')
            axes[2].set_title('Stress Tensor Components')
            axes[2].legend()
            axes[2].grid(True)

        plt.tight_layout()
        plt.show()

