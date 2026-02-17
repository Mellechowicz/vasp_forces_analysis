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

