# visualizer.py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class ForceVisualizer:
    def __init__(self, dataframe):
        self.df = dataframe
        sns.set_theme(style="whitegrid")

    def plot_trajectory_stats(self):
        """Plots mean and max force evolution over simulation steps."""
        stats = self.df.groupby('step')['magnitude'].agg(['mean', 'max'])

        plt.figure(figsize=(10, 5))
        plt.plot(stats.index, stats['mean'], label='Mean Force', marker='o')
        plt.plot(stats.index, stats['max'], label='Max Force', linestyle='--', color='red')
        plt.xlabel('Ionic Step')
        plt.ylabel('Force Magnitude (eV/A)')
        plt.title('Force Trajectory Analysis')
        plt.legend()
        plt.show()

    def plot_force_distribution(self):
        """Plots histogram and KDE of force magnitudes."""
        plt.figure(figsize=(8, 6))
        sns.histplot(data=self.df, x="magnitude", hue="element", kde=True, element="step")
        plt.title("Distribution of Force Magnitudes by Element")
        plt.xlabel("Force (eV/A)")
        plt.show()

    def plot_3d_scatter(self, step_idx=None):
        """
        3D scatter plot of atom positions colored by force magnitude.
        Defaults to the last step.
        """
        if step_idx is None:
            step_idx = self.df['step'].max()

        step_data = self.df[self.df['step'] == step_idx]

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        sc = ax.scatter(
                step_data['x'], step_data['y'], step_data['z'],
                c=step_data['magnitude'], cmap='viridis', s=100, alpha=0.8
                )

        ax.set_title(f'Atom Positions & Force Magnitude (Step {step_idx})')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.colorbar(sc, label='Force Magnitude (eV/A)')
        plt.show()

    def plot_component_correlation(self):
        """Pair plot to inspect correlations between force components."""
        sns.pairplot(self.df[['fx', 'fy', 'fz', 'element']], hue='element')
        plt.suptitle("Force Component Correlation", y=1.02)
        plt.show()
