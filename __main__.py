# main.py
from vasp_parser import VaspParser
from force_analysis import ForceAnalyzer
from visualizer import ForceVisualizer
from force_ml import ForcePredictor
import matplotlib.pyplot as plt

def main():
    # 1. Parsing
    filename = 'vasprun.xml' 
    print(f"--- Parsing {filename} ---")
    try:
        parser = VaspParser(filename)
        df = parser.extract_calculations()
        print(f"Extracted {len(df)} atom-step entries.")
    except Exception as e:
        print(f"Error parsing file: {e}")
        return

    if df.empty:
        print("No force/position data found.")
        return

    # 2. Analysis
    print("\n--- Statistical & Logical Analysis ---")
    analyzer = ForceAnalyzer(df)

    print("Basic Stats:")
    print(analyzer.basic_stats())

    print("\nConvergence Check:")
    print(analyzer.check_convergence())

    print("\nNormality Test (Shapiro-Wilk):")
    print(analyzer.component_normality_test())

    print("\nStats by Atom Type:")
    print(analyzer.atom_type_analysis())

    # 3. Visualization
    print("\n--- Generating Visualizations ---")
    viz = ForceVisualizer(analyzer.df) # Use dataframe with magnitude column

    viz.plot_trajectory_stats()
    viz.plot_force_distribution()
    viz.plot_3d_scatter()
    # viz.plot_component_correlation() # Uncomment for larger matrix plot

    # 4. Machine Learning
    print("\n--- Machine Learning Pipeline ---")
    # Using the same file as a 'list' for demonstration
    ml = ForcePredictor([filename]) 

    X_test, y_test, preds = ml.train()

    # Visualize ML Results (Actual vs Predicted Fx)
    plt.figure(figsize=(6,6))
    plt.scatter(y_test.iloc[:, 0], preds[:, 0], alpha=0.5)
    plt.plot([y_test.iloc[:, 0].min(), y_test.iloc[:, 0].max()], 
             [y_test.iloc[:, 0].min(), y_test.iloc[:, 0].max()], 'r--')
    plt.xlabel("Actual Fx")
    plt.ylabel("Predicted Fx")
    plt.title("ML Prediction Accuracy: Fx")
    plt.show()

if __name__ == "__main__":
    main()
