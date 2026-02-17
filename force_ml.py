import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

class MLModel:
    def __init__(self, data, n_estimators=100, max_depth=None, min_samples_split=2):
        self.data = data
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.model = None
        self.is_trained = False

    def train(self):
        X = self.data[['x', 'y', 'z', 'element']]
        y = self.data[['fx', 'fy', 'fz']]

        preprocessor = ColumnTransformer(
                transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), ['element'])],
                remainder='passthrough'
                )

        # n_jobs=1 prevents Tkinter/Multiprocessing conflict
        self.model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(
                n_estimators=self.n_estimators, 
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                n_jobs=1
                ))
            ])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        print(f"Training ML model (Trees: {self.n_estimators}, Depth: {self.max_depth})...")
        self.model.fit(X_train, y_train)
        self.is_trained = True
        print(f"Model Accuracy (R2): {self.model.score(X_test, y_test):.4f}")

    def predict_forces(self, positions, elements):
        if not self.is_trained:
            raise Exception("Model not trained.")

        df = pd.DataFrame(positions, columns=['x', 'y', 'z'])
        df['element'] = elements
        return self.model.predict(df)

class StructureGenerator:
    def __init__(self, ml_model, template_positions, template_elements, lattice):
        self.ml_model = ml_model
        self.positions = np.array(template_positions)
        self.elements = template_elements
        self.lattice = lattice

    def generate_zero_force_structures(self, n_structures, coordinate_system="Direct", steps=50, learning_rate=0.1, noise_level=None):
        """
        Generates relaxed structures.
        """
        generated_structures = []

        # Determine noise level: Use User input if provided, else use Heuristic
        if noise_level is not None:
            current_noise = noise_level
        else:
            # Heuristic: Direct (0-1) needs small noise, Cartesian needs larger
            current_noise = 0.02 if coordinate_system == "Direct" else 0.2

        print(f"Generating {n_structures} structures (Mode: {coordinate_system}, Noise: {current_noise}, LR: {learning_rate})...")

        for i in range(n_structures):
            # 1. Perturb
            current_pos = self.positions + np.random.normal(0, current_noise, self.positions.shape)

            # 2. Relax (Steepest Descent)
            for step in range(steps):
                pred_forces = self.ml_model.predict_forces(current_pos, self.elements)

                # Check convergence
                max_f = np.max(np.linalg.norm(pred_forces, axis=1))
                if max_f < 0.05: 
                    break

                current_pos += learning_rate * pred_forces

            generated_structures.append(current_pos)

        return generated_structures

