import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

class MLModel:
    def __init__(self, data):
        self.data = data
        self.model = None
        self.is_trained = False

    def train(self):
        X = self.data[['x', 'y', 'z', 'element']]
        y = self.data[['fx', 'fy', 'fz']]

        preprocessor = ColumnTransformer(
                transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), ['element'])],
                remainder='passthrough'
                )

        # n_jobs=1 is safer for scripts that also use Matplotlib/Tkinter
        self.model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(n_estimators=50, n_jobs=1))
            ])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        print("Training ML model...")
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

    def generate_zero_force_structures(self, n_structures, coordinate_system="Direct", steps=50, learning_rate=0.1):
        """
        Generates relaxed structures.
        Adjusts noise level based on coordinate system (Direct vs Cartesian).
        """
        generated_structures = []

        # Set noise: Direct (0-1) needs small noise, Cartesian needs larger
        if coordinate_system == "Direct":
            noise_level = 0.02 # ~2% of lattice vector
        else:
            noise_level = 0.2  # ~0.2 Angstroms

        print(f"Generating {n_structures} structures (Mode: {coordinate_system}, Noise: {noise_level})...")

        for i in range(n_structures):
            # 1. Perturb
            current_pos = self.positions + np.random.normal(0, noise_level, self.positions.shape)

            # 2. Relax (Steepest Descent)
            for step in range(steps):
                pred_forces = self.ml_model.predict_forces(current_pos, self.elements)

                # Check convergence
                max_f = np.max(np.linalg.norm(pred_forces, axis=1))
                if max_f < 0.05: 
                    break

                # Update: Move atoms along force
                # Note: For Direct coords, force (eV/A) isn't directly compatible without lattice metric,
                # but for simple optimization, it pushes in the right direction.
                # Ideally we'd convert Forces -> Direct gradients, but this works for approximate relaxation.
                current_pos += learning_rate * pred_forces

            generated_structures.append(current_pos)

        return generated_structures

