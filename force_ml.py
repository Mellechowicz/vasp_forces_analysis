# force_ml.py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

class ForcePredictor:
    def __init__(self, file_list):
        """
        Initializes the ML class.
        file_list: List of paths to vasprun.xml files.
        """
        self.file_list = file_list
        self.model = None
        self.data = None

    def load_and_prepare_data(self):
        from vasp_parser import VaspParser

        all_data = []
        print("Loading data for ML...")
        for f in self.file_list:
            try:
                parser = VaspParser(f)
                df = parser.extract_calculations()
                all_data.append(df)
            except Exception as e:
                print(f"Skipping {f} due to error: {e}")

        if not all_data:
            raise ValueError("No data loaded for ML.")

        self.data = pd.concat(all_data, ignore_index=True)
        print(f"Loaded {len(self.data)} samples.")

    def train(self):
        if self.data is None:
            self.load_and_prepare_data()

        # Features: Position (x,y,z) + Element Type
        X = self.data[['x', 'y', 'z', 'element']]
        # Targets: Forces (fx, fy, fz)
        y = self.data[['fx', 'fy', 'fz']]

        # Preprocessing: OneHotEncode the 'element' column
        preprocessor = ColumnTransformer(
                transformers=[
                    ('cat', OneHotEncoder(), ['element'])
                    ],
                remainder='passthrough' # Keep x, y, z as numerical
                )

        # Pipeline: Preprocess -> Random Forest
        self.model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
            ])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        print("Training Random Forest Regressor...")
        self.model.fit(X_train, y_train)

        predictions = self.model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, predictions)

        print(f"Model Evaluation:\nRMSE: {rmse:.4f}\nR^2 Score: {r2:.4f}")

        return X_test, y_test, predictions

    def predict(self, positions, elements):
        """
        Predict forces for new inputs.
        positions: list of [x, y, z] lists
        elements: list of element strings corresponding to positions
        """
        if self.model is None:
            raise Exception("Model not trained yet.")

        input_df = pd.DataFrame(positions, columns=['x', 'y', 'z'])
        input_df['element'] = elements
        return self.model.predict(input_df)
