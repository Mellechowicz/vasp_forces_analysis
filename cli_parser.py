import argparse

class CLIParser:
    """Command-line interface parser for VASP AI Toolkit."""
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="VASP AI Toolkit")
        self._add_arguments()

    def _add_arguments(self):
        # Input/Output
        self.parser.add_argument('files', nargs='+', help='Input vasprun.xml files')

        # Functional Flags
        self.parser.add_argument('--plot', action='store_true', help='Enable visualization')
        self.parser.add_argument('--ml', action='store_true', help='Enable ML training')
        self.parser.add_argument('--generate', type=int, default=0, help='Generate N zero-force structures (requires --ml)')
        self.parser.add_argument('--decision', action='store_true', help='Run convergence decision logic')

        # Decision Thresholds
        self.parser.add_argument('--f_tol', type=float, default=0.02, help='Force tolerance (eV/A)')
        self.parser.add_argument('--p_tol', type=float, default=5.0, help='Pressure tolerance (kB)')

        # ML Model Hyperparameters
        self.parser.add_argument('--n_estimators', type=int, default=100, help='Number of trees in Random Forest')
        self.parser.add_argument('--max_depth', type=int, default=None, help='Max depth of trees (default: None)')
        self.parser.add_argument('--min_samples_split', type=int, default=2, help='Min samples required to split a node')

        # Structure Generation Tuning
        self.parser.add_argument('--learning_rate', type=float, default=0.1, help='Step size for structure relaxation')
        self.parser.add_argument('--steps', type=int, default=50, help='Max iterations for relaxation')
        self.parser.add_argument('--noise_level', type=float, default=None, 
                                 help='Perturbation noise level (default: 0.02 for Direct, 0.2 for Cartesian)')

    def parse(self):
        return self.parser.parse_args()

