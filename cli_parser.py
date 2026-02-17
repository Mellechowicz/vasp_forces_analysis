import argparse

class CLIParser:
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

        # Thresholds
        self.parser.add_argument('--f_tol', type=float, default=0.02, help='Force tolerance (eV/A)')
        self.parser.add_argument('--p_tol', type=float, default=5.0, help='Pressure tolerance (kB)')

    def parse(self):
        return self.parser.parse_args()

