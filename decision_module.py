import sys
import numpy as np

class RelaxationDecision:
    def __init__(self, analyzer, force_thresh=0.02, pressure_thresh=5.0):
        """
        analyzer: Instance of force_analysis.Analyzer
        force_thresh: Max force convergence criterion (eV/A)
        pressure_thresh: Max pressure convergence criterion (kB)
        """
        self.ana = analyzer
        self.f_thresh = force_thresh
        self.p_thresh = pressure_thresh

    def evaluate(self):
        """
        Evaluates convergence and returns a status code and suggested ISIF.
        """
        # Get last step data
        last_step_idx = self.ana.df['step'].max()
        last_step_df = self.ana.df[self.ana.df['step'] == last_step_idx]

        # Force Check
        max_force = last_step_df['magnitude'].max()
        forces_converged = max_force < self.f_thresh

        # Pressure/Stress Check
        pressure_converged = True
        current_pressure = 0.0

        if 'pressure' in last_step_df.columns:
            current_pressure = abs(last_step_df['pressure'].iloc[0])
            pressure_converged = current_pressure < self.p_thresh

        print(f"Decision Check -> Max Force: {max_force:.4f}/{self.f_thresh}, Pressure: {current_pressure:.2f}/{self.p_thresh}")

        if forces_converged and pressure_converged:
            print("CONVERGED: Structure is relaxed.")
            return 0, None # Exit code 0, No ISIF needed

        # Logic for next step if not converged
        # If forces are huge, stick to ion relaxation (ISIF=2)
        # If forces are okay but pressure is high, relax cell (ISIF=3)

        suggested_isif = 3 # Default: Full relaxation

        if not forces_converged and max_force > 0.5:
            # Forces very high, dangerous to change cell shape
            suggested_isif = 2
        elif forces_converged and not pressure_converged:
            # Forces good, only cell needs adjusting
            suggested_isif = 3 

        print(f"NOT CONVERGED. Suggested next run: ISIF = {suggested_isif}")
        return 1, suggested_isif

