import sys
import pandas as pd
import numpy as np

# Import custom modules
from cli_parser import CLIParser
from vasp_parser import VaspParser
from force_analysis import Analyzer
from visualizer import Visualizer
from force_ml import MLModel, StructureGenerator
from poscar_io import PoscarWriter
from decision_module import RelaxationDecision

def main():
    # 1. Parse Arguments
    cli = CLIParser()
    args = cli.parse()

    # 2. Load Data
    all_data = []

    # Metadata placeholders
    last_lattice = None
    last_elements = None
    last_positions = None
    last_counts = None
    last_unique_elements = None
    detected_coord_type = "Direct"

    print("--- Parsing Files ---")
    for f in args.files:
        try:
            vp = VaspParser(f)
            df = vp.extract_data()
            if not df.empty:
                all_data.append(df)

                # Update metadata (always from the latest file processed)
                last_lattice = vp.extract_basis()
                last_step_idx = df['step'].max()
                last_step_df = df[df['step'] == last_step_idx]

                last_elements = last_step_df['element'].tolist()
                last_positions = last_step_df[['x','y','z']].values
                last_unique_elements = vp.atom_types
                last_counts = vp.counts
                detected_coord_type = vp.coordinate_type

                print(f"Loaded {f}: {len(df)} entries (Coords: {vp.coordinate_type})")
        except Exception as e:
            print(f"Error reading {f}: {e}")

    if not all_data:
        print("No valid data found.")
        sys.exit(1)

    combined_df = pd.concat(all_data, ignore_index=True)
    analyzer = Analyzer(combined_df)

    # 3. Decision Module (Optional)
    # Checks convergence and exits if requested
    if args.decision:
        print("\n--- Convergence Decision ---")
        decider = RelaxationDecision(analyzer, force_thresh=args.f_tol, pressure_thresh=args.p_tol)
        exit_code, suggested_isif = decider.evaluate()

        # If user only wanted a decision check, we might want to exit here
        # But usually we continue to ML/Plotting unless it's a pipeline script
        # For this logic, we print suggestions. If strictly used in bash pipeline:
        if exit_code == 0:
            print("Structure Converged. Exiting 0.")
            sys.exit(0)
        else:
            print(f"Structure NOT Converged. Next ISIF: {suggested_isif}")
            # We don't force exit 1 here to allow viewing plots/ML, 
            # but in a bash script, you'd check stdout.

    # 4. Analysis Output
    print("\n--- Statistics ---")
    print(analyzer.force_stats())
    if 'pressure' in combined_df.columns:
        print(analyzer.pressure_stats())

    # 5. Machine Learning & Generation
    if args.ml or args.generate > 0:
        print("\n--- Machine Learning ---")
        ml = MLModel(combined_df)
        ml.train()

        if args.generate > 0:
            if last_positions is None:
                print("Error: No template structure found.")
            else:
                gen = StructureGenerator(ml, last_positions, last_elements, last_lattice)
                new_structs = gen.generate_zero_force_structures(
                        args.generate, 
                        coordinate_system=detected_coord_type
                        )

                for idx, pos in enumerate(new_structs):
                    fname = f"POSCAR_generated_{idx}.vasp"
                    PoscarWriter.write(
                            fname, 
                            last_lattice, 
                            pos, 
                            last_unique_elements, 
                            last_counts, 
                            coordinate_system=detected_coord_type,
                            title=f"ML_Gen_{idx}_ISIF_{suggested_isif if args.decision and suggested_isif else 'Opt'}"
                            )

    # 6. Visualization (Must be last to avoid blocking)
    if args.plot:
        print("\n--- Visualizing ---")
        viz = Visualizer(combined_df)
        viz.plot_trajectory()
        viz.plot_stress_distribution()

if __name__ == "__main__":
    main()
