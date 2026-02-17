import argparse
import sys
import pandas as pd
import numpy as np
from vasp_parser import VaspParser
from force_analysis import Analyzer
from visualizer import Visualizer
from force_ml import MLModel, StructureGenerator
from poscar_io import PoscarWriter

def main():
    parser = argparse.ArgumentParser(description="VASP XML Analysis & ML Generation")
    parser.add_argument('files', metavar='F', type=str, nargs='+', help='List of vasprun.xml files')
    parser.add_argument('--generate', type=int, default=0, help='Number of zero-force cells to generate')
    args = parser.parse_args()

    all_data = []

    # Metadata from the last file loaded
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

                # Store metadata for generation
                last_lattice = vp.extract_basis()
                last_step_idx = df['step'].max()
                last_step_df = df[df['step'] == last_step_idx]

                last_elements = last_step_df['element'].tolist()
                last_positions = last_step_df[['x','y','z']].values
                last_unique_elements = vp.atom_types
                last_counts = vp.counts
                detected_coord_type = vp.coordinate_type

                print(f"Loaded {f}: {len(df)} atoms*steps (Coords: {vp.coordinate_type})")
            else:
                print(f"No valid data in {f}")
        except Exception as e:
            print(f"Error reading {f}: {e}")

    if not all_data:
        print("No valid data found.")
        sys.exit(1)

    combined_df = pd.concat(all_data, ignore_index=True)

    # --- Analysis ---
    print("\n--- Analysis ---")
    ana = Analyzer(combined_df)
    print("Force Stats:\n", ana.force_stats())
    if 'pressure' in combined_df.columns:
        print("\nPressure Stats:\n", ana.pressure_stats())

    # --- ML Training & Generation ---
    if args.generate > 0:
        print("\n--- ML Training & Generation ---")
        if last_positions is None:
            print("Error: No template structure found.")
        else:
            ml = MLModel(combined_df)
            ml.train()

            generator = StructureGenerator(ml, last_positions, last_elements, last_lattice)

            # Pass detected coordinate type to generator
            new_structs = generator.generate_zero_force_structures(
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
                        title=f"ML_Generated_{idx}_ForceOptimized"
                        )

    # --- Visualizing (Run Last) ---
    print("\n--- Visualizing ---")
    viz = Visualizer(combined_df)
    viz.plot_trajectory()
    viz.plot_stress_distribution()

if __name__ == "__main__":
    main()

