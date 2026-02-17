# vasp_parser.py
import numpy as np
from defusedxml import ElementTree
import pandas as pd

class VaspParser:
    def __init__(self, filepath):
        self.filepath = filepath
        self.tree = ElementTree.parse(filepath)
        self.root = self.tree.getroot()
        self.atom_types = []
        self.symbols = []
        self._parse_atom_info()

    def _parse_atom_info(self):
        """Parses atom types and counts to map indices to elements."""
        atominfo = self.root.find('atominfo')
        if atominfo is None:
            raise ValueError("No <atominfo> found in XML.")

        # Get atom counts per type
        atoms_array = atominfo.find(".//array[@name='atoms']/set")
        if atoms_array is None:
            # Fallback logic if specific array structure differs
             pass

        # Extract atomic symbols and create a list matching the atoms
        # This is a simplified extraction based on standard vasprun.xml structure
        for rc in atoms_array.findall('rc'):
            cols = rc.findall('c')
            element = cols[0].text.strip()
            self.symbols.append(element)
            # Map element to a simple integer type for ML
            if element not in self.atom_types:
                self.atom_types.append(element)

    def get_atom_types_vector(self):
        """Returns a numeric representation of atom types for ML."""
        type_map = {sym: i for i, sym in enumerate(self.atom_types)}
        return [type_map[s] for s in self.symbols]

    def extract_calculations(self):
        """
        Extracts positions and forces from all <calculation> steps.
        Returns a list of dictionaries.
        """
        data = []

        # Iterate over all calculation blocks (ionic steps)
        for calculation in self.root.findall('calculation'):
            forces = self._extract_varray(calculation, 'forces')
            positions = self._extract_varray(calculation, 'positions')

            # Skip steps that might not have both (e.g., initial setup)
            if forces is not None and positions is not None:
                df = pd.DataFrame(positions, columns=['x', 'y', 'z'])
                df[['fx', 'fy', 'fz']] = forces
                df['element'] = self.symbols
                df['step'] = len(data)
                data.append(df)

        return pd.concat(data, ignore_index=True) if data else pd.DataFrame()

    def _extract_varray(self, parent_node, name):
        """Helper to extract numpy array from <varray> tag."""
        varray = parent_node.find(f"./varray[@name='{name}']")
        if varray is None:
            # Sometimes positions are inside a <structure> tag inside calculation
            struct = parent_node.find('structure')
            if struct:
                varray = struct.find(f"./varray[@name='{name}']")

        if varray is not None:
            rows = []
            for v in varray.findall('v'):
                rows.append([float(x) for x in v.text.split()])
            return np.array(rows)
        return None
