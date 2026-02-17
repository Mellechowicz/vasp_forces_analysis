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
        self.counts = []
        self.coordinate_type = "Direct" # Default assumption
        self._parse_atom_info()

    def _parse_atom_info(self):
        """Parses atom types and counts."""
        atominfo = self.root.find('atominfo')
        if atominfo is None: return

        # Method 1: explicit atoms array
        atoms_array = atominfo.find(".//array[@name='atoms']/set")
        if atoms_array is not None:
            self.symbols = []
            for rc in atoms_array.findall('rc'):
                cols = rc.findall('c')
                element = cols[0].text.strip()
                self.symbols.append(element)

            # Deduce types and counts
            unique_elements = []
            for s in self.symbols:
                if s not in unique_elements:
                    unique_elements.append(s)
            self.atom_types = unique_elements
            self.counts = [self.symbols.count(e) for e in self.atom_types]

        else:
            # Method 2: atomtypes array
            types_array = atominfo.find(".//array[@name='atomtypes']/set")
            if types_array:
                for rc in types_array.findall('rc'):
                    cols = rc.findall('c')
                    count = int(cols[0].text.strip())
                    element = cols[1].text.strip()
                    self.atom_types.append(element)
                    self.counts.append(count)
                    self.symbols.extend([element] * count)

    def extract_basis(self):
        """Extracts the final lattice basis vectors."""
        struct = self.root.findall('structure')[-1]
        basis_node = struct.find(".//varray[@name='basis']")
        if basis_node:
            rows = []
            for v in basis_node.findall('v'):
                rows.append([float(x) for x in v.text.split()])
            return np.array(rows)
        return np.eye(3)

    def extract_data(self):
        """Extracts positions, forces, and stress."""
        data = []

        for i, calc in enumerate(self.root.findall('calculation')):
            forces = self._extract_varray(calc, 'forces')
            positions = self._extract_varray(calc, 'positions')
            stress = self._extract_varray(calc, 'stress')

            energy_val = None
            energy_block = calc.find('energy')
            if energy_block is not None:
                # Prefer free energy (TOTEN)
                e_node = energy_block.find("./i[@name='e_fr_energy']")
                if e_node is not None:
                    energy_val = float(e_node.text.strip())

            if forces is not None and positions is not None:
                # Coordinate Type Check (Heuristic on first valid step)
                if i == 0 or self.coordinate_type == "Direct":
                    # If any coordinate is > 1.5, likely Cartesian (unless unit cell is tiny)
                    if np.max(np.abs(positions)) > 1.5:
                        self.coordinate_type = "Cartesian"
                    else:
                        self.coordinate_type = "Direct"

                if len(positions) != len(self.symbols):
                    # Skip mismatched steps
                    continue

                df_step = pd.DataFrame(positions, columns=['x', 'y', 'z'])
                df_step[['fx', 'fy', 'fz']] = forces
                df_step['element'] = self.symbols
                df_step['step'] = i
                df_step['file_source'] = self.filepath
                df_step['energy'] = energy_val

                if stress is not None:
                    # Pressure in kB (approx mean of diagonal)
                    pressure = -np.mean(np.diag(stress))
                    df_step['pressure'] = pressure
                    df_step['stress_xx'] = stress[0,0]
                    df_step['stress_yy'] = stress[1,1]
                    df_step['stress_zz'] = stress[2,2]

                data.append(df_step)

        return pd.concat(data, ignore_index=True) if data else pd.DataFrame()

    def _extract_varray(self, parent_node, name):
        """Helper to extract numpy array from <varray> tag."""
        varray = parent_node.find(f"./varray[@name='{name}']")
        if varray is None:
            struct = parent_node.find('structure')
            if struct:
                varray = struct.find(f"./varray[@name='{name}']")

        if varray is not None:
            rows = []
            for v in varray.findall('v'):
                rows.append([float(x) for x in v.text.split()])
            return np.array(rows)
        return None

