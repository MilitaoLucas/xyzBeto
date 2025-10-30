#!/usr/bin/env python3
"""
XYZ File Converter Tool

This script processes XYZ files containing molecular clusters and converts them
to different formats including PDB and Excel. It identifies individual molecules
within the cluster based on bonding patterns and creates labeled outputs.
"""

import argparse
import sys
import ase
import os
import io
from ase.io import xyz
from ase.data import covalent_radii, atomic_numbers
from typing import Optional
import numpy as np
from rdkit.Chem import AllChem as Chem
import pandas as pd
def write_cluster_pdb(molecules_labeled: dict, bonds: np.ndarray, output_file: str = "all_molecules.pdb"):
    """
    Save all molecules as a single cluster in one PDB file.
    PDB format is used as it correctly stores custom atom and residue names.
    """

    cluster_mol = Chem.RWMol()
    label_to_idx = {}
    all_coords = []

    residue_number = 1

    for mol_name, mol_data in molecules_labeled.items():
        for atom_data in mol_data:
            og_label = atom_data[0]
            element_symbol = og_label[0]
            label = f"{og_label}"
            atom_name_label = f"{label}"
            if len(atom_name_label) > 4:
                atom_name_label = atom_name_label[:4]

            atomic_num = atomic_numbers[element_symbol]
            atom = Chem.Atom(atomic_num)
            atom_idx = cluster_mol.AddAtom(atom)

            atom_obj = cluster_mol.GetAtomWithIdx(atom_idx)

            info = Chem.AtomPDBResidueInfo()
            info.SetName(atom_name_label)
            info.SetResidueName(mol_name)
            info.SetResidueNumber(residue_number)

            atom_obj.SetMonomerInfo(info)

            coords = atom_data[1:].astype(float)
            all_coords.append(coords)
            label_to_idx[og_label] = atom_idx

        residue_number += 1

    for bond in bonds:
        atom1_idx = label_to_idx[bond[0]]
        atom2_idx = label_to_idx[bond[1]]
        cluster_mol.AddBond(atom1_idx, atom2_idx, Chem.BondType.SINGLE)

    cluster_mol = cluster_mol.GetMol()

    conf = Chem.Conformer(cluster_mol.GetNumAtoms())
    for i, coords in enumerate(all_coords):
        conf.SetAtomPosition(i, tuple(coords))
    cluster_mol.AddConformer(conf)

    writer_string = Chem.MolToPDBBlock(cluster_mol)

    with open(output_file, 'w') as f:
        f.write(writer_string)

    return cluster_mol

def dfs(adj_matrix: np.ndarray[np.int8], node: int, visited: np.ndarray[bool], component: Optional[list]):
    visited[node] = True
    component.append(node)

    neighbors = np.where(adj_matrix[node] == 1)[0]
    for i in neighbors:
        if not visited[i]:
            dfs(adj_matrix, i, visited, component)
    return component

def find_all_molecules(adj_matrix: np.ndarray[np.int8]):
    n = adj_matrix.shape[0]
    visited = np.zeros(n, dtype=bool)
    components = []
    for node in range(n):
        if not visited[node]:
            component = []
            dfs(adj_matrix, node, visited, component)
            components.append(component)

    return components

def main(filename: str, mol_prefix: str = "mol", min_distance: float = 0.2, tolerance: float = 1.2,
                         output_pdb: str = "all_molecules.pdb", output_excel: str = None, output_csv: str = None):
    with open(filename, "r") as f:
        contents = [i.strip() for i in f.readlines() if i.strip() != ""]

    count = int(contents[0])
    map_labels = {}
    for i in range(1, count+1):
        a = [i for i in contents[i].split(" ") if i != ""]
        map_labels[a[0]] = tuple(a[1:])

    xyz_string = f"{count}{os.linesep*2}"
    for k, v in map_labels.items():
        xyz_string += f"{k[0]} {'   '.join(v)}{os.linesep}"

    unlabeled_io = io.StringIO(xyz_string)
    molase = ase.io.read(unlabeled_io, format="xyz")
    distances = molase.get_all_distances()
    Z = molase.numbers
    orderd_atoms = list(map_labels.keys())
    bonds = []
    bonds_labeled = []
    for i in range(count):
        for j in range(i+1, count):
            rsum = covalent_radii[Z[i]] + covalent_radii[Z[j]]
            d = distances[i][j]
            if min_distance < d <= rsum * tolerance:
                bonds.append((i,j))
                bonds_labeled.append((orderd_atoms[i], orderd_atoms[j]))
    bonds = np.array(bonds)
    bonds_labeled = np.array(bonds_labeled)
    adj_matrix = np.zeros((count, count), dtype=np.int8)
    adj_matrix[bonds[:, 0], bonds[:, 1]] = 1
    adj_matrix[bonds[:, 1], bonds[:, 0]] = 1
    molecules = find_all_molecules(adj_matrix)
    molecules_labeled = {}
    ls_data = []
    for it, mol in enumerate(molecules):
        data = np.array([[k, *v] for k, v in map_labels.items()])[mol]
        key = f"{mol_prefix}_{it}"
        ls_data += [[key, str(i)] for i in data[:, 0]]
        molecules_labeled[key] = data
    df = pd.DataFrame(ls_data)
    
    # Add Excel output if specified
    if output_excel is not None:
        df.to_excel(output_excel, index=False, header=["Molecule Label", "Atom Label"])
    
    # Set default CSV output if not specified
    if output_csv is None:
        output_csv = filename.replace(".xyz", ".csv")
    
    df.to_csv(output_csv, index=False, header=["Molecule Label", "Atom Label"])
    
    write_cluster_pdb(molecules_labeled, bonds_labeled, output_pdb)
    return molecules_labeled, ls_data


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Finds and labels molecules in csv (can be opened in excel) and converts to pdb",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
 %(prog)s Final_cluster.xyz
 %(prog)s input.xyz --mol-prefix molecule --output-pdb cluster.pdb
 %(prog)s data.xyz --min-distance 0.3 --tolerance 1.5
 %(prog)s input.xyz --output-csv molecules.csv
 %(prog)s input.xyz --output-excel results.xlsx --output-csv results.csv
       """
    )
    
    parser.add_argument(
        "input_file",
        help="Path to the input XYZ file containing molecular cluster data"
    )
    
    parser.add_argument(
        "--mol-prefix",
        default="mol",
        help="Prefix for molecule names in output files (default: mol)"
    )
    
    parser.add_argument(
        "--min-distance",
        type=float,
        default=0.2,
        help="Minimum distance threshold for bond detection (default: 0.2)"
    )
    
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1.2,
        help="Tolerance factor for covalent radius sum in bond detection (default: 1.2)"
    )
    
    parser.add_argument(
        "--output-pdb",
        default="all_molecules.pdb",
        help="Output PDB filename (default: all_molecules.pdb)"
    )
    
    parser.add_argument(
        "--output-excel",
        help="Output Excel filename (optional - if not provided, no Excel file will be created)"
    )
    
    parser.add_argument(
        "--output-csv",
        help="Output CSV filename (default: same as input with .csv extension)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    return parser.parse_args()

def tool_call():
    """Main function to handle command-line execution."""
    args = parse_arguments()

    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' not found.", file=sys.stderr)
        sys.exit(1)

    # Set default CSV output if not specified
    if args.output_csv is None:
        args.output_csv = args.input_file.replace(".xyz", ".csv")

    try:
        if args.verbose:
            print(f"Processing XYZ file: {args.input_file}")
            print(f"Molecule prefix: {args.mol_prefix}")
            print(f"Min distance: {args.min_distance}")
            print(f"Tolerance: {args.tolerance}")
            print(f"Output PDB: {args.output_pdb}")
            if args.output_excel:
                print(f"Output Excel: {args.output_excel}")
            print(f"Output CSV: {args.output_csv}")

        molecules_labeled, ls_data = main(
            filename=args.input_file,
            mol_prefix=args.mol_prefix,
            min_distance=args.min_distance,
            tolerance=args.tolerance,
            output_pdb=args.output_pdb,
            output_excel=args.output_excel,
            output_csv=args.output_csv
        )

        if args.verbose:
            print(f"Successfully processed {len(molecules_labeled)} molecules")
            print(f"Output files created:")
            print(f"  - PDB: {args.output_pdb}")
            if args.output_excel:
                print(f"  - Excel: {args.output_excel}")
            print(f"  - CSV: {args.output_csv}")

    except Exception as e:
        print(f"Error processing file: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    tool_call()



