"""This module contains the implementation of the SASA descriptors."""

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdFreeSASA import (
    CalcSASA,
    MakeFreeSasaPolarAtomQuery,
    MakeFreeSasaAPolarAtomQuery,
)
from ._abc import DescriptorsABC

SYMBOL_RADIUS = {
    "H": 1.10,
    "C": 1.70,
    "N": 1.55,
    "O": 1.52,
    "P": 1.80,
    "S": 1.80,
    "SE": 1.90,
    "F": 1.47,
    "CL": 1.75,
    "BR": 1.83,
    "I": 1.98,
    "LI": 1.81,
    "BE": 1.53,
    "B": 1.92,
    "NA": 2.27,
    "MG": 1.74,
    "AL": 1.84,
    "SI": 2.10,
    "K": 2.75,
    "CA": 2.31,
    "GA": 1.87,
    "GE": 2.11,
    "AS": 1.85,
    "RB": 3.03,
    "SR": 2.49,
    "IN": 1.93,
    "SN": 2.17,
    "SB": 2.06,
    "TE": 2.06,
}


def _classifyAtoms(mol, polar_atoms=[7, 8, 15, 16]):
    # Taken from https://github.com/rdkit/rdkit/issues/1827#issuecomment-385193214

    radii = []
    for atom in mol.GetAtoms():
        atom.SetProp("SASAClassName", "Apolar")  # mark everything as apolar to start
        if (
            atom.GetAtomicNum() in polar_atoms
        ):  # identify polar atoms and change their marking
            atom.SetProp("SASAClassName", "Polar")  # mark as polar
        elif atom.GetAtomicNum() == 1:
            if atom.GetBonds()[0].GetOtherAtom(atom).GetAtomicNum() in polar_atoms:
                atom.SetProp("SASAClassName", "Polar")  # mark as polar
        radii.append(SYMBOL_RADIUS[atom.GetSymbol().upper()])
    return radii


class SASA(DescriptorsABC):
    """solvent accessible surface area (SASA)

    Returns:
        list[float]: A list of 3 values: the surface area of the molecule,
        the surface area of the molecule with polar atoms, and the
        surface area of the molecule with polar and non-polar atoms
    """

    def __call__(self, SMILES):
        mol = Chem.MolFromSmiles(SMILES)
        molH = Chem.AddHs(mol)
        AllChem.EmbedMolecule(molH)
        radius = _classifyAtoms(molH)
        return [
            CalcSASA(molH, radius),
            CalcSASA(molH, radius, query=MakeFreeSasaAPolarAtomQuery()),
            CalcSASA(molH, radius, query=MakeFreeSasaPolarAtomQuery()),
        ]
