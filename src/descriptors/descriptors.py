from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Descriptors import (
    ExactMolWt,
    MolLogP,
    TPSA,
    NumHAcceptors,
    NumHDonors,
    NumRotatableBonds,
    NumAromaticRings,
)

from ._abc import DescriptorsABC


class DescriptorGenerator(DescriptorsABC):
    """generate specified descriptors from SMILES

    Args:
        descriptors (list): list of descriptors
    """

    def __init__(self, descriptors: list):
        self.descriptors = descriptors

    def __call__(self, SMILES):
        results = []
        for descriptor in self.descriptors:
            d = descriptor()
            results += d(SMILES)
        return results


class MolWt(DescriptorsABC):
    def __call__(self, SMILES):
        mol = Chem.MolFromSmiles(SMILES)
        return [ExactMolWt(mol)]


class logP(DescriptorsABC):
    """logP, partition coefficient (oil/ water)"""

    def __call__(self, SMILES):
        mol = Chem.MolFromSmiles(SMILES)
        return [MolLogP(mol)]


class TopoPSA(DescriptorsABC):
    """topological polar surface area (TPSA)"""

    def __call__(self, SMILES):
        mol = Chem.MolFromSmiles(SMILES)
        return [TPSA(mol)]


class HBonds(DescriptorsABC):
    """numbers related to H-Bonds (like donor, acceptor)"""

    def __call__(self, SMILES):
        mol = Chem.MolFromSmiles(SMILES)
        return [NumHAcceptors(mol), NumHDonors(mol)]


class SASA(DescriptorsABC):
    """solvent accessible surface area (SASA)"""

    def __call__(self, SMILES):
        mol = Chem.MolFromSmiles(SMILES)
        molH = Chem.AddHs(mol)
        AllChem.EmbedMolecule(molH)
        radius = classifyAtoms(molH)
        return [CalcSASA(molH, radius)]


class RotBond(DescriptorsABC):
    """number of rotatable bonds"""

    def __call__(self, SMILES):
        mol = Chem.MolFromSmiles(SMILES)
        return [NumRotatableBonds(mol)]


class AromaRing(DescriptorsABC):
    """number of aromatic rings."""

    def __call__(self, SMILES):
        mol = Chem.MolFromSmiles(SMILES)
        return [NumAromaticRings(mol)]

class BondFeatures(DescriptorsABC):
    """number of bonds and number of bonds of each type"""

    def __call__(self, SMILES):
        mol = Chem.MolFromSmiles(SMILES)
        bond_type = [b.GetBondTypeAsDouble() for b in mol.GetBonds()]
        bond_count = {1: 0, 1.5: 0, 2: 0, 3: 0}
        for b in bond_type:
            bond_count[b] += 1
        bond_count = list(bond_count.values())

        return [mol.GetNumBonds()] + bond_count
