from rdkit import Chem
from rdkit.Chem.Descriptors import ExactMolWt, MolLogP, TPSA, NumHAcceptors, NumHDonors, NumRotatableBonds, NumAromaticRings
from rdkit.Chem.rdFreeSASA import CalcSASA, classifyAtoms

from ._abc import DescriptorsABC

class MolWt(DescriptorsABC):
    def __call__(self, SMILES):
        mol = Chem.MolFromSmiles(SMILES)
        return [ExactMolWt(mol)]

""" logP, partition coefficient (oil/ water) """

class logP(DescriptorsABC):
    def __call__(self, SMILES):
        mol = Chem.MolFromSmiles(SMILES)
        return [MolLogP(mol)]
    
""" topological polar surface area (TPSA) """

class TPSA(DescriptorsABC):
    def __call__(self, SMILES):
        mol = Chem.MolFromSmiles(SMILES)
        return [TPSA(mol)]

""" numbers related to H-Bonds (like donor, acceptor) """

class HBonds(DescriptorsABC):
    def __call__(self, SMILES):
        mol = Chem.MolFromSmiles(SMILES)
        return [NumHAcceptors(mol), NumHDonors(mol)]

""" solvent accessible surface area (SASA) """

class SASA(DescriptorsABC):
    def __call__(self, SMILES):
        mol = Chem.MolFromSmiles(SMILES)
        radius = classifyAtoms(mol)
        return [CalcSASA(mol, radius)]

""" number of rotatable bonds """

class RotBond(DescriptorsABC):
    def __call__(self, SMILES):
        mol = Chem.MolFromSmiles(SMILES)
        return [NumRotatableBonds(mol)]

""" number of aromatic rings. """

class AromaRing(DescriptorsABC):
    def __call__(self, SMILES):
        mol = Chem.MolFromSmiles(SMILES)
        return [NumAromaticRings(mol)]