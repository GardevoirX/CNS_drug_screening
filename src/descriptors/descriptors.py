from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Descriptors import ExactMolWt, MolLogP, TPSA, NumHAcceptors, NumHDonors, NumRotatableBonds, NumAromaticRings
from rdkit.Chem.rdFreeSASA import CalcSASA, classifyAtoms

from ._abc import DescriptorsABC

class MolWt(DescriptorsABC):
    def __call__(self, SMILES):
        mol = Chem.MolFromSmiles(SMILES)
        return [ExactMolWt(mol)]

class logP(DescriptorsABC):
    """ logP, partition coefficient (oil/ water) """
    def __call__(self, SMILES):
        mol = Chem.MolFromSmiles(SMILES)
        return [MolLogP(mol)]

class TopoPSA(DescriptorsABC):
    """ topological polar surface area (TPSA) """
    def __call__(self, SMILES):
        mol = Chem.MolFromSmiles(SMILES)
        return [TPSA(mol)]

class HBonds(DescriptorsABC):
    """ numbers related to H-Bonds (like donor, acceptor) """
    def __call__(self, SMILES):
        mol = Chem.MolFromSmiles(SMILES)
        return [NumHAcceptors(mol), NumHDonors(mol)]

class SASA(DescriptorsABC):
    """ solvent accessible surface area (SASA) """
    def __call__(self, SMILES):
        mol = Chem.MolFromSmiles(SMILES)
        molH = Chem.AddHs(mol)
        AllChem.EmbedMolecule(molH)
        radius = classifyAtoms(molH)
        return [CalcSASA(molH, radius)]

class RotBond(DescriptorsABC):
    """ number of rotatable bonds """
    def __call__(self, SMILES):
        mol = Chem.MolFromSmiles(SMILES)
        return [NumRotatableBonds(mol)]

class AromaRing(DescriptorsABC):
    """ number of aromatic rings. """
    def __call__(self, SMILES):
        mol = Chem.MolFromSmiles(SMILES)
        return [NumAromaticRings(mol)]