from rdkit import Chem
from rdkit.Chem.Descriptors import ExactMolWt

from ._abc import DescriptorsABC

class MolWt(DescriptorsABC):
    def __call__(self, SMILES):
        mol = Chem.MolFromSmiles(SMILES)
        return [ExactMolWt(mol)]