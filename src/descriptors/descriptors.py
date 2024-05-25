import numpy as np

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
from rdkit.Chem.rdMolDescriptors import (
    GetMorganFingerprintAsBitVect,
    GetMACCSKeysFingerprint,
    CalcGETAWAY,
    CalcMORSE,
    BCUT2D,
    CalcWHIM,
    GetFeatureInvariants,
    GetUSR,
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
        mol = Chem.MolFromSmiles(SMILES)
        molH = Chem.AddHs(mol)
        AllChem.EmbedMolecule(molH)
        for descriptor in self.descriptors:
            d = descriptor()
            results += d(molH)
        return results


class MolWt(DescriptorsABC):
    def __call__(self, mol):
        return [ExactMolWt(mol)]


class logP(DescriptorsABC):
    """logP, partition coefficient (oil/ water)"""

    def __call__(self, mol):
        return [MolLogP(mol)]


class TopoPSA(DescriptorsABC):
    """topological polar surface area (TPSA)"""

    def __call__(self, mol):
        return [TPSA(mol)]


class HBonds(DescriptorsABC):
    """numbers related to H-Bonds (like donor, acceptor)"""

    def __call__(self, mol):
        return [NumHAcceptors(mol), NumHDonors(mol)]


class SASA(DescriptorsABC):
    """solvent accessible surface area (SASA)"""

    def __call__(self, mol):
        radius = classifyAtoms(mol)
        return [CalcSASA(mol, radius)]


class RotBond(DescriptorsABC):
    """number of rotatable bonds"""

    def __call__(self, mol):
        return [NumRotatableBonds(mol)]


class AromaRing(DescriptorsABC):
    """number of aromatic rings."""

    def __call__(self, mol):
        return [NumAromaticRings(mol)]


class BondFeatures(DescriptorsABC):
    """number of bonds and number of bonds of each type"""

    def __call__(self, mol):
        bond_type = [b.GetBondTypeAsDouble() for b in mol.GetBonds()]
        bond_count = {1: 0, 1.5: 0, 2: 0, 3: 0}
        for b in bond_type:
            bond_count[b] += 1
        bond_count = list(bond_count.values())

        return [mol.GetNumBonds()] + bond_count


class MolVolume(DescriptorsABC):
    """Volume of the molecule"""

    def __call__(self, mol):
        return [AllChem.ComputeMolVolume(mol)]


class MorganFingerPrint(DescriptorsABC):

    def __call__(self, mol):
        fp = GetMorganFingerprintAsBitVect(mol, 5, nBits=2048)
        fp = fp.ToBitString()
        return [int(item) for item in list(fp)]


class MACCSFingerPrint(DescriptorsABC):
    def __call__(self, mol):
        fp = GetMACCSKeysFingerprint(mol)
        fp = fp.ToBitString()
        return [int(item) for item in list(fp)]


class BCUT(DescriptorsABC):
    def __call__(self, mol):
        return BCUT2D(mol)


class GetAWay(DescriptorsABC):
    """GETAWAY descriptor, 273 continuous features"""
    def __call__(self, mol):
        return CalcGETAWAY(mol)


class WHIM(DescriptorsABC):
    """WHIM descriptor, 114 continuous features"""
    def __call__(self, mol):
        results = np.array(CalcWHIM(mol))
        results[np.isnan(results)] = 0.0

        return list(results)


class Invariants(DescriptorsABC):
    def __call__(self, mol):
        return GetFeatureInvariants(mol)


class USR(DescriptorsABC):
    """USR descriptor, 12 continuous features"""
    def __call__(self, mol):
        return GetUSR(mol)


class MORSE(DescriptorsABC):
    """MORSE descriptor, 224 continuous features"""
    def __call__(self, mol):
        return CalcMORSE(mol)
