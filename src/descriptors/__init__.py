from .descriptors import DescriptorGenerator
from .descriptors import (
    MolWt, logP, TopoPSA, HBonds, RotBond, AromaRing, MolVolume
    )
from ._sasa import SASA

AVAILABLE_DESCRIPTORS = [
    MolWt, logP, TopoPSA, HBonds, SASA, RotBond, AromaRing, MolVolume
]