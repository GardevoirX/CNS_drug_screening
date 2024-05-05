from .descriptors import DescriptorGenerator
from .descriptors import (
    MolWt, logP, TopoPSA, HBonds, SASA, RotBond, AromaRing
    )

AVAILABLE_DESCRIPTORS = [
    MolWt, logP, TopoPSA, HBonds, SASA, RotBond, AromaRing
]