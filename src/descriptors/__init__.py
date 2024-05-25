from .descriptors import DescriptorGenerator
from .descriptors import (
    MolWt, logP, TopoPSA, HBonds, RotBond, AromaRing, MolVolume, MorganFingerPrint, MACCSFingerPrint, BCUT, GetAWay, GetFeatureInvariants, USR, WHIM, MORSE
    )
from ._sasa import SASA

AVAILABLE_DESCRIPTORS = [
    #MolWt, logP, TopoPSA, HBonds, SASA, RotBond, AromaRing, MolVolume, MorganFingerPrint, MACCSFingerPrint, BCUT, GetFeatureInvariants, USR, WHIM, MORSE
    MolWt, logP, SASA, MorganFingerPrint, MACCSFingerPrint, WHIM, MORSE
]