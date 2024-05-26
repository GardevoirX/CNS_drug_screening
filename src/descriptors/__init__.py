from .descriptors import DescriptorGenerator
from .descriptors import (
    MolWt,
    MolAbsCharge,
    logP,
    TopoPSA,
    HBonds,
    RotBond,
    AromaRing,
    MolVolume,
    MorganFingerPrint,
    MACCSFingerPrint,
    BCUT,
    GetAWay,
    GetFeatureInvariants,
    USR,
    WHIM,
    MORSE,
    USRCAT,
    TopologicalTorsionFingerprint,
    MQNs,
    PEOE_VSA,
    SMR_VSA,
    SlogP_VSA
)
from ._sasa import SASA

AVAILABLE_DESCRIPTORS = [
    MolWt,
    MolAbsCharge,
    RotBond,
    # SASA,
    # logP,
    # TopoPSA,
    # MolVolume,
    # WHIM,
    # MORSE,
    # USR,
    USRCAT,
    MQNs,
    PEOE_VSA,
    # SMR_VSA,
    SlogP_VSA
]
