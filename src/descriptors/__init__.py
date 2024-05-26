from .descriptors import DescriptorGenerator
from .descriptors import (
    MolWt,
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
    # SASA,
    # logP,
    # TopoPSA,
    # MolVolume,
    # WHIM,
    MORSE,
    USRCAT,
    MQNs,
    PEOE_VSA,
    # SMR_VSA,
    SlogP_VSA
]
