from .descriptors import DescriptorGenerator
from .descriptors import (
    MolWt,
    MolAbsCharge,
    NumHeavyAtoms,
    logP,
    TopoPSA,
    HBonds,
    RotBond,
    NumValenceElectron,
    NumMaxAbsPartialCharge,
    NumMinAbsPartialCharge,
    AromaRing,
    MolVolume,
    MorganFingerPrint,
    MACCSFingerPrint,
    BCUT,
    GetAWay,
    Invariants,
    USR,
    WHIM,
    MORSE,
    USRCAT,
    TopologicalTorsionFingerprint,
    MQNs,
    PEOE_VSA,
    SMR_VSA,
    SlogP_VSA,
    Autocorr2D,
)
from ._sasa import SASA

AVAILABLE_DESCRIPTORS = [
    # Human experts descriptors
    MolWt,
    MolAbsCharge,
    NumMaxAbsPartialCharge,
    NumMinAbsPartialCharge,
    RotBond,
    NumHeavyAtoms,

    # High dimensional descriptors
    # Topological fingerprint descriptors
    TopologicalTorsionFingerprint,
    MorganFingerPrint,
    # Topological descriptors
    USR,
    USRCAT,
    # Quantum descriptors
    MQNs,
    # Electrical properties descriptors
    PEOE_VSA,
    # Partition coefficients descriptors
    SlogP_VSA,
    # Topological descriptors
    Autocorr2D,
]
