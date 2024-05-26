import random
import pytest
from numpy import allclose
from rdkit import Chem
from rdkit.Chem import AllChem
from src.descriptors import AVAILABLE_DESCRIPTORS, DescriptorGenerator

TEST_SMILES = 'CC(=O)Nc1ccc(cc1)O'
MOL = Chem.MolFromSmiles(TEST_SMILES)
MOLH = Chem.AddHs(MOL)
AllChem.EmbedMolecule(MOLH)


@pytest.mark.parametrize("descriptor", AVAILABLE_DESCRIPTORS)
def test_descriptors(descriptor):
    des = descriptor()
    result = des(MOLH)
    assert isinstance(result, list)
    assert len(result) != 0

def test_descriptor_generator():

    selected_descriptors = random.sample(AVAILABLE_DESCRIPTORS, 5)
    generator = DescriptorGenerator(selected_descriptors)
    results = generator(TEST_SMILES)

    individual_results = []
    for descriptor in selected_descriptors:
        des = descriptor()
        individual_results += des(MOLH)

    assert allclose(results, individual_results, atol=5e-1)
