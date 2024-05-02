import random
import pytest
from numpy import allclose
from src.descriptors import AVAILABLE_DESCRIPTORS, DescriptorGenerator

TEST_SMILES = 'CC(=O)Nc1ccc(cc1)O'


@pytest.mark.parametrize("descriptor", AVAILABLE_DESCRIPTORS)
def test_descriptors(descriptor):
    des = descriptor()
    result = des(TEST_SMILES)
    assert isinstance(result, list)
    assert len(result) != 0

def test_descriptor_generator():

    selected_descriptors = random.sample(AVAILABLE_DESCRIPTORS, 5)
    generator = DescriptorGenerator(selected_descriptors)
    results = generator(TEST_SMILES)

    individual_results = []
    for descriptor in selected_descriptors:
        des = descriptor()
        individual_results += des(TEST_SMILES)

    assert allclose(results, individual_results, atol=5e-1)
