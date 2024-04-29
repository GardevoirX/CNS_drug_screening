import pytest
from src.descriptors import AVAILABLE_DESCRIPTORS

TEST_SMILES = 'CC(=O)Nc1ccc(cc1)O'


@pytest.mark.parametrize("descriptor", AVAILABLE_DESCRIPTORS)
def test_descriptors(descriptor):
    des = descriptor()
    result = des(TEST_SMILES)
    assert isinstance(result, list)
    assert len(result) == 1
