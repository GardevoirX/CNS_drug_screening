from src.descriptors import AVAILABLE_DESCRIPTORS

TEST_SMILES = 'CC(=O)Nc1ccc(cc1)O'

def test_descriptors():
    for descriptor in AVAILABLE_DESCRIPTORS:
        des = descriptor()
        result = des(TEST_SMILES)
        assert isinstance(result, list)
        assert len(result) == 1
