import pytest
import torch

from numpy import allclose
from src.data.dataset import CNSDataset
from src.descriptors import DescriptorGenerator, MolWt


def test_dataset():
    dataset = CNSDataset(pytest.EXAMPLE_DATA, transform=DescriptorGenerator([MolWt]))
    assert dataset._processed_data.size() == torch.Size([5, 1])
    max = torch.max(dataset._processed_data, axis=0).values
    min = torch.min(dataset._processed_data, axis=0).values
    dataset.normalize(max, min)
    assert torch.max(dataset._processed_data, axis=0).values == 1
    assert torch.min(dataset._processed_data, axis=0).values == 0
    mean = torch.mean(dataset._processed_data, axis=0)
    std = torch.std(dataset._processed_data, axis=0)
    dataset.standardize(mean, std)
    assert allclose(torch.mean(dataset._processed_data, axis=0).cpu(), 0, atol=1e-7)
    assert allclose(torch.std(dataset._processed_data, axis=0).cpu(), 1, atol=1e-7)
