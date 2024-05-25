import pandas as pd
import torch

from rich.progress import track
from torch import Tensor, any, isnan
from torch.utils.data import Dataset


class CNSDataset(Dataset):
    def __init__(
        self,
        data_file: str,
        transform=None,
        target_transform=None,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    ):
        self.raw_data = pd.read_csv(data_file)
        self.transform = transform
        self.target_transform = target_transform
        self.device = device
        self._do_transform()

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, idx):

        return self._processed_data[idx], self._processed_label[idx]

    def _do_transform(self):
        if self.transform is None:
            self._processed_data = self.raw_data["SMILES"]
        else:
            self._processed_data = Tensor(
                [self.transform(smiles) for smiles in track(self.raw_data["SMILES"])]
            ).to(self.device)
        if self.target_transform is None:
            self._processed_label = self.raw_data["TARGET"]
        else:
            self._processed_label = Tensor([
                self.target_transform(label) for label in self.raw_data["TARGET"]
            ]).to(self.device)

    @property
    def SMILES(self):
        return self.raw_data["SMILES"]

    @property
    def labels(self):
        return self.raw_data["TARGET"]

    def standardize(self, means, stds):
        self._processed_data = (self._processed_data - means) / stds
        if any(mask := any(isnan(self._processed_data), axis=0)):
            print("NaN in descriptors, remove it")
        self._processed_data = self._processed_data[:, ~mask]
