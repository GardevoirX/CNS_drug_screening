import pandas as pd

from torch.utils.data import Dataset

class CNSDataset(Dataset):
    def __init__(self, data_file: str, transform=None, target_transform=None):
        self.raw_data = pd.read_csv(data_file)
        self.transform = transform
        self.target_transform = target_transform
        
    def __len__(self):
        return len(self.raw_data)
    
    def __getitem__(self, idx):
        item = self.raw_data.iloc[idx]
        properties = item['SMILES']
        label = item['TARGET']
        if self.transform is not None:
            properties = self.transform(properties)
        if self.target_transform is not None:
            label = self.target_transform(label)
        
        return properties, label
    
    @property
    def SMILES(self):
        return self.raw_data['SMILES']
    
    @property
    def labels(self):
        return self.raw_data['TARGET']