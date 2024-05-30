import argparse

import pandas as pd
import torch

from src.data.dataset import CNSDataset
from src.descriptors import DescriptorGenerator, AVAILABLE_DESCRIPTORS

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser(description="Inference")
    parser.add_argument("--data_file", type=str, default="dataset/mol_test.csv")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    model = torch.load("model.pt")
    parameters = torch.load("auxiliary.pt")
    model.eval()
    test = CNSDataset(
        args.data_file, transform=DescriptorGenerator(AVAILABLE_DESCRIPTORS)
    )
    test.normalize(parameters["max"], parameters["min"])

    result = model(test._processed_data).cpu().detach().numpy()
    result[result < 0.5] = 0
    result[result >= 0.5] = 1
    result = result.astype(int)
    df = pd.DataFrame({"SMILES": test.raw_data["SMILES"], "TARGET": result.reshape(-1)})
    df.to_csv("submission.csv", index=False)
