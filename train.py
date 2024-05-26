import pandas as pd
import torch
import torch.nn as nn

from rich.progress import track
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader

from src.data.dataset import CNSDataset
from src.descriptors import DescriptorGenerator, AVAILABLE_DESCRIPTORS

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.random.manual_seed(0)

TRAIN_DATASET = "dataset\mol_train.csv"
TEST_DATASET = "dataset\mol_test.csv"
BATCH_SIZE = 32
TOTAL_EPOCH = 500


def train_mlp(model, loss_fn, optimizer, scheduler, train_loader, total_epoch=300):
    model.train()
    for nepoch in range(total_epoch):
        for batch in train_loader:
            y_pred = model(batch[0])
            loss = loss_fn(y_pred, batch[1].to(device).reshape(-1, 1).float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
    return model


if __name__ == "__main__":
    whole_dataset = CNSDataset(
        TRAIN_DATASET, transform=DescriptorGenerator(AVAILABLE_DESCRIPTORS)
    )
    max = torch.max(whole_dataset._processed_data, axis=0).values
    min = torch.min(whole_dataset._processed_data, axis=0).values
    whole_dataset.normalize(max, min)

    test = CNSDataset(
        TEST_DATASET, transform=DescriptorGenerator(AVAILABLE_DESCRIPTORS)
    )
    test.normalize(max, min)

    nfeatures = whole_dataset._processed_data.shape[1]

    model = nn.Sequential(
        nn.Linear(nfeatures, 3072),
        nn.LayerNorm(3072),
        nn.ReLU(),
        nn.Dropout(0.8),
        nn.Linear(3072, 2048),
        nn.LayerNorm(2048),
        nn.ReLU(),
        nn.Dropout(0.6),
        nn.Linear(2048, 1024),
        nn.LayerNorm(1024),
        nn.ReLU(),
        nn.Dropout(0.6),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, 128),
        nn.LayerNorm(128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 1),
        nn.Sigmoid(),
    )
    model.to(device)
    loss_fn = nn.BCELoss()
    loss_fn.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-3)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=1e-4,
        epochs=500,
        steps_per_epoch=int(700 / 32) + 1,
        final_div_factor=1e7,
    )

    train_loader = DataLoader(whole_dataset, batch_size=BATCH_SIZE)

    for nepoch in track(range(TOTAL_EPOCH), description="Training..."):
        for batch in train_loader:
            y_pred = model(batch[0])
            loss = loss_fn(y_pred, batch[1].to(device).reshape(-1, 1).float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

    # Inference
    model.eval()
    result = model(test._processed_data).cpu().detach().numpy()
    result[result < 0.5] = 0
    result[result >= 0.5] = 1
    result = result.astype(int)

    df = pd.DataFrame({"SMILES": test.raw_data["SMILES"], "TARGET": result.reshape(-1)})
    df.to_csv("submission.csv", index=False)
