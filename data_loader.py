import torch
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl

class SWGODataModule(pl.LightningDataModule):
    def __init__(self, train_data_path, val_data_path, test_data_path=None, batch_size=512, num_workers=4):
        super().__init__()
        self.train_data_path = train_data_path
        self.val_data_path   = val_data_path
        self.test_data_path  = test_data_path
        self.batch_size      = batch_size
        self.num_workers     = num_workers

    def setup(self, stage=None):
        print("Loading pre-split datasets...")

        # Each .pt file contains a tuple (inputs, labels)
        train_inputs, train_labels = torch.load(self.train_data_path)
        val_inputs,   val_labels   = torch.load(self.val_data_path)

        print(f"Train inputs shape: {train_inputs.shape}, labels shape: {train_labels.shape}")
        print(f"Val   inputs shape: {val_inputs.shape}, labels shape: {val_labels.shape}")

        self.train_dataset = TensorDataset(train_inputs, train_labels)
        self.val_dataset   = TensorDataset(val_inputs,   val_labels)

        if self.test_data_path is not None:
            test_inputs, test_labels = torch.load(self.test_data_path)
            print(f"Test  inputs shape: {test_inputs.shape}, labels shape: {test_labels.shape}")
            self.test_dataset = TensorDataset(test_inputs, test_labels)
        else:
            self.test_dataset = None

        print(f"Data loaded: {len(self.train_dataset)} train, {len(self.val_dataset)} val"
              + (f", {len(self.test_dataset)} test" if self.test_dataset else ""))

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        if self.test_dataset is not None:
            return DataLoader(self.test_dataset, batch_size=self.batch_size,
                              shuffle=False, num_workers=self.num_workers, pin_memory=True)
        return None
