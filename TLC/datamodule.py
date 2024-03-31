from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import pytorch_lightning as pl

from dataset import TrafficLightDataset
from lib.config import CONF


class TrafficLightModule(pl.LightningDataModule):
    def __init__(self, txt_file, batch_size=64):
        super().__init__()
        self.batch_size = batch_size
        self.txt_file = txt_file
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def setup(self, stage=None):
        dataset = TrafficLightDataset(self.txt_file, transform=self.transform)
        train_size = int(CONF.datamodule_tlc_classifier.split_ratio * len(dataset))
        val_size = len(dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(dataset, [train_size, val_size])

        num_train = len(self.train_dataset)
        num_val = len(self.val_dataset)
        print(f"Training samples: {num_train}  Validation samples: {num_val}")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=32, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=32, pin_memory=True)


