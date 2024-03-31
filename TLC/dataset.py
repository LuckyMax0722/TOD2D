from PIL import Image
from torch.utils.data import Dataset

from lib.config import CONF


class TrafficLightDataset(Dataset):
    def __init__(self, txt_file, transform=None):
        self.data = []
        self.transform = transform

        with open(txt_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                path, label = line.strip().split()
                self.data.append((path, int(label)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label


if __name__ == '__main__':
    D = TrafficLightDataset(CONF.dataset_tlc_classifier.labels_txt_path)
    img, label = D[0]
    img.show()
