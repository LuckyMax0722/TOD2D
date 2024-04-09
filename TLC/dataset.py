from PIL import Image
from torch.utils.data import Dataset

from lib.config import CONF


class TrafficLightDataset(Dataset):
    def __init__(self, txt_file, transform=None):
        self.data = self.load_data(txt_file)
        self.transform = transform

    def load_data(self, data_path):
        data = []
        with open(data_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split(' ')
                image_path = parts[0]
                color_label = int(parts[1])
                direction_label = int(parts[2])
                data.append((image_path, color_label, direction_label))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, color_label, direction_label = self.data[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, color_label, direction_label


if __name__ == '__main__':
    D = TrafficLightDataset(CONF.dataset_tlc_classifier.labels_txt_path)
    img, color_label, direction_label = D[0]
    img.show()
    print(color_label, direction_label)
