import os
import logging
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data.dataset import Dataset

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

class CovidDataset(Dataset):
    def __init__(self, path, transform=None, limit=None):
        self.path = path
        self.limit = limit
        self.images, self.masks = self.read_dataset(path)
        self.transform = transform
        self.logger = logging.getLogger(__name__)
        if limit is not None:
            self.images = self.images[:limit]
            self.masks = self.masks[:limit]
        self.logger.info(f"Dataset loaded with {len(self.images)} samples.")

    def read_dataset(self, path):
        imgs, masks = [], []
        with open(path, 'r') as f:
            for line in f:
                img_file_name = line.rstrip()
                img_file_path = f'MosMedData/slices/imgs/{img_file_name}'
                msk_file_path = f'MosMedData/slices/masks/{img_file_name}'
                if os.path.exists(img_file_path) and os.path.exists(msk_file_path):
                    imgs.append(np.load(img_file_path))
                    masks.append(np.load(msk_file_path))
        f.close()
        return imgs, masks

    def __len__(self):
        return min(len(self.images), self.limit) if self.limit else len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx][None, ...] / 255.0
        mask = self.masks[idx][None, ...]
        mask = np.clip(mask, 0.0, 1.0)

        return torch.FloatTensor(image), torch.FloatTensor(mask)