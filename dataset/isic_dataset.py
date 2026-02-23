import os
import torch
from torch.utils.data import Dataset
import cv2
from typing import Optional, Callable

class ISICDataset(Dataset):
    def __init__(self, root_dir: str, img_fdr: str, mask_fdr: str, transform: Optional[Callable]=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_dir = os.path.join(root_dir, img_fdr)
        self.mask_dir = os.path.join(root_dir, mask_fdr)

        self.images = os.listdir(self.image_dir)
        self.masks = os.listdir(self.mask_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_name = self.images[index]
        img_path = os.path.join(self.image_dir, img_name)
        mask_name = self.masks[index]
        mask_path = os.path.join(self.mask_dir, mask_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, 0)

        if self.transform:
            image = self.transform(image)
            mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
        return image, mask

