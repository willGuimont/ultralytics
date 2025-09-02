import pathlib

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class SilvaDataloader(Dataset):
    def __init__(self, root):
        self.root = pathlib.Path(root) / 'data/tif-8'
        self.imgs = sorted(self.root.rglob('*.tif'))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, item):
        return Image.open(self.imgs[item])
