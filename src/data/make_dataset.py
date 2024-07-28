import torch
import numpy as np
from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True # Takes care of corrupt images

class Classification_Dataset:
    def __init__(self, Images_Path, Labels, Resize=None, Augmentations=None):
        self.Images_Path = Images_Path
        self.Labels = Labels
        self.Resize = Resize
        self.Augmentations = Augmentations

    def __len__(self):
        return len(self.Images_Path)

    def __getitem__(self, item):
        IMG = Image.open(self.Images_Path[item])
        IMG = IMG.convert("RGB")
        Labels = self.Labels[item]

        # If resize is set
        if self.Resize is not None:
            IMG = IMG.resize((self.Resize[1], self.Resize[0]), resample=Image.BILINEAR)

        IMG = np.array(IMG)

        if self.Augmentations is not None:
            Augmented = self.Augmentations(image=IMG)
            IMG = Augmented["image"]

        # Pytorch expects CHW instead of HWC
        IMG = np.transpose(IMG, (2, 0, 1)).astype(np.float32)
        return {
            "Images": torch.tensor(IMG, dtype=torch.float),
            "Labels": torch.tensor(Labels, dtype=torch.long),
        }

