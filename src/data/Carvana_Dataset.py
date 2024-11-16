import os
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision import transforms

class Carvana_Dataset(Dataset):
    def __init__(self, Root_Path, Test = False):
        self.Root_Path = Root_Path
        if Test:
            self.Images = sorted([Root_Path+"/manual_test/"+i for i in os.listdir(Root_Path+"/manual_test/")])
            self.Masks = sorted([Root_Path+"/manual_test_masks/"+i for i in os.listdir(Root_Path+"/manual_test_masks/")])
        else:
            self.Images = sorted([Root_Path + "/train/" + i for i in os.listdir(Root_Path + "/train/")])
            self.Masks = sorted([Root_Path + "/train_masks/" + i for i in os.listdir(Root_Path + "/train_masks/")])
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor()])

    def __getitem__(self, index):
        image = Image.open(self.Images[index]).convert("RGB")
        mask = Image.open(self.Masks[index]).convert("L")
        return self.transform(image), self.transform(mask)

    def __len__(self):
        return len(self.Images)