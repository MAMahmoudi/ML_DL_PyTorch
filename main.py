import os
import numpy as np
import pandas as pd
import albumentations
import torch
from sklearn import metrics
from sklearn.model_selection import train_test_split

from src.data import Classification_Dataset
from src.models import train_model
from src.models import predict_model
from src.models.AlexNet import AlexNet

if __name__ == "__main__":
    Data_Path = "C:/Users/Mahmoudi/Documents/ENSIA/My_Projects/DL/Learning_Computer_Vision/data/external/ISIC 2024 - Skin Cancer Detection with 3D-TBP/isic-2024-challenge/train-image/image"
    DataSets = {
        "isic-2024-challenge" : "isic-2024-challenge/train-image/image",

    }
    Device = "cuda"
    epochs = 10
"""
    DataFrame = pd.read_csv("C:/Users/Mahmoudi/Documents/ENSIA/My_Projects/DL/Learning_Computer_Vision/data/external/ISIC 2024 - Skin Cancer Detection with 3D-TBP/isic-2024-challenge/train-metadata.csv", low_memory=False)
    Images = DataFrame.isic_id.values.tolist()
    Images = [os.path.join(Data_Path, Image_Name + ".jpg") for Image_Name in Images]
    #DataFrame['target'] = DataFrame['target'].astype('str')
    Labels = DataFrame.target.values
    AlexNet = AlexNet()
    AlexNet.to(Device)
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    Augmentation = albumentations.Compose([albumentations.Normalize(mean, std, max_pixel_value=255.0, always_apply=True)])
    Train_Images, Valid_Images, Train_Labels, Valid_Labels = train_test_split(Images, Labels, stratify=Labels, random_state=42)
    Train_Dataset = make_dataset.Classification_Dataset(Images_Path=Train_Images, Labels=Train_Labels, Resize=(227,227), Augmentations=Augmentation)
    Train_Loader = torch.utils.data.DataLoader(Train_Dataset, batch_size= 16, shuffle=True, num_workers=4)
    Valid_Dataset = make_dataset.Classification_Dataset(Images_Path=Valid_Images, Labels=Valid_Labels, Resize=(227, 227), Augmentations=Augmentation)
    Valid_Loader = torch.utils.data.DataLoader(Valid_Dataset, batch_size=16, shuffle=False, num_workers=4)
    Optimizer = torch.optim.Adam(AlexNet.parameters(), lr=5e-4)

    for epoch in range(epochs):
        train_model.train(Train_Loader, AlexNet, Optimizer, Device=Device)
        Predictions, Valid_Labels = predict_model.evaluate(Valid_Loader, AlexNet, Device=Device)
        roc_auc = metrics.roc_auc_score(Valid_Labels, Predictions)
        print(f"Epoch: {epoch + 1}, Valid ROC AUC={roc_auc}")
"""