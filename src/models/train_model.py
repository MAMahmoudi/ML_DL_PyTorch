import torch
import  torch.nn as nn

from tqdm import tqdm

def train(Data_Loader, Model, Optimizer, Device):
    # Put model in train mode
    Model.train()
    for Step, Data in tqdm(enumerate(Data_Loader), total= len(Data_Loader), desc="Processing images"):
        Inputs = Data["Images"]
        Labels = Data["Labels"]
        Inputs = Inputs.to(Device, dtype=torch.float)
        Labels = Labels.to(Device, dtype=torch.float)

        Optimizer.zero_grad()
        Outputs = Model(Inputs)
        Loss = nn.BCEWithLogitsLoss()(Outputs, Labels.view(-1, 1))
        Loss.backward()
        Optimizer.step()