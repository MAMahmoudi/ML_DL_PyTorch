import torch
import  torch.nn as nn

from tqdm import tqdm

def evaluate(Data_Loader, Model, Device):
    # Put model in evaluation mode
    Model.eval()
    Final_Outputs = []
    Final_Labels = []

    with torch.no_grad():
        for Data in Data_Loader:
            Inputs = Data["Images"]
            Labels = Data["Labels"]
            Inputs = Inputs.to(Device, dtype=torch.float)
            Labels = Labels.to(Device, dtype=torch.float)

            Outputs = Model(Inputs)

            # Convert outputs and labels to lists
            Outputs = Outputs.detach().cpu().numpy().tolist()
            Labels = Labels.detach().cpu().numpy().tolist()
            Final_Outputs.extend(Outputs)
            Final_Labels.extend(Labels)
        return Final_Outputs, Final_Labels