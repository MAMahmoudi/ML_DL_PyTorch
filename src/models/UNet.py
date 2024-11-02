import torch
import torch.nn as nn


def Double_Conv(Input_Channels, Output_Channels):
    Conv = nn.Sequential(
        nn.Conv2d(Input_Channels, Output_Channels, kernel_size=3),
        nn.ReLU(inplace=True),
        nn.Conv2d(Output_Channels, Output_Channels, kernel_size=3),
        nn.ReLU(inplace=True),
    )
    return Conv

def Crop_Image(Tensor, Target_Tensor):
    Target_Size = Target_Tensor.size()[2]
    Tensor_Size = Tensor.size()[2]
    Delta = Tensor_Size - Target_Size
    Delta = Delta // 2
    return  Tensor[:, :, Delta:Tensor_Size - Delta, Delta: Tensor_Size - Delta]

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Down_Conv_1 = Double_Conv(1, 64)
        self.Down_Conv_2 = Double_Conv(64, 128)
        self.Down_Conv_3 = Double_Conv(128, 256)
        self.Down_Conv_4 = Double_Conv(256, 512)
        self.Down_Conv_5 = Double_Conv(512, 1024)

        self.Conv_Trans_1 = nn.ConvTranspose2d(in_channels=1024,out_channels=512, kernel_size=2, stride=2)
        self.Up_Conv_1 = Double_Conv(1024, 512)
        self.Conv_Trans_2 = nn.ConvTranspose2d(in_channels=512,out_channels=256, kernel_size=2, stride=2)
        self.Up_Conv_2 = Double_Conv(512, 256)
        self.Conv_Trans_3 = nn.ConvTranspose2d(in_channels=256,out_channels=128, kernel_size=2, stride=2)
        self.Up_Conv_3 = Double_Conv(256, 128)
        self.Conv_Trans_4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.Up_Conv_4 = Double_Conv(128, 64)
        self.Output = nn.Conv2d(in_channels=64, out_channels=2,kernel_size=1)


    def forward(self, Image):
        # The way down
        x1 = self.Down_Conv_1(Image)
        x2 = self.max_pool_2x2(x1)
        x3 = self.Down_Conv_2(x2)
        x4 = self.max_pool_2x2(x3)
        x5 = self.Down_Conv_3(x4)
        x6 = self.max_pool_2x2(x5)
        x7 = self.Down_Conv_4(x6)
        x8 = self.max_pool_2x2(x7)
        x9 = self.Down_Conv_5(x8)
        # The way up
        x = self.Conv_Trans_1(x9)
        y = Crop_Image(x7, x)
        x = self.Up_Conv_1(torch.cat([x, y], 1))
        x = self.Conv_Trans_2(x)
        y = Crop_Image(x5, x)
        x = self.Up_Conv_2(torch.cat([x, y], 1))
        x = self.Conv_Trans_3(x)
        y = Crop_Image(x3, x)
        x = self.Up_Conv_3(torch.cat([x, y], 1))
        x = self.Conv_Trans_4(x)
        y = Crop_Image(x1, x)
        x = self.Up_Conv_4(torch.cat([x, y], 1))
        x = self.Output(x)
        print(x.size())

if __name__ == "__main__":
    Image = torch.rand((1,1,572,572))
    model = UNet()
    model.forward(Image)
