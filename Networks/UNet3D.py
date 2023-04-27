import torch 
import torch.nn as nn 
from collections import OrderedDict
class UNet3D(nn.Module):
    def __init__(self, in_channels=1, init_features=4, out_channels=5):
        super(UNet3D, self).__init__()
        features = init_features 
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #encoders
        #1->f->2f
        self.encoder1 = UNet3D._encBlock(in_channels, features, name = "enc1")
        #2f->2f
        self.pool1 = nn.MaxPool3d(2, stride = 2) 
        #2f->2f->4f
        self.encoder2 = UNet3D._encBlock(2*features, 2*features, name = "enc2")
        #4f->4f
        self.pool2 = nn.MaxPool3d(2, stride = 2)
        #4f->4f->8f
        self.encoder3 = UNet3D._encBlock(4*features, 4*features, name = "enc3")
        #8f->8f
        self.pool3 = nn.MaxPool3d(2, stride = 2)
        #8f -> 8f -> 16f
        self.bottleneck = UNet3D._encBlock(8*features, 8*features, name = "bottleneck")

        #Decoders
        #16f -> 8f
        self.upconv3 = nn.ConvTranspose3d(16*features, 8*features, kernel_size = (2,2,2), stride = (2,2,2))
        #combine the 2 8f's, then 16f -> 8f -> 8f
        self.decoder3 = UNet3D._decBlock(16*features, 8*features, name = "dec3")
        #8f -> 4f
        self.upconv2 = nn.ConvTranspose3d(8*features, 4*features, kernel_size = (2,2,2), stride = (2,2,2))
        #combine the 2 4f's then 8f -> 4f -> 4f
        self.decoder2 = UNet3D._decBlock(8*features, 4*features, name = "dec2")
        #4f -> 2f 
        self.upconv1 = nn.ConvTranspose3d(4*features, 2*features, kernel_size = (2,2,2), stride = (2,2,2))
        #combine the 2 2f's then 4f -> 2f -> 2f 
        self.decoder1 = UNet3D._decBlock(4*features, 2*features, name = "dec1")
        #2f -> out_channels = 5
        self.conv = nn.Conv3d(2*features, out_channels, kernel_size = 1, padding = "same") 
        self.activation = nn.Softmax(dim = 1) #input is batch x 5 x 40 x 128 x 128
        
    def forward(self, x): 
        x = x.float()#.to(self.device)
        # print("x", x.shape)
        enc2fx40x128x128 = self.encoder1(x)
        # print("enc2fx40x128x128", enc2fx40x128x128.shape)
        enc4fx20x64x64 = self.encoder2(self.pool1(enc2fx40x128x128))
        # print("enc4fx20x64x64", enc4fx20x64x64.shape)
        enc8fx10x32x32 = self.encoder3(self.pool2(enc4fx20x64x64))
        # print("enc8fx10x32x32", enc8fx10x32x32.shape)
        
        b16fx5x16x16 = self.bottleneck(self.pool3(enc8fx10x32x32))
        # print("b16fx5x16x16", b16fx5x16x16.shape)

        up8fx10x32x32 = self.upconv3(b16fx5x16x16)
        # print("up8fx10x32x32", up8fx10x32x32.shape)
        # print("catted shape: ", torch.cat((enc8fx10x32x32,up8fx10x32x32)).shape)
        dec8fx10x32x32 = self.decoder3(torch.cat((enc8fx10x32x32,up8fx10x32x32), dim=1))
        # print("dec8fx10x32x32", dec8fx10x32x32.shape)
        up4fx20x64x64 = self.upconv2(dec8fx10x32x32)
        # print("up4fx20x64x64", up4fx20x64x64.shape)
        dec4fx20x64x64 = self.decoder2(torch.cat((enc4fx20x64x64, up4fx20x64x64), dim=1))
        # print("dec4fx20x64x64", dec4fx20x64x64.shape)
        up2fx40x128x128 = self.upconv1(dec4fx20x64x64)
        # print("up2fx40x128x128", up2fx40x128x128.shape)
        dec2fx40x128x128 = self.decoder1(torch.cat((enc2fx40x128x128, up2fx40x128x128), dim=1))
        # print("dec2fx40x128x128", dec2fx40x128x128.shape)

        preactivation = self.conv(dec2fx40x128x128) #5 x 40 x 128 x 128 
        pred = self.activation(preactivation) 
        return pred 




    # in_channels -> features -> 2*features. Typically features should equal in_channels, other than the first one.
    @staticmethod
    def _encBlock(in_channels, features, name): 
        return nn.Sequential(OrderedDict([
            (name+'1', nn.Conv3d(in_channels, out_channels = 2*features, kernel_size=3, padding="same")),
            (name+'relu1', nn.ReLU()),
            (name+'bn1', nn.BatchNorm3d(2*features)),
            (name+'dropout1',nn.Dropout(0.5))
            (name+'2', nn.Conv3d(features, out_channels = 2*features, kernel_size = 3, padding = "same")),
            (name+'relu2', nn.ReLU()),
            (name+'bn2', nn.BatchNorm3d(2*features)),
            (name+'dropout2', nn.Dropout(0.5))
        ]))

    # in_channels -> features -> features so like 256->128->128. in_channels should be double features.
    @staticmethod
    def _decBlock(in_channels, features, name): 
        return nn.Sequential(OrderedDict([
            (name+'1', nn.Conv3d(in_channels, out_channels = features, kernel_size=3, padding="same")),
            (name+'relu1', nn.ReLU()),
            (name+'bn1', nn.BatchNorm3d(features)),
            (name+'dropout1',nn.Dropout(0.5))
            (name+'2', nn.Conv3d(features, out_channels = features, kernel_size = 3, padding = "same")),
            (name+'relu2', nn.ReLU()),
            (name+'bn2', nn.BatchNorm3d(features)),
            (name+'dropout2', nn.Dropout(0.5))
        ]))
