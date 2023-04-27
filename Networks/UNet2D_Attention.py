import torch 
import torch.nn as nn 
from collections import OrderedDict
#This is a simplified version, I should try both the double blocks and the non double blocks
class UNet2D_Attention(nn.Module):
    #Max pool changes (N,c,h,w) into (N,c/x,h,w) where x is the size of the pool. 
    #all other UNet implementations seem to do a double conv block? 

    def __init__(self, in_channels=1, init_features=4, out_channels=5):
        super(UNet2D_Attention, self).__init__()
        features = init_features
        
        #encoders
        #1->f->2f
        self.encoder1 = UNet2D_Attention._encBlock(in_channels, features, name = "enc1")
        #2f->2f
        self.pool1 = nn.MaxPool2d(2, stride = 2) 
        #2f->2f->4f
        self.encoder2 = UNet2D_Attention._encBlock(2*features, 2*features, name = "enc2")
        #4f->4f
        self.pool2 = nn.MaxPool2d(2, stride = 2)
        #4f->4f->8f
        self.encoder3 = UNet2D_Attention._encBlock(4*features, 4*features, name = "enc3")
        #8f->8f
        self.pool3 = nn.MaxPool2d(2, stride = 2)
        #8f -> 8f -> 16f
        self.bottleneck = UNet2D_Attention._encBlock(8*features, 8*features, name = "bottleneck")

        #Decoders
        self.upconv3 = nn.ConvTranspose2d(16*features, 8*features, kernel_size = (2,2), stride = (2,2))
        self.att3 = Attention_block(F_g = 8*features, F_l = 8*features, F_int = 4*features)
        self.decoder3 = UNet2D_Attention._decBlock(16*features, 8*features, name = "dec3")

        self.upconv2 = nn.ConvTranspose2d(8*features, 4*features, kernel_size = (2,2), stride = (2,2))
        self.att2 = Attention_block(F_g = 4*features, F_l = 4*features, F_int = 2*features)
        self.decoder2 = UNet2D_Attention._decBlock(8*features, 4*features, name = "dec2")

        self.upconv1 = nn.ConvTranspose2d(4*features, 2*features, kernel_size = (2,2), stride = (2,2))
        self.att1 = Attention_block(F_g = 2*features, F_l = 2*features, F_int = features)
        self.decoder1 = UNet2D_Attention._decBlock(4*features, 2*features, name = "dec1")

        #If we didn't double up on the first one and did like 1->f->f then we'd
        #end up on f channels here which might be better. Probs doesn't make a difference
        #2f -> out_channels = 5
        self.conv = nn.Conv2d(2*features, out_channels, kernel_size = 1, padding = "same") #I was using kernel size 2 in the other one 
        # self.activation = nn.Sigmoid() - this could be a reason for it only predicting 2 classes. But surely it'd go 0/1 not 1/2? not sure.
        self.activation = nn.Softmax(dim = 1) #input is batch x 5 x 40 x 128 x 128
        

    #I need to change the names so that they don't include the 40, 20, 10, 5 etc. but it should all still work the same. 
    def forward(self, x): 
        x = x.float() 

        enc2fx40x128x128 = self.encoder1(x)
        enc4fx20x64x64 = self.encoder2(self.pool1(enc2fx40x128x128))
        enc8fx10x32x32 = self.encoder3(self.pool2(enc4fx20x64x64))
        
        b16fx5x16x16 = self.bottleneck(self.pool3(enc8fx10x32x32))

        up8fx10x32x32 = self.upconv3(b16fx5x16x16)
        att3 = self.att3(g=up8fx10x32x32, x=enc8fx10x32x32)
        dec8fx10x32x32 = self.decoder3(torch.cat((att3,up8fx10x32x32), dim=1))
        
        up4fx20x64x64 = self.upconv2(dec8fx10x32x32)
        att2 = self.att2(g=up4fx20x64x64, x=enc4fx20x64x64)
        dec4fx20x64x64 = self.decoder2(torch.cat((att2, up4fx20x64x64), dim=1))
        
        up2fx40x128x128 = self.upconv1(dec4fx20x64x64)
        att1 = self.att1(g=up2fx40x128x128, x=enc2fx40x128x128)
        dec2fx40x128x128 = self.decoder1(torch.cat((att1, up2fx40x128x128), dim=1))

        preactivation = self.conv(dec2fx40x128x128) #5 x 40 x 128 x 128 
        pred = self.activation(preactivation) 
        return pred 

    # in_channels -> features -> 2*features. Typically features should equal in_channels, other than the first one.
    @staticmethod
    def _encBlock(in_channels, features, name): 
        return nn.Sequential(OrderedDict([
            (name+'1', nn.Conv2d(in_channels, out_channels = features, kernel_size=3, padding="same")),
            (name+'relu1', nn.ReLU()),
            (name+'bn1', nn.BatchNorm2d(features)),
            (name+'dropout1',nn.Dropout(0.5)),
            (name+'2', nn.Conv2d(features, out_channels = 2*features, kernel_size = 3, padding = "same")),
            (name+'relu2', nn.ReLU()),
            (name+'bn2', nn.BatchNorm2d(2*features)),
            (name+'dropout2', nn.Dropout(0.5))
        ]))

    # in_channels -> features -> features so like 256->128->128. in_channels should be double features.
    @staticmethod
    def _decBlock(in_channels, features, name): 
        return nn.Sequential(OrderedDict([
            (name+'1', nn.Conv2d(in_channels, out_channels = features, kernel_size=3, padding="same")),
            (name+'relu1', nn.ReLU()),
            (name+'bn1', nn.BatchNorm2d(features)),
            (name+'dropout1',nn.Dropout(0.5)),
            (name+'2', nn.Conv2d(features, out_channels = features, kernel_size = 3, padding = "same")),
            (name+'relu2', nn.ReLU()),
            (name+'bn2', nn.BatchNorm2d(features)),
            (name+'dropout2', nn.Dropout(0.5))
        ]))

class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
            )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi  