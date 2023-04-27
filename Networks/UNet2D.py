import torch 
import torch.nn as nn 
from collections import OrderedDict
#This is a simplified version, I should try both the double blocks and the non double blocks
class UNet(nn.Module):
    #Max pool changes (N,c,h,w) into (N,c/x,h,w) where x is the size of the pool. 
    #all other UNet implementations seem to do a double conv block? 

    def __init__(self, in_channels=1, init_features=4, out_channels=5):
        super(UNet, self).__init__()
        features = init_features
        
        #encoders
        self.encoder1 = UNet._block(in_channels, features, name = "conv1")
        self.pool1 = nn.MaxPool2d(2, stride = 2) 
        self.encoder2 = UNet._block(features, 2*features, name = "conv2")
        self.pool2 = nn.MaxPool2d(2, stride = 2) 
        self.encoder3 = UNet._block(2*features, 4*features, name = "conv2")
        self.pool3 = nn.MaxPool2d(2, stride = 2) 

        self.bottleneck = UNet._block(4*features, 8*features, name = "bottleneck")

        #decoders
        self.upconv3 = nn.ConvTranspose2d(8*features, 4*features, kernel_size=(2,2), stride = (2,2))
        self.decoder3 = UNet._block(8*features, 4*features, name = "decoder2")
        self.upconv2 = nn.ConvTranspose2d(4*features, 2*features, kernel_size=(2,2), stride = (2,2))
        self.decoder2 = UNet._block(4*features, 2*features, name = "decoder2")
        self.upconv1 = nn.ConvTranspose2d(2*features, features, kernel_size=(2,2), stride = (2,2)) 
        self.decoder1 = UNet._block(2*features, features, name = "decoder1")

        #classifier
        self.conv = nn.Conv2d(features, out_channels, kernel_size=1, padding = "same")
        self.activation = nn.Softmax(dim=1)


    def forward(self, x):
      x = x.float() 
      
      enc4x128x128 = self.encoder1(x)
      # print("enc4x128x128", enc4x128x128.shape)

      enc8x64x64 = self.encoder2(self.pool1(enc4x128x128))
      # print("enc8x64x64",enc8x64x64.shape)

      enc16x32x32 = self.encoder3(self.pool2(enc8x64x64))
      # print("b16x32x32",b16x32x32.shape)

      b = self.bottleneck(self.pool3(enc16x32x32))

      up = self.upconv3(b)
      dec = self.decoder3(torch.cat((enc16x32x32,up),dim=1))

      up8x64x64 = self.upconv2(dec)
      dec8x64x64 = self.decoder2(torch.cat((enc8x64x64,up8x64x64), dim = 1)) #input is 16x64x64, output is 8x64x64 again 

      up4x128x128 = self.upconv1(dec8x64x64)
      dec4x128x128 = self.decoder1(torch.cat((enc4x128x128,up4x128x128), dim = 1)) #input is 8x128x128 output is 4x128x128 again 

      preactivation = self.conv(dec4x128x128)
      # print("preactivation",preactivation.shape)

      pred = self.activation(preactivation)
      return pred
      # pred = self.activation(self.conv(dec4x128x128)) #input is 4x128x128 and output is 2x128x128

    @staticmethod
    def _block(in_channels, features, name):
      return nn.Sequential(OrderedDict([
          (name, nn.Conv2d(in_channels,out_channels = features, kernel_size=3, padding="same")), #get closer to what you'd expect when you use padding of 1 
          (name+'relu', nn.ReLU()),
          (name+'bn', nn.BatchNorm2d(features)),
          (name+'dropout', nn.Dropout(0.5))
        ]))