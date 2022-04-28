import torch
import torch.nn as nn

class Unet_Encoder_Block(nn.Module):

    def __init__(self, in_channels, features):
        super(Unet_Encoder_Block, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, features, padding=1, kernel_size=3)
        self.relu_1  = nn.ReLU()
        self.conv2 = nn.Conv2d(features, features, padding=1, kernel_size=3)
        self.relu_2  = nn.ReLU()
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.relu_1(out)
        out = self.conv2(out)
        out = self.relu_2(out)
        return out


class Unet(nn.Module):
    def __init__(self, in_channels, out_classes, features=16):
        super(Unet, self).__init__()

        # Encoder 
        self.encoder_1 = Unet_Encoder_Block(in_channels, features)
        self.pool_1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder_2 = Unet_Encoder_Block(features, features * 2)
        self.pool_2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder_4 = Unet_Encoder_Block(features * 2, features * 4)
        self.pool_4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder_8 = Unet_Encoder_Block(features * 4, features * 8)
        self.pool_8 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder_16 = Unet_Encoder_Block(features * 8, features * 16)
       

        # Decoder 
        self.upconv_8 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder_8 = Unet_Encoder_Block((features * 8) * 2, features * 8)
       
        self.upconv_4 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder_4 = Unet_Encoder_Block((features * 4) * 2, features * 4)
     
        self.upconv_2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder_2 = Unet_Encoder_Block((features * 2) * 2, features * 2)
       
        self.upconv_1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.decoder_1 = Unet_Encoder_Block(features * 2, features)
        
        
        # Classifier 
        self.classifier = nn.Conv2d(features, out_classes, kernel_size=1)
        # self.softmax = nn.Softmax()

    def forward(self, x):

        # Encoder 
        encoder_1 = self.encoder_1(x)
        pool_1 = self.pool_1(encoder_1)

        encoder_2 = self.encoder_2(pool_1)
        pool_2 = self.pool_2(encoder_2)

        encoder_4 = self.encoder_4(pool_2)
        pool_4 = self.pool_4(encoder_4)

        encoder_8 = self.encoder_8(pool_4)
        pool_8 = self.pool_8(encoder_8)

        encoder_16 = self.encoder_16(pool_8)

        # Decoder
        decoder_8 = self.upconv_8(encoder_16)

        decoder_8 = torch.cat((decoder_8, encoder_8), dim=1)
        decoder_8 = self.decoder_8(decoder_8)
        
        decoder_4 = self.upconv_4(decoder_8)
        decoder_4 = torch.cat((decoder_4, encoder_4), dim=1)
        decoder_4 = self.decoder_4(decoder_4)
        
        decoder_2 = self.upconv_2(decoder_4)
        decoder_2 = torch.cat((decoder_2, encoder_2), dim=1)
        decoder_2 = self.decoder_2(decoder_2)
        
        decoder_1 = self.upconv_1(decoder_2)
        decoder_1 = torch.cat((decoder_1, encoder_1), dim=1)
        decoder_1 = self.decoder_1(decoder_1)

        # Classifier
        classifier = self.classifier(decoder_1)
        # classifier = self.softmax(classifier)
        
        return classifier

if __name__ == "__main__":
    block = Unet_Encoder_Block(3, 16)
    x = torch.randn(1, 3, 64, 64)
    print(block(x).shape)
