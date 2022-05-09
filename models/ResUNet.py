import torch
import torch.nn as nn

class ResUNetEncoderBlock(nn.Module):

    def __init__(self, in_channels, features, kernel_size=3):
        super(ResUNetEncoderBlock, self).__init__()

        self.convPath1 = nn.Conv2d(in_channels, features, padding='same', kernel_size=kernel_size)
        self.batchNormPath1 = nn.BatchNorm2d(features)
        self.reluPath1 = nn.ReLU()

        self.convPath2 = nn.Conv2d(features, features, padding='same', kernel_size=kernel_size)
        self.batchNormPath2 = nn.BatchNorm2d(features)

        self.convShortcut = nn.Conv2d(in_channels, features, padding='same', kernel_size=1)
        self.batchNormShortcut = nn.BatchNorm2d(features)

        self.reluAddition = nn.ReLU()

    def forward(self, x):

        path = self.convPath1(x)
        path = self.batchNormPath1(path)
        path = self.reluPath1(path)

        path = self.convPath2(path)
        path = self.batchNormPath2(path)

        shortcut = self.convShortcut(x)
        shortcut = self.batchNormShortcut(shortcut)

        addition = torch.cat((path, shortcut), dim=1)

        out = self.reluAddition(addition)

        return out


class ResUNet(nn.Module):
    def __init__(self, in_channels, out_classes, features=16, kernel_size=3):
        super(ResUNet, self).__init__()

        # Encoder
        self.encoder_1 = ResUNetEncoderBlock(in_channels, features, kernel_size=kernel_size)
        self.pool_1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder_2 = ResUNetEncoderBlock(features * 2, features * 2, kernel_size=kernel_size)
        self.pool_2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder_4 = ResUNetEncoderBlock(features * 2 * 2, features * 4, kernel_size=kernel_size)
        self.pool_4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder_8 = ResUNetEncoderBlock(features * 2 * 4, features * 8, kernel_size=kernel_size)
        self.pool_8 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder_16 = ResUNetEncoderBlock(features * 2 * 8, features * 16, kernel_size=kernel_size)

        # Decoder
        self.upconv_8 = nn.ConvTranspose2d(features * 2 * 16, features * 8, kernel_size=2, stride=2)
        self.decoder_8 = ResUNetEncoderBlock((features * 8) * 3, features * 8, kernel_size=kernel_size)

        self.upconv_4 = nn.ConvTranspose2d(features * 2 * 8, features * 4, kernel_size=2, stride=2)
        self.decoder_4 = ResUNetEncoderBlock((features * 4) * 3, features * 4, kernel_size=kernel_size)

        self.upconv_2 = nn.ConvTranspose2d(features * 2 * 4, features * 2, kernel_size=2, stride=2)
        self.decoder_2 = ResUNetEncoderBlock((features * 2) * 3, features * 2, kernel_size=kernel_size)

        self.upconv_1 = nn.ConvTranspose2d(features * 2 * 2, features, kernel_size=2, stride=2)
        self.decoder_1 = ResUNetEncoderBlock(features * 3, features, kernel_size=kernel_size)

        # Classifier
        self.convClassifier1 = nn.Conv2d(features * 2, out_classes * 2, padding="same", kernel_size=kernel_size)
        self.batchNormClassifier = nn.BatchNorm2d(out_classes * 2)
        self.reluClassifier = nn.ReLU()

        self.convClassifier2 = nn.Conv2d(out_classes * 2, out_classes, padding="same", kernel_size=1)

    def forward(self, x):

        # Encoder
        encoder_1 = self.encoder_1(x)
        # print(encoder_1.shape)
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
        classifier = self.convClassifier1(decoder_1)
        classifier = self.batchNormClassifier(classifier)
        classifier = self.reluClassifier(classifier)

        classifier = self.convClassifier2(classifier)

        return classifier


if __name__ == "__main__":
    model = ResUNet(3, 17)
    from torchinfo import summary
    summary(model, input_size=(5, 3, 512, 512))