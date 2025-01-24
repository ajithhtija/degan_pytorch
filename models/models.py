import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms


# Generator Model
class Generator(nn.Module):
    def __init__(self, input_channels=1, biggest_layer=512):
        super(Generator, self).__init__()

        def conv_block(in_channels, out_channels, dropout=False):
            layers = [
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            ]
            if dropout:
                layers.append(nn.Dropout(0.5))
            return nn.Sequential(*layers)

        def upconv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
                nn.ReLU(inplace=True),
            )

        self.encoder1 = conv_block(input_channels, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.encoder2 = conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.encoder3 = conv_block(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        self.encoder4 = conv_block(256, biggest_layer // 2, dropout=True)
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = conv_block(biggest_layer // 2, biggest_layer, dropout=True)

        self.upconv4 = upconv_block(biggest_layer, biggest_layer // 2)
        self.decoder4 = conv_block(biggest_layer, biggest_layer // 2)

        self.upconv3 = upconv_block(biggest_layer // 2, 256)
        self.decoder3 = conv_block(512, 256)

        self.upconv2 = upconv_block(256, 128)
        self.decoder2 = conv_block(256, 128)

        self.upconv1 = upconv_block(128, 64)
        self.decoder1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(2, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        e1 = self.encoder1(x)
        p1 = self.pool1(e1)

        e2 = self.encoder2(p1)
        p2 = self.pool2(e2)

        e3 = self.encoder3(p2)
        p3 = self.pool3(e3)

        e4 = self.encoder4(p3)
        p4 = self.pool4(e4)

        b = self.bottleneck(p4)

        d4 = self.upconv4(b)
        d4 = torch.cat((d4, e4), dim=1)
        d4 = self.decoder4(d4)

        d3 = self.upconv3(d4)
        d3 = torch.cat((d3, e3), dim=1)
        d3 = self.decoder3(d3)

        d2 = self.upconv2(d3)
        d2 = torch.cat((d2, e2), dim=1)
        d2 = self.decoder2(d2)

        d1 = self.upconv1(d2)
        d1 = torch.cat((d1, e1), dim=1)
        return self.decoder1(d1)


# Discriminator Model
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 1, kernel_size=4, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)
'''
# GAN Model
class GAN(nn.Module):
    def __init__(self, generator, discriminator):
        super(GAN, self).__init__()
        self.generator = generator
        self.discriminator = discriminator

    def forward(self, x):
        gen_output = self.generator(x)
        disc_output = self.discriminator(gen_output, x)
        return disc_output, gen_output


# Instantiate models
generator = Generator()
discriminator = Discriminator()
# gan = GAN(generator, discriminator)

# Optimizers
optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
'''