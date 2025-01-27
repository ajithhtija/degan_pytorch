import os
import numpy as np
import math
from tqdm import tqdm
import cv2
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from utils import *
import subprocess
from models.models import *

'''# Define the generator model
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # Define the U-Net-like architecture here
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
            # Add more layers as needed
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Define the discriminator model
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

# Dataset class
class ImageDataset(Dataset):
    def __init__(self, deg_path, clean_path, transform=None):
        self.deg_images = sorted(os.listdir(deg_path))
        self.clean_images = sorted(os.listdir(clean_path))
        self.deg_path = deg_path
        self.clean_path = clean_path
        self.transform = transform

    def __len__(self):
        return len(self.deg_images)

    def __getitem__(self, idx):
        deg_image = Image.open(os.path.join(self.deg_path, self.deg_images[idx])).convert('L')
        clean_image = Image.open(os.path.join(self.clean_path, self.clean_images[idx])).convert('L')

        if self.transform:
            deg_image = self.transform(deg_image)
            clean_image = self.transform(clean_image)

        # gen patches
        deg_image, clean_image = getPatches(deg_image, clean_image, mystride=48)
        
        return deg_image, clean_image


# Training function
def train(generator, discriminator, dataloader, epochs, device, start):
    # criterion_gan = nn.MSELoss()
    criterion_gan = nn.BCEWithLogitsLoss()
    criterion_pixelwise = nn.MSELoss()
    # criterion_pixelwise = nn.L1Loss()

    optimizer_g = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    for epoch in range(start + 1, epochs + 1):
        for deg_images, clean_images in tqdm(dataloader):
            deg_images, clean_images = deg_images.to(device), clean_images.to(device)
            
            # print(deg_images.shape, clean_images.shape)

            deg_images = torch.squeeze(deg_images, dim = 0)
            clean_images = torch.squeeze(clean_images, dim = 0)
            
            # print(deg_images.shape, clean_images.shape)
            # Train discriminator
            valid = torch.ones((deg_images.size(0), 1, 16, 16), device=device)
            fake = torch.zeros((deg_images.size(0), 1, 16, 16), device=device)

            optimizer_d.zero_grad()
            # print(deg_images.shape, '????')
            gen_images = generator(deg_images)
            
            real_loss = criterion_gan(discriminator(torch.cat((clean_images, deg_images), 1)), valid)
            fake_loss = criterion_gan(discriminator(torch.cat((gen_images.detach(), deg_images), 1)), fake)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_d.step()

            # Train generator
            optimizer_g.zero_grad()

            g_loss_gan = criterion_gan(discriminator(torch.cat((gen_images, deg_images), 1)), valid)
            g_loss_pixel = criterion_gan(gen_images, clean_images)
            g_loss = g_loss_gan + g_loss_pixel
            g_loss.backward()
            optimizer_g.step()

        print(f"Epoch [{epoch}/{epochs}] D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")


        '''
        PREDICTION---
        
        '''

        if epoch%20 == 0:
            with open('train_info.txt', 'a') as f:
                f.writelines(f"Epoch: {epoch} D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}\n\n")
            
            torch.save(generator, f'trained_weights/generator_{epoch}.pth')
            torch.save(discriminator, f'trained_weights/discriminator_{epoch}.pth')
            
            test_images = sorted(os.listdir('2018/original'))

            with torch.inference_mode():
                psnr_values= []
                for i in test_images:
                    com = ['python','enhance.py','binarize',f'2018/original/{i}',f'2018/predicted/epoch_{epoch}_{i}', f'{epoch}']
                    subprocess.run(com)
                    original  = cv2.imread(f'2018/gt/{i}', cv2.IMREAD_GRAYSCALE)
                    predicted = cv2.imread(f'2018/predicted/epoch_{epoch}_{i}', cv2.IMREAD_GRAYSCALE)

                    psnr_values.append(psnr(original,predicted))
            print(f"till {epoch} epoches : {np.mean(psnr_values)}\n")
     


# Initialize models and training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



paths = sorted(Path('trained_weights/').iterdir(), key=os.path.getmtime, reverse = True)
if len(paths) > 0:
    paths[0] = str(paths[0]).replace("\\", '/')
    name = paths[0].split('/')[1]
    ep = (name.split('_')[1]).split('.')[0]

    if os.path.exists(paths[0]):
        discriminator = torch.load(f'trained_weights/discriminator_{ep}.pth')
        generator = torch.load(f'trained_weights/generator_{ep}.pth')
        start = int(ep)
        print(f'loading saved model at epoch {ep}...')

else:
    with open('train_info.txt', 'w') as f:
        f.close()
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    start = 0
    print('starting afresh...')

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

dataset = ImageDataset(deg_path='data/A/', clean_path='data/B/', transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

train(generator, discriminator, dataloader, start = start, epochs=80, device=device)
