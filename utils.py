import torch
import torch.nn.functional as F
import math

def psnr(img1, img2):
    """
    Calculate the Peak Signal-to-Noise Ratio (PSNR) between two images.
    img1, img2: PyTorch tensors with values normalized between 0 and 1.
    """
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 1.0
    return 20 * math.log10(PIXEL_MAX / torch.sqrt(mse))


def split2(dataset, size, h, w):
    """
    Split a batch of images into smaller patches of size 256x256.
    dataset: PyTorch tensor of shape (batch_size, channels, height, width).
    size: Batch size.
    h, w: Dimensions of the input images.
    """
    nsize1, nsize2 = 256, 256
    patches = []

    for i in range(size):
        img = dataset[i]
        for ii in range(0, h, nsize1):
            for iii in range(0, w, nsize2):
                patch = img[:, ii:ii + nsize1, iii:iii + nsize2]
                patches.append(patch)

    return torch.stack(patches)


def merge_image2(splitted_images, h, w):
    """
    Merge a list of patches back into a full image.
    splitted_images: PyTorch tensor of shape (num_patches, channels, 256, 256).
    h, w: Original dimensions of the image to reconstruct.
    """
    image = torch.zeros((1, h, w), device=splitted_images.device)
    nsize1, nsize2 = 256, 256
    ind = 0

    for ii in range(0, h, nsize1):
        for iii in range(0, w, nsize2):
            image[:, ii:ii + nsize1, iii:iii + nsize2] = splitted_images[ind]
            ind += 1

    return image


def getPatches(watermarked_image, clean_image, mystride):
    """
    Extract patches from watermarked and clean images with a stride.
    watermarked_image, clean_image: PyTorch tensors of shape (channels, height, width).
    mystride: Stride size for extracting patches.
    """
    nsize = 256
    watermarked_patches = []
    clean_patches = []

    # Pad the watermarked image
    h = ((watermarked_image.shape[1] // nsize) + 1) * nsize
    w = ((watermarked_image.shape[2] // nsize) + 1) * nsize
    padded_watermarked = F.pad(watermarked_image, (0, w - watermarked_image.shape[2], 0, h - watermarked_image.shape[1]))

    for j in range(0, h - nsize, mystride):
        for k in range(0, w - nsize, mystride):
            patch = padded_watermarked[:, j:j + nsize, k:k + nsize]
            watermarked_patches.append(patch)

    # Pad the clean image
    h = ((clean_image.shape[1] // nsize) + 1) * nsize
    w = ((clean_image.shape[2] // nsize) + 1) * nsize
    padded_clean = F.pad(clean_image, (0, w - clean_image.shape[2], 0, h - clean_image.shape[1]), value=1.0)

    for j in range(0, h - nsize, mystride):
        for k in range(0, w - nsize, mystride):
            patch = padded_clean[:, j:j + nsize, k:k + nsize]
            clean_patches.append(patch)

    return torch.stack(watermarked_patches), torch.stack(clean_patches)
