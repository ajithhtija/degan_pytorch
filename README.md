we are doing Image binarization using GAN where Genrator has a U-net architecture:
Here we are using DIBCO dataset for traning:
In the data Folder:
'A' contains the degraded images 
'B' contains the clean images 
this is folder is for traning where image name should be same such that degraded image in 'A' corresponds to same clean image in 'B'.
Genrator and discriminator architecture are defined in models.py 
here, as the images are less we do traning by dividing the images into patches of 1*256*256 of this size 
with the help of enhance.py, we tested the images and utils.py has the needed funtions like psnr value calculation.
