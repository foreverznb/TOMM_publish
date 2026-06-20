import os
import sys
package_path = os.path.abspath('./')
if package_path not in sys.path:
    sys.path.append(package_path)
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from utils import *
import matplotlib.pyplot as plt
import numpy as np
from torchmetrics.image import StructuralSimilarityIndexMeasure
import zlib
import pickle
import cv2
import encoder_fr

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(3407)


if __name__ == '__main__':
    # Data augmentation and normalization for training
    transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
    ])

    # Create the dataset
    img_dir = './azuki_original'
    dataset = CustomImageDataset(img_dir=img_dir, transform=transform, img_index_json='./img_index.json')

    # Create DataLoaders
    test_loader = DataLoader(dataset, batch_size=50, shuffle=False, num_workers=16)

    # Define constants
    NUM_FILES = 10000  # Number of unique file names
    EMBEDDING_DIM = 64  # Dimension for file name embeddings
    LATENT_SHAPE = (8, 64, 64)  # Shape of the latent representation for the decoder
    IMG_SHAPE = (3, 512, 512)  # Output image shape (C, H, W)

    # Define loss function 
    criterion = nn.MSELoss()
    ssim = StructuralSimilarityIndexMeasure().to(device)

    # Save compressed data to a binary file
    compressed_model_save_path = "./state_dict_compressed.bin"
    compressed_model_size = os.path.getsize(compressed_model_save_path) / 1048576 #* 1024
    print(f"The size of the compressed model is {compressed_model_size:.4f} MB.")

    # decompress the model
    with open(compressed_model_save_path, "rb") as f:
        compressed_data = f.read()
    decompressed_data = zlib.decompress(compressed_data)
    decompressed_data = pickle.loads(decompressed_data)

    model = encoder_fr.FileNameToImage(num_files=NUM_FILES, embedding_dim=EMBEDDING_DIM, latent_shape=LATENT_SHAPE, img_shape=IMG_SHAPE).to(device)
    cleanNet(model)
    model_dict = model.state_dict()

    # Load the decompressed data
    for key in model_dict.keys():
        model_dict[key] = decompressed_data.pop(0)

    model.load_state_dict(model_dict)

    totalnum, totalbias, totalzeros = countParams(model)
    print(f'Total number of weights: {totalnum}')
    print(f'Total number of biases: {totalbias}')
    print(f'Total number of zeros: {totalzeros}')
    print(f'Total number of parameters: {totalnum + totalbias}')

    model_no_fr = encoder_fr.FileNameToImage_no_fr(num_files=NUM_FILES, embedding_dim=EMBEDDING_DIM, latent_shape=LATENT_SHAPE, img_shape=IMG_SHAPE).to(device)
    dict_no_fr = model_no_fr.state_dict()
    dict_fr = model.state_dict()
    dict_no_fr = upload_dict(dict_fr, dict_no_fr)
    model_no_fr.load_state_dict(dict_no_fr)
 
    loss, output_imgs, ssim_score, output_imgs_name = evaluate(model_no_fr, ssim, test_loader, criterion, device)
    psnr = calculate_psnr(loss)
    print(f'Test PSNR: {psnr:.2f}')
    print(f'Test SSIM: {ssim_score:.4f}')
    print('Saving images...')
    if not os.path.exists('./generated_imgs'):
        os.makedirs('./generated_imgs')
    for j in range(len(output_imgs)):        
        img = np.clip(cv2.resize(output_imgs[j], (512, 512), interpolation=cv2.INTER_CUBIC), 0, 1)
        plt.imsave(f'./generated_imgs/{output_imgs_name[j]}.png', img)
    print('Images saved.')
