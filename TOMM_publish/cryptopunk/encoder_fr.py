import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append("../")
import frequency_regularization.freq_reg._freqreg as fr


#     # Define constants
# NUM_FILES = 1000  # Number of unique file names
# EMBEDDING_DIM = 128  # Dimension for file name embeddings
# LATENT_SHAPE = (64, 8, 8)  # Shape of the latent representation for the decoder
# IMG_SHAPE = (3, 128, 128)  # Output image shape (C, H, W)

class FileNameToImage(nn.Module):
    def __init__(self, num_files, embedding_dim, latent_shape, img_shape):
        super(FileNameToImage, self).__init__()
        self.embedding_dim = embedding_dim
        self.latent_shape = latent_shape
        self.img_shape = img_shape
        self.minrate = 0.3
        
        # Embedding layer to map file name indices to dense vectors
        self.embedding = nn.Embedding(num_files, embedding_dim)
        
        # Fully connected layer to reshape embedding into latent tensor
        self.fc1 = fr.Linear(embedding_dim, latent_shape[0] * latent_shape[1] * latent_shape[2], bias=False, minrate=0.5)
        self.decoder2 = nn.Sequential(
            fr.ConvTranspose2d(latent_shape[0], 32, kernel_size=3, stride=1, padding=1, bias=False, minrate=self.minrate),
            nn.ReLU(),
            fr.ConvTranspose2d(32, 16, kernel_size=3, padding=1, bias=False, minrate=self.minrate),
            nn.ReLU(),
            fr.ConvTranspose2d(16, img_shape[0], kernel_size=3, padding=1, bias=False, minrate=self.minrate),
            nn.Sigmoid()  # Output in range [0, 1]
        )


    
    def forward(self, file_name_idx):
        # Map file name index to embedding
        embedding = self.embedding(file_name_idx)
        
        # Fully connected layer to reshape embedding
        latent = self.fc1(embedding)
        latent = latent.view(-1, *self.latent_shape)  # Reshape to latent tensor
        
        # Decode to reconstruct the image
        image = self.decoder2(latent)
        return image
    
class FileNameToImage_no_fr(nn.Module):
    def __init__(self, num_files, embedding_dim, latent_shape, img_shape):
        super(FileNameToImage_no_fr, self).__init__()
        self.embedding_dim = embedding_dim
        self.latent_shape = latent_shape
        self.img_shape = img_shape
        self.minrate = 1.0
        
        # Embedding layer to map file name indices to dense vectors
        self.embedding = nn.Embedding(num_files, embedding_dim)
        
        # Fully connected layer to reshape embedding into latent tensor
        self.fc1 = nn.Linear(embedding_dim, latent_shape[0] * latent_shape[1] * latent_shape[2], bias=False)
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(latent_shape[0], 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            nn.ConvTranspose2d(16, img_shape[0], kernel_size=3, padding=1, bias=False),
            nn.Sigmoid()  # Output in range [0, 1]
        )


    
    def forward(self, file_name_idx):
        # Map file name index to embedding
        embedding = self.embedding(file_name_idx)
        
        # Fully connected layer to reshape embedding
        latent = self.fc1(embedding)
        latent = latent.view(-1, *self.latent_shape)  # Reshape to latent tensor
        
        # Decode to reconstruct the image
        image = self.decoder2(latent)
        return image
    