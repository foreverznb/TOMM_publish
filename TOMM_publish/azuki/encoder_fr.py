import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append("../")
import frequency_regularization.freq_reg._freqreg as fr

class FileNameToImage(nn.Module):
    def __init__(self, num_files, embedding_dim, latent_shape, img_shape):
        super(FileNameToImage, self).__init__()
        self.embedding_dim = embedding_dim
        self.latent_shape = latent_shape
        self.img_shape = img_shape
        self.minrate = 0.8
        
        # Embedding layer to map file name indices to dense vectors
        self.embedding = nn.Embedding(num_files, embedding_dim)
        
        # Fully connected layer to reshape embedding into latent tensor
        self.fc1 = fr.Linear(embedding_dim, latent_shape[0] * latent_shape[1] * latent_shape[2], bias=True, minrate=1.0)
        self.decoder2 = nn.Sequential(
            fr.ConvTranspose2d(latent_shape[0], 512, kernel_size=4, stride=2, padding=1, minrate=self.minrate),
            nn.ReLU(),
            fr.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, minrate=self.minrate),
            nn.ReLU(),
            fr.ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=1, minrate=self.minrate),
            nn.ReLU(),
            fr.ConvTranspose2d(128, 64, kernel_size=3, padding=1, minrate=self.minrate),
            nn.ReLU(),
            fr.ConvTranspose2d(64, 32, kernel_size=3, padding=1, minrate=self.minrate),
            nn.ReLU(),
            fr.ConvTranspose2d(32, 16, kernel_size=3, padding=1, minrate=self.minrate),
            nn.ReLU(),
            fr.ConvTranspose2d(16, img_shape[0], kernel_size=3, padding=1, minrate=self.minrate),
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
        
        # Embedding layer to map file name indices to dense vectors
        self.embedding = nn.Embedding(num_files, embedding_dim)
        
        # Fully connected layer to reshape embedding into latent tensor
        self.fc1 = nn.Linear(embedding_dim, latent_shape[0] * latent_shape[1] * latent_shape[2], bias=True)
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(latent_shape[0], 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, img_shape[0], kernel_size=3, padding=1),
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
    