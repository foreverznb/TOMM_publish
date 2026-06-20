import os
import time
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import frequency_regularization.freq_reg._freqreg as fr
import json
from thop import profile

class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform=None, img_index_json=None):
        """
        Args:
            img_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.img_dir = img_dir
        self.transform = transform
        if img_index_json:
            with open(img_index_json, 'r') as f:
                items = json.load(f).values()
                self.img_labels = [(k+'.png', k+'.png') for k in items]
        else:
            self.img_labels = [(file, file) for file in os.listdir(img_dir)]  # Example assumes labels are in the filename

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_name, label = self.img_labels[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return img_name.split('.')[0], idx, image


# Evaluation function
def evaluate(model, ssim, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    output_imgs = []
    output_imgs_name = []
    try:
        _, sample_inputs, _ = next(iter(test_loader))
        sample_inputs = sample_inputs[:1].to(device)
        with torch.no_grad():
            flops, _ = profile(model, inputs=(sample_inputs,), verbose=False)
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            for _ in range(50):
                _ = model(sample_inputs)
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            latency = (time.perf_counter() - start) / 50 * 1000
        print(f'Average FLOPs per image: {flops / 1e9:.4f} GFLOPs')
        print(f'Average latency per image: {latency:.4f} ms')
    except Exception as e:
        print(f'FLOPs/latency profiling skipped: {e}')
    with torch.no_grad():
        for i, (img_names, inputs, label) in enumerate(test_loader):
            label = label.to(device)
            inputs = inputs.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, label)
            running_loss += loss.item() * inputs.size(0)
            for img in outputs:
                output_imgs.append(img.permute(1, 2, 0).detach().cpu().numpy())
            for name in img_names:
                output_imgs_name.append(name)
    epoch_loss = running_loss / len(test_loader.dataset)
    ssim_score = ssim(outputs, label)
    return epoch_loss, output_imgs, ssim_score, output_imgs_name

def calculate_psnr(loss):
    return 20 * np.log10(1.0 / np.sqrt(loss))

def countParams(model):
    totalnum = 0
    totalbias = 0
    totalzeros = 0
    for name, layer in model.named_modules():
        try:
            num = torch.sum(layer.weight.data.abs() > 0).item()
            totalnum += num  
            num = torch.sum(layer.weight.data.abs() < 0.0000000001).item()
            totalzeros += num
        except:
            pass
        try:
            num = torch.sum(layer.bias.data.abs() > 0).item()
            totalbias += num
            num = torch.sum(layer.bias.data.abs() < 0.0000000001).item()
            totalzeros += num
        except:
            pass
    return totalnum, totalbias, totalzeros

def cleanNet(model):
    for name, layer in model.named_modules():
        try:
            idx = layer.IDROP.abs() < 1e-20
            layer.weight.data[idx] = 0
            layer.IDROP = None
        except:
            pass
        try:
            layer.ZMAT = None
        except:
            pass

        try:
            layer.IMAT = None
        except:
            pass
        try:
            idx = layer.BMAT.abs() < 0.0000000001
            layer.bias.data[idx] = 0
            layer.BMAT = None
        except:
            pass


    return None

def upload_dict(state_dict_fr, state_dict_no_fr):
    for key in state_dict_no_fr.keys():
        # print(key)
        # print(state_dict_fr[key].shape)
        # print(state_dict_no_fr[key].shape)
        if state_dict_fr[key].shape != state_dict_no_fr[key].shape:
            print('error, shape not equal')
        if len(state_dict_fr[key].shape) == 1:
            state_dict_no_fr[key] = fr.idct(state_dict_fr[key])
        elif len(state_dict_fr[key].shape) == 2:
            if key == 'embedding.weight':
                state_dict_no_fr[key] = state_dict_fr[key]
            else:
                state_dict_no_fr[key] = fr.idct_2d(state_dict_fr[key])
            # state_dict_no_fr[key] = fr.idct_2d(state_dict_fr[key][:state_dict_no_fr[key].shape[0],:state_dict_no_fr[key].shape[1]])
        elif len(state_dict_fr[key].shape) == 3:
            state_dict_no_fr[key] = fr.idct_3d(state_dict_fr[key])
        elif len(state_dict_fr[key].shape) == 4:
            state_dict_no_fr[key] = fr.idct_4d(state_dict_fr[key])
        else:
            print('error')
    return state_dict_no_fr