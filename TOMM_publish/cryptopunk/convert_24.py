import os
import numpy as np
import matplotlib.pyplot as plt
# read all images in the folder and convert them to 24x24
img_names = os.listdir('./cryptopunk_original')
for img_name in img_names:
    print(f'Processing {img_name}')
    img = plt.imread(f'./cryptopunk_original/{img_name}')
    img = img[:,:,:3]  # remove alpha channel
    img = img[::14, ::14, :]
    plt.imsave(f'./cryptopunk_24/{img_name.split(".")[0]}.png', img)


