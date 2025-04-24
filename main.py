import os
import numpy as np
import matplotlib.pyplot as plt

def read_dataset(path):
    imgs, masks = [], []
    with open(path, 'r') as f:
        for line in f:
            img_file_name = line.rstrip()
            img_file_path = f'MosMedData/slices/imgs/{img_file_name}'
            msk_file_path = f'MosMedData/slices/masks/{img_file_name}'
            if os.path.exists(img_file_path) and os.path.exists(msk_file_path):
                imgs.append(img_file_path)
                masks.append(msk_file_path)
    f.close()
    return imgs, masks

def visualize_img(img, mask):
    plt.subplot(1, 2, 1)
    plt.title("Image")
    plt.imshow(img, cmap='gray' if img.ndim == 2 else None)

    plt.subplot(1, 2, 2)
    plt.title("Mask")
    plt.imshow(mask, cmap='gray')
    plt.show()

if __name__ == '__main__':
    images, masks = [], []

    # Loading Matched Images and Masks from Train Data
    for i in range(5):
        images, masks = read_dataset(f'MosMedData/slices/train_new{i}.txt')
        print(f'{len(images)} images, {len(masks)} masks')

    # visualize_img(np.load(images[10]), np.load(masks[10])) # Visualize Image and it's Mask

    # print(f'{np.load(images[0]).shape}, {np.load(masks[0]).shape}')