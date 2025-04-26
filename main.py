import os
import logging
import matplotlib.pyplot as plt

from unet import UNet
from dataset import CovidDataset

import torch
from torch import Tensor
from torchvision.transforms import transforms
from torch.utils.data import DataLoader

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

def dice_loss(input: Tensor, target: Tensor):
    # Dice loss (objective to minimize) between 0 and 1
    return 1 - dice_coeff(input, target, reduce_batch_first=False)

def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()

if __name__ == '__main__':

    LEARN_RATE = 1e-4
    BATCH_SIZE = 8
    EPOCHS = 100

    logger = logging.getLogger(__name__)

    torch.cuda.empty_cache()
    # torch.manual_seed(42)

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {DEVICE}")

    train_dataset = CovidDataset('MosMedData/slices/train_new1.txt')
    val_dataset = CovidDataset('MosMedData/slices/valid_new1.txt')

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = UNet().to(DEVICE)

    criterion = torch.nn.BCEWithLogitsLoss()  # use BCEWithLogitsLoss() if not using sigmoid
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler('cuda')

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        total_dice = 0

        for imgs, masks in train_dataloader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)

            optimizer.zero_grad()

            with torch.autocast(device_type='cuda'):
                preds = model(imgs)
                loss = criterion(preds, masks)
                # loss += dice_loss(torch.sigmoid(preds), masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            total_dice += dice_coeff(torch.sigmoid(preds), masks)

        logger.info(f"Epoch {epoch+1} | Loss: {total_loss/ len(train_dataloader):.4f} | Dice: {total_dice / len(train_dataloader):.4f}")

        if (epoch+1) % 10 == 0:
            path_dir = 'checkpoints/model_epoch_{}.pth'.format(epoch + 1)
            if not os.path.exists('checkpoints'):
                os.makedirs('checkpoints')
            torch.save(model.state_dict(), path_dir)
            logger.info(f"Model saved at {path_dir}")

    model.eval()
    with torch.no_grad():
        img, mask = train_dataset[0]
        pred = model(img[None].to(DEVICE))[0].cpu().squeeze().numpy()

    plt.subplot(1, 3, 1); plt.imshow(img.squeeze(), cmap="gray"); plt.title("Image")
    plt.subplot(1, 3, 2); plt.imshow(mask.squeeze(), cmap="gray"); plt.title("Mask")
    plt.subplot(1, 3, 3); plt.imshow(pred > 0.5, cmap="gray"); plt.title("Prediction")
    plt.show()
