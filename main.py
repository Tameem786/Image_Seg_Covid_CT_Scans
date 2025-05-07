import os
import argparse
import logging
from tqdm import tqdm
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
    return dice.mean().item()

def plotGraph(train_losses, train_dices, val_losses, val_dices, fold):
    plt.figure(figsize=(12, 5))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title(f'Loss - Fold {fold+1}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot Dice Score
    plt.subplot(1, 2, 2)
    plt.plot(train_dices, label='Train Score')
    plt.plot(val_dices, label='Val Score')
    plt.title(f'Dice Score - Fold {fold+1}')
    plt.xlabel('Epochs')
    plt.ylabel('Dice Score')
    plt.legend()

    plt.tight_layout()
    os.makedirs('results', exist_ok=True)
    plt.savefig(f'results/training_metrics_fold_{fold+1}.png')
    plt.close()


def evaluate(model, criterion, val_loader):
    model.eval()
    total_score = 0.0
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, masks)
            score = dice_coeff(torch.sigmoid(outputs), masks)
            total_score += score
            total_loss += loss.item()
            num_batches += 1

    avg_score = total_score / num_batches
    avg_loss = total_loss /  num_batches
    return avg_loss, avg_score

def eval_model(device, fold):
    model = UNet().to(device)
    model.load_state_dict(torch.load('checkpoints/best_model.pth'))
    model.eval()
    for i in range(fold):
        val_dataset = CovidDataset(f'MosMedData/slices/valid_new{i}.txt') # data = train_dataset[0]
        with torch.no_grad():
            img, mask = val_dataset[0]
            pred = model(img[None].to(device))[0].cpu().squeeze().numpy()

        plt.subplot(1, 3, 1); plt.imshow(img.squeeze(), cmap="gray"); plt.title("Image")
        plt.subplot(1, 3, 2); plt.imshow(mask.squeeze(), cmap="gray"); plt.title("Mask")
        plt.subplot(1, 3, 3); plt.imshow(pred > 0.5, cmap="gray"); plt.title("Prediction")
        plt.savefig(f'results/evaluation_fold{fold}.png')
        plt.close()
        # plt.show()

def train(fold, lr, batch, epochs, device):
    best_dice_score = 0.0

    for i in tqdm(range(FOLD_NUM), desc='Fold Running'):
        logger.info(f"Started FOLD {i+1}")

        train_dataset = CovidDataset(f'MosMedData/slices/train_new{i}.txt')
        val_dataset = CovidDataset(f'MosMedData/slices/valid_new{i}.txt')

        train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

        model = UNet().to(DEVICE)

        criterion = torch.nn.BCEWithLogitsLoss()  # use BCEWithLogitsLoss() if not using sigmoid
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        scaler = torch.amp.GradScaler('cuda')

        total_train_losses = []
        total_train_scores = []

        total_val_losses = []
        total_val_scores = []

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

            total_train_losses.append(total_loss / len(train_dataloader))
            total_train_scores.append(total_dice / len(train_dataloader))

            avg_val_loss, avg_val_score = evaluate(model, criterion, val_dataloader)

            total_val_losses.append(avg_val_loss)
            total_val_scores.append(avg_val_score)

            if (epoch+1)%20 == 0:
                logger.info(f"Epoch {epoch+1} | Train Loss: {total_train_losses[-1]:.4f} | Train Dice Score: {total_train_scores[-1]:.4f}")
                logger.info(f"Epoch {epoch+1} | Val Loss: {avg_val_loss:.4f} | Val Dice Score: {avg_val_score:.4f}")

        score = sum(total_val_scores) / len(total_val_scores)
        if best_dice_score < score:
            best_dice_score = score
            logger.info(f'Best Dice Score {best_dice_score}')
            path_dir = f'checkpoints/best_model.pth'
            if not os.path.exists('checkpoints'):
                os.makedirs('checkpoints')
            torch.save(model.state_dict(), path_dir)
            logger.info(f"Model saved at {path_dir}")

        # Plotting the loss and dice coefficient
        plotGraph(total_train_losses, total_train_scores, total_val_losses, total_val_scores, i)


if __name__ == '__main__':

    FOLD_NUM = 5
    LEARN_RATE = 1e-4
    BATCH_SIZE = 8
    EPOCHS = 100

    logger = logging.getLogger(__name__)
    parser = argparse.ArgumentParser()

    parser.add_argument('--train', type=bool, help='True if you want to train, else False', default=False)
    parser.add_argument('--test', type=bool, help='True if you want to test, else False', default=False)

    args = parser.parse_args()

    # torch.cuda.empty_cache()
    # torch.manual_seed(42)

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {DEVICE}")


    if args.train == True:
        logger.info('Start Training')
        # train(FOLD_NUM, LEARN_RATE, BATCH_SIZE, EPOCHS, DEVICE)
    elif args.test == True:
        logger.info('Start Testing')
        # eval_model(DEVICE, FOLD_NUM)
    else:
        logger.error("Invalid Command. 'python main.py --train True' for training. 'python main.py --test True' for testing.")

    

