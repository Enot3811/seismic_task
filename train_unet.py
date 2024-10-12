from pathlib import Path
import shutil
import math

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import albumentations as A

from unet import UNet
from utils import NoiseGenerator
from segy_slice_dataset import SegySliceDataset


def main():

    # Training params
    batch_size = 16
    lr = 2e-4
    end_lr = 2e-6
    epochs = 5
    device = torch.device('cuda')
    save_loss_step = 25
    save_path = Path('trains/train6')

    # Noise generator
    n_steps = 100
    beta = torch.linspace(0.0001, 0.04, n_steps, device=device)
    make_noise = NoiseGenerator(beta)

    # Datasets and dataloaders
    crop_size = (224, 224)
    transforms = A.Compose(transforms=[
        A.RandomCrop(*crop_size),
        A.Normalize(mean=(0.5,), std=(0.19,), max_pixel_value=1)
    ])
    sgy_path = '../data/seismic/seismic.sgy'
    train_axes = (1,)
    train_dset = SegySliceDataset(sgy_path, train_axes, transforms=transforms)
    train_loader = DataLoader(
        train_dset, batch_size, shuffle=True, drop_last=True)
    val_axes = (0,)
    val_dset = SegySliceDataset(sgy_path, val_axes, transforms=transforms)
    val_loader = DataLoader(
        val_dset, batch_size, shuffle=True, drop_last=True)
    val_save_loss_step = (math.ceil(len(val_loader) /
                          math.ceil(len(train_loader) / save_loss_step)))

    # Model
    unet = UNet(image_channels=1, n_channels=32).to(device=device)

    # Optimizer
    optimizer = torch.optim.AdamW(unet.parameters(), lr=lr)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=end_lr)

    # Prepare for training
    if save_path.exists():
        input(f'{str(save_path)} already exists. Continue to rewrite it.')
        shutil.rmtree(save_path)
    tensorboard_dir = save_path / 'tensorboard'
    ckpt_dir = save_path / 'ckpts'
    tensorboard_dir.mkdir(parents=True)
    ckpt_dir.mkdir(parents=True)

    log_writer = SummaryWriter(str(tensorboard_dir))

    # Train
    best_metric = None
    for e in range(epochs):

        # Train step
        losses = []
        t_train_losses = []
        pbar = tqdm(train_loader, desc=f'Epoch {e} train')
        for i, batch in enumerate(pbar):
            batch = batch.to(device=device)

            # Noise samples
            t = (torch.randint(0, n_steps, (batch_size,), dtype=torch.long)
                 .to(device=device))  # Random 't's
            batch_noised, noise = make_noise(batch, t)

            pred_noise = unet(batch_noised, t)
            
            loss = F.mse_loss(noise, pred_noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

            if i % save_loss_step == 0 or i == len(train_loader):
                train_loss = torch.mean(torch.tensor(losses))
                pbar.set_postfix({'loss': f'{train_loss.item():.4f}'})
                t_train_losses.append(train_loss)
                losses.clear()

        # Val step
        with torch.no_grad():
            unet.eval()
            losses.clear()
            t_val_losses = []
            pbar = tqdm(val_loader, desc=f'Epoch {e} val')
            for i, batch in enumerate(pbar):
                batch = batch.to(device=device)

                # Noise samples
                t = (torch.randint(0, n_steps, (batch_size,), dtype=torch.long)
                        .to(device=device))  # Random 't's
                batch_noised, noise = make_noise(batch, t)

                pred_noise = unet(batch_noised, t)

                loss = F.mse_loss(noise, pred_noise)
                losses.append(loss)

                if i % val_save_loss_step == 0 or i == len(val_loader):
                    val_loss = torch.mean(torch.tensor(losses))
                    pbar.set_postfix({'loss': f'{val_loss.item():.4f}'})
                    t_val_losses.append(val_loss)
                    losses.clear()

        lr_scheduler.step()
        lr = lr_scheduler.get_last_lr()[0]

        # Log
        for step, (train_loss, val_loss) in enumerate(zip(t_train_losses,
                                                          t_val_losses)):
            log_writer.add_scalars('loss', {
                'train': train_loss,
                'val': val_loss
            }, global_step=e * len(train_loader) + step)
        log_writer.add_scalar('lr', lr, e)
        
        # Save model
        checkpoint = {
            'model_state_dict': unet.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': e + 1
        }
        torch.save(checkpoint, ckpt_dir / 'last_checkpoint.pth')

        if (best_metric is None or best_metric > loss):
            torch.save(checkpoint, ckpt_dir / 'best_checkpoint.pth')
            best_metric = loss

    log_writer.close()


if __name__ == '__main__':
    main()
