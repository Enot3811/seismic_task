from pathlib import Path
import shutil
import math
from typing import Dict, Any

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import albumentations as A
from loguru import logger

from model.unet import UNet
from utils.data_utils.data_functions import make_noise_generator
from utils.segy_utils.segy_slice_dataset import SegySliceDataset


# TODO добавить логгер
def train_unet(config: Dict[str, Any], model: torch.nn.Module):

    # Device
    if config['device'] == 'cuda':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            logger.warning('Cuda is not available. Switching device to "CPU".')
            device = torch.device('cpu')
    elif config['device'] == 'cpu':
        device = torch.device('cpu')
    else:
        raise ValueError('Wrong device')
    
    # Random
    torch.random.manual_seed(config['random_seed'])
    
    # Directories
    save_dir = Path(config['save_dir'])
    if save_dir.exists():
        input(f'{str(save_dir)} already exists. Continue to rewrite it.')
        shutil.rmtree(save_dir)
    tensorboard_dir = save_dir / 'tensorboard'
    ckpt_dir = save_dir / 'ckpts'
    tensorboard_dir.mkdir(parents=True)
    ckpt_dir.mkdir(parents=True)

    # Tensorboard
    log_writer = SummaryWriter(str(tensorboard_dir))

    # Noise generator
    beta = torch.linspace(0.0001, 0.04, config['noise_steps'], device=device)
    make_noise = make_noise_generator(beta)

    # Augmentations
    if config['crop_size'] or config['mean_std']:
        transforms = []
        if config['crop_size']:
            transforms.append(A.RandomCrop(*config['crop_size']))
        if config['mean_std']:
            transforms.append(A.Normalize(*config['mean_std'],
                                          max_pixel_value=1))
        transforms = A.Compose(transforms=transforms)
    else:
        transforms=None

    # Datasets and dataloaders
    train_dset = SegySliceDataset(
        config['sgy_path'], config['train_axes'], transforms=transforms)
    val_dset = SegySliceDataset(
        config['sgy_path'], config['val_axes'], transforms=transforms)

    train_loader = DataLoader(train_dset, config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dset, config['batch_size'], shuffle=True)

    # Model
    unet = UNet(image_channels=1, n_channels=32).to(device=device)

    # Optimizer
    optimizer = torch.optim.AdamW(unet.parameters(), lr=config['start_lr'])
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['epochs'], eta_min=config['end_lr'])

    # Do train
    best_metric = None
    for e in range(config['epochs']):

        # Train step
        train_losses = []
        pbar = tqdm(train_loader, desc=f'Epoch {e} train')
        for i, batch in enumerate(pbar):
            batch = batch.to(device=device)

            # Noise samples
            t = (torch.randint(
                0, config['noise_steps'], (batch.shape[0],),
                dtype=torch.long).to(device=device))  # Random 't's
            batch_noised, noise = make_noise(batch, t)

            pred_noise = unet(batch_noised, t)
            
            loss = F.mse_loss(noise, pred_noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # Val step
        with torch.no_grad():
            unet.eval()
            val_losses = []
            pbar = tqdm(val_loader, desc=f'Epoch {e} val')
            for i, batch in enumerate(pbar):
                batch = batch.to(device=device)

                # Noise samples
                t = (torch.randint(
                    0, config['noise_steps'], (batch.shape[0],),
                    dtype=torch.long).to(device=device))  # Random 't's
                batch_noised, noise = make_noise(batch, t)

                pred_noise = unet(batch_noised, t)

                loss = F.mse_loss(noise, pred_noise)
                val_losses.append(loss)

        lr_scheduler.step()
        lr = lr_scheduler.get_last_lr()[0]

        # Log
        train_loss = torch.mean(torch.tensor(train_losses))
        val_loss = torch.mean(torch.tensor(val_losses))

        log_writer.add_scalars('loss', {
            'train': train_loss,
            'val': val_loss
        }, global_step=e)
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


# TODO всё переписать
def infer_on_noise(config: Dict[str, Any]):
    """Infer trained unet on pure noise."""
    ckpt_pth = 'trains/train5/ckpts/best_checkpoint.pth'
    device = torch.device('cuda')
    unet = UNet(image_channels=1, n_channels=32).to(device=device)
    unet.load_state_dict(torch.load(ckpt_pth)['model_state_dict'])

    # mean = 0.5
    # std = 0.19

    n_steps = 100
    beta = torch.linspace(0.0001, 0.04, n_steps).to(device=device)
    denoise_sample = Denoiser(beta)

    # Make and show 10 examples
    x = torch.randn(10, 1, 224, 224).to(device=device)
    for i in range(n_steps):
        t = torch.tensor(n_steps - i - 1, dtype=torch.long).to(device=device)
        with torch.no_grad():
            pred_noise = unet(x.float(), t.unsqueeze(0))
            x = denoise_sample(x, pred_noise, t.unsqueeze(0))
    # x = x * std + mean

    for i in range(10):
        img = x[i].cpu().permute(1, 2, 0).numpy()

        cv2.imshow('denoised', img)
        key = cv2.waitKey(0)
        if key == 27:
            break


def infer_on_noised_slices(config: Dict[str, Any]):
    """Infer trained unet on dataset slices."""
    ckpt_pth = 'trains/train5/ckpts/best_checkpoint.pth'
    device = torch.device('cuda')
    unet = UNet(image_channels=1, n_channels=32).to(device=device)
    unet.load_state_dict(torch.load(ckpt_pth)['model_state_dict'])

    mean = 0.5
    std = 0.19

    n_steps = 100
    beta = torch.linspace(0.0001, 0.04, n_steps).to(device=device)
    denoise_sample = Denoiser(beta)
    make_noise = make_noise_generator(beta)

    sgy_path = '../data/seismic/seismic.sgy'
    crop_size = (224, 224)
    transforms = A.Compose(transforms=[
        A.RandomCrop(*crop_size),
        A.Normalize(mean=(mean,), std=(std,), max_pixel_value=1)
    ])
    val_axes = (0,)
    val_dset = SegySliceDataset(sgy_path, val_axes, transforms=transforms)

    for source in val_dset:

        x = source[None, ...].to(device=device)
        noised_x, noise = make_noise(
            x, torch.tensor(50, dtype=torch.long, device=device))
        x = noised_x

        for i in range(50, n_steps):
            t = torch.tensor(n_steps - i - 1, dtype=torch.long).to(device=device)
            with torch.no_grad():
                pred_noise = unet(x.float(), t.unsqueeze(0))
                x = denoise_sample(x, pred_noise, t.unsqueeze(0))

        x = x[0].cpu().permute(1, 2, 0).numpy()
        noised_x = noised_x[0].cpu().permute(1, 2, 0).numpy()
        source = source.permute(1, 2, 0).numpy()

        img = np.hstack((source, noised_x, x))

        cv2.imshow('Sample', img)
        key = cv2.waitKey(0)
        if key == 27:
            break
