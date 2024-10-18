from pathlib import Path
import shutil
import math
from typing import Dict, Any, Tuple

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import albumentations as A
from loguru import logger
import matplotlib.pyplot as plt

from utils.data_utils.data_functions import (
    make_noise_generator, make_denoiser, show_images_cv2)
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
    model.to(device=device)
    
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
        config['sgy_path'], config['train_axes'], transforms=transforms,
        values_range=config['values_range'])
    val_dset = SegySliceDataset(
        config['sgy_path'], config['val_axes'], transforms=transforms,
        values_range=config['values_range'])

    train_loader = DataLoader(train_dset, config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dset, config['batch_size'], shuffle=True)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['start_lr'])
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['epochs'], eta_min=config['end_lr'])

    # Do train
    best_metric = None
    for e in range(config['epochs']):

        # Train step
        model.train()
        train_losses = []
        pbar = tqdm(train_loader, desc=f'Epoch {e} train')
        for batch in train_loader:
            batch = batch.to(device=device)

            # Noise samples
            t = (torch.randint(
                0, config['noise_steps'], (batch.shape[0],),
                dtype=torch.long).to(device=device))  # Random 't's
            batch_noised, noise = make_noise(batch, t)

            pred_noise = model(batch_noised, t)
            
            loss = F.mse_loss(noise, pred_noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            pbar.update()
            pbar.set_postfix_str(f'Batch loss: {loss.item()}')

        train_loss = torch.mean(torch.tensor(train_losses))
        pbar.set_postfix_str(f'Epoch loss: {train_loss.item()}')
        pbar.close()

        # Val step
        with torch.no_grad():
            model.eval()
            val_losses = []
            pbar = tqdm(val_loader, desc=f'Epoch {e} val')
            for batch in val_loader:
                batch = batch.to(device=device)

                # Noise samples
                t = (torch.randint(
                    0, config['noise_steps'], (batch.shape[0],),
                    dtype=torch.long).to(device=device))  # Random 't's
                batch_noised, noise = make_noise(batch, t)

                pred_noise = model(batch_noised, t)

                loss = F.mse_loss(noise, pred_noise)
                val_losses.append(loss)
                pbar.update()
                pbar.set_postfix_str(f'Batch loss: {loss.item()}')

            val_loss = torch.mean(torch.tensor(val_losses))
            pbar.set_postfix_str(f'Epoch loss: {val_loss.item()}')
            pbar.close()

        lr_scheduler.step()
        lr = lr_scheduler.get_last_lr()[0]

        # Log
        log_writer.add_scalars('loss', {
            'train': train_loss,
            'val': val_loss
        }, global_step=e)
        log_writer.add_scalar('lr', lr, e)
        
        # Save model
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': e + 1
        }
        torch.save(checkpoint, ckpt_dir / 'last_checkpoint.pth')

        if (best_metric is None or best_metric > loss):
            torch.save(checkpoint, ckpt_dir / 'best_checkpoint.pth')
            best_metric = loss

    log_writer.close()


def infer_on_noise(
    config: Dict[str, Any],
    model: torch.nn.Module,
    n_examples: int = 10,
    show_mode: str = 'plt',
    device: str = 'cpu'
):
    """Infer trained unet on pure noise."""
    # Device
    if device == 'cuda':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            logger.warning('Cuda is not available. Switching device to "CPU".')
            device = torch.device('cpu')
    elif device == 'cpu':
        device = torch.device('cpu')
    else:
        raise ValueError('Wrong device')
    model = model.to(device=device)
    
    # Show mode
    if show_mode not in ('plt', 'cv2'):
        raise ValueError

    # Denoiser
    beta = torch.linspace(
        0.0001, 0.04, config['noise_steps']).to(device=device)
    denoise_sample = make_denoiser(beta)

    # Make 10 examples
    x = torch.randn(10, 1, 224, 224).to(device=device)
    for i in tqdm(range(config['noise_steps'])):
        t = torch.tensor(
            config['noise_steps'] - i - 1, dtype=torch.long).to(device=device)
        with torch.no_grad():
            pred_noise = model(x, t.unsqueeze(0))
            x = denoise_sample(x, pred_noise, t.unsqueeze(0))
    
    # Normalize
    if config['mean_std']:
        raise NotImplementedError()  # TODO
    x[x < 0.0] = 0.0
    x[x > 1.0] = 1.0

    # Show examples
    imgs = [x[i].cpu().permute(1, 2, 0).numpy() for i in range(x.shape[0])]
    if show_mode == 'plt':
        # Make a grid from samples
        nrows = math.ceil(n_examples / 5)
        fig, axes = plt.subplots(nrows=nrows, ncols=5, figsize=(15, 6))
        axes = axes.flatten()
        for i, img in enumerate(imgs):
            axes[i].imshow(img, cmap='Greys_r')
            axes[i].axis('off')
        plt.tight_layout()
        plt.show()
    else:
        show_images_cv2(imgs, destroy_windows=True)


def infer_on_noised_slices(
    config: Dict[str, Any],
    model: torch.nn.Module,
    n_examples: int = 10,
    show_mode: str = 'plt',
    device: str = 'cpu'
):
    """Infer trained unet on dataset slices."""
    # Device
    if device == 'cuda':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            logger.warning('Cuda is not available. Switching device to "CPU".')
            device = torch.device('cpu')
    elif device == 'cpu':
        device = torch.device('cpu')
    else:
        raise ValueError('Wrong device')
    model = model.to(device=device)

    # Show mode
    if show_mode not in ('plt', 'cv2'):
        raise ValueError
    
    # Noiser and denoiser
    beta = torch.linspace(
        0.0001, 0.04, config['noise_steps']).to(device=device)
    denoise_sample = make_denoiser(beta)
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

    val_dset = SegySliceDataset(
        config['sgy_path'], config['val_axes'], transforms=transforms,
        values_range=config['values_range'])
    sources = torch.stack(
        [val_dset[i] for i in range(n_examples)]).to(device=device)

    noised_x, noise = make_noise(
        sources, torch.tensor(50, dtype=torch.long, device=device))
    x = noised_x

    for i in range(50, config['noise_steps']):
        t = torch.tensor(config['noise_steps'] - i - 1, dtype=torch.long).to(device=device)
        with torch.no_grad():
            pred_noise = model(x, t.unsqueeze(0))
            x = denoise_sample(x, pred_noise, t.unsqueeze(0))

    # Normalize
    if config['mean_std']:
        raise NotImplementedError()  # TODO
    x[x < 0.0] = 0.0
    x[x > 1.0] = 1.0

    # Show examples
    if show_mode == 'plt':
        nrows = math.ceil(n_examples / 5)

        for imgs, title in [(sources, 'Origin'),
                            (noised_x, 'Noised slices'),
                            (x, 'Denoised slices')]:
            imgs = [imgs[i].cpu().permute(1, 2, 0).numpy()
                    for i in range(imgs.shape[0])]

            fig, axes = plt.subplots(nrows=nrows, ncols=5, figsize=(15, 6))
            fig.suptitle(title)
            axes = axes.flatten()
            for i, img in enumerate(imgs):
                axes[i].imshow(img, cmap='Greys_r')
                axes[i].axis('off')
            fig.tight_layout()
        plt.show()
    else:
        # TODO переделать
        show_images_cv2(imgs, destroy_windows=True)
