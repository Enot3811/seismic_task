"""Infer trained unet on dataset slices."""

import torch
import cv2
import albumentations as A
import numpy as np

from unet import UNet
from utils import Denoiser, NoiseGenerator
from segy_slice_dataset import SegySliceDataset


def main():
    ckpt_pth = 'trains/train5/ckpts/best_checkpoint.pth'
    device = torch.device('cuda')
    unet = UNet(image_channels=1, n_channels=32).to(device=device)
    unet.load_state_dict(torch.load(ckpt_pth)['model_state_dict'])

    mean = 0.5
    std = 0.19

    n_steps = 100
    beta = torch.linspace(0.0001, 0.04, n_steps).to(device=device)
    denoise_sample = Denoiser(beta)
    make_noise = NoiseGenerator(beta)

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


if __name__ == '__main__':
    main()
