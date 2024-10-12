"""Infer trained unet on pure noise."""

import torch
import cv2

from unet import UNet
from utils import Denoiser


def main():
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


if __name__ == '__main__':
    main()
