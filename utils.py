import torch


def gather(consts: torch.Tensor, t: torch.Tensor):
    """Gather consts for $t$ and reshape to feature map shape"""
    c = consts.gather(-1, t)
    return c.reshape(-1, 1, 1, 1)


def NoiseGenerator(beta):
    alpha = 1. - beta
    alpha_bar = torch.cumprod(alpha, dim=0)
    def generate_noise(x0, t):
        mean = gather(alpha_bar, t) ** 0.5 * x0
        var = 1 - gather(alpha_bar, t)
        eps = torch.randn_like(x0, device=x0.device)
        return mean + (var ** 0.5) * eps, eps # also returns noise
    return generate_noise


def Denoiser(beta):
    alpha = 1. - beta
    alpha_bar = torch.cumprod(alpha, dim=0)
    def denoise_sample(xt, noise, t):
        alpha_t = gather(alpha, t)
        alpha_bar_t = gather(alpha_bar, t)
        eps_coef = (1 - alpha_t) / (1 - alpha_bar_t) ** .5
        mean = 1 / (alpha_t ** 0.5) * (xt - eps_coef * noise)
        var = gather(beta, t)
        eps = torch.randn(xt.shape, device=xt.device)
        return mean + (var ** 0.5) * eps
    return denoise_sample
