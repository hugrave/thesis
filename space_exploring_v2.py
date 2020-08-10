import argparse
import numpy as np

import torch
from torchvision import utils
from model import Generator
from sample_generation_net import Net

import lpips
import torch.distributions as tdist

device = torch.device("cuda")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--net_ckpt', type=str, required=True)
    parser.add_argument('--g_ckpt', type=str, required=True)
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--pics', type=int, default=20)

    args = parser.parse_args()


    # Init generation and net
    g = Generator(256, 512, 8)
    g.load_state_dict(torch.load(args.g_ckpt)['g_ema'], strict=False)
    g.eval()
    g = g.to(device)

    net = Net()
    net.load_state_dict(torch.load(args.net_ckpt)['net'])
    net.eval()
    net = net.to(device)


    for i in range(args.pics):
        inputs = torch.randn(1, 512).to(device)
        latent = net(inputs)
        image, _ = g([latent], input_is_latent=True)

        utils.save_image(
            image,
            f'generated_images/test_{str(i)}.png',
            nrow=1,
            normalize=True,
            range=(-1, 1),
        )