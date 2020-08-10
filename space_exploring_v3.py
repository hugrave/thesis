import argparse
import numpy as np

import torch
from torchvision import utils
from model import Generator
from sample_generation_net import Net

import lpips
import torch.distributions as tdist

device = torch.device("cuda")


probabilities = {
    "input": 0.6,
    "random": 0.4
}

max_n = 3

def generate(g, latents):

    layers = []
    input_latents = []
    random_latents = []
    choices = np.random.choice(list(probabilities.keys()), 14, p=list(probabilities.values()))
    for layer in range(14):
        if choices[layer] == "input":
            if len(input_latents) + len(random_latents) >= max_n and len(input_latents) != 0:
                # Select an already chosen latent vector
                idx = np.random.choice(range(len(input_latents)), 1)[0]
                latent = input_latents[idx]
            else:
                # Select a new latent vector from the input pool
                idx = np.random.choice(range(len(latents)), 1)[0]
                latent = latents[idx]
                found = False
                for l in input_latents:
                    if torch.all(l.eq(latent)): found = True
                if not found: input_latents.append(latent)
        else:
            if len(input_latents) + len(random_latents) >= max_n and len(random_latents) != 0:
                idx = np.random.choice(range(len(random_latents)), 1)[0]
                latent = random_latents[idx]
            else:
                latent = g.get_latent(torch.randn(1, 512).to(device))
                random_latents.append(latent)
        
        layers.append(latent)

    layers = torch.cat(layers)
    img, _ = g(layers, input_is_latent=True, style_mixing=True)
    return img

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--g_ckpt', type=str, required=True)
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--pics', type=int, default=20)
    parser.add_argument('projected_files', metavar='FILES', nargs='+')

    args = parser.parse_args()


    # Init generation and net
    g = Generator(256, 512, 8)
    g.load_state_dict(torch.load(args.g_ckpt)['g_ema'], strict=False)
    g.eval()
    g = g.to(device)

    # Extract the referent latent
    latents = []
    for projected_file in args.projected_files:
        projected = torch.load(projected_file)
        for key in projected:
            if key != "noises":
                latents.append(projected[key]["latent"])

    latents = [l.reshape(1, -1).to(device) for l in latents]


    for i in range(args.pics):
        image = generate(g, latents)
        

        utils.save_image(
            image,
            f'generated_images/test_{i}.png',
            nrow=1,
            normalize=True,
            range=(-1, 1),
        )