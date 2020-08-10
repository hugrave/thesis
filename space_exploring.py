import argparse
import numpy as np

import torch
from torchvision import utils
from model import Generator

import lpips
import torch.distributions as tdist


def sample_mixture(means, variances, weights, size, step):
    latent = []
    models = len(means)
    choices = np.random.choice(models, size, p=weights)
    
    for i in range(0, size, step):
        distr = choices[i]
        for k in range(step):
            if i+k >= size: break
            latent.append(np.random.normal(means[distr][i+k], variances[distr][i+k]))
    
    return np.array(latent)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--pics', type=int, default=100)
    parser.add_argument('projected_files', metavar='FILES', nargs='+')

    args = parser.parse_args()

    device = "cuda"

    # Loading the projected latent space representation
    latents = []
    noises = []
    for projected_file in args.projected_files:
        projected = torch.load(projected_file)
        for key in projected:
            if key != "noises":
                latents.append(projected[key]["latent"])
        noises.append(projected["noises"])



    # Loading the generator model
    g_ema = Generator(args.size, 512, 8)
    g_ema.load_state_dict(torch.load(args.ckpt)['g_ema'], strict=False)
    g_ema.eval()
    g_ema = g_ema.to(device)


    ###############
    # Mixture of multivariate Gaussians
    # Uncomment this section to use GMM with multivariate distributions
    ###############
    
    # variance_hp = 1

    # choices = np.random.choice(list(range(len(latents))), args.pics)
    # variance = np.diag(np.ones(latents[0].shape[0]) * variance_hp)

    # for i in range(args.pics):

    #     latent = latents[choices[i]]
    #     mean = latent.detach().reshape(-1).cpu().numpy()    
    #     new_latent = np.random.multivariate_normal(mean, variance, args.pics)
    #     latent = torch.from_numpy(new_latent).float().to(device)
        
    #     # Generate the image associated to the vector
    #     with torch.no_grad():        
    #         img_gen, _ = g_ema([latent[i].reshape(1, -1)], input_is_latent=True, noise=noises[0])
    #         utils.save_image(
    #             img_gen,
    #             f'generated_images/gmm_mv_{i}.png',
    #             nrow=1,
    #             normalize=True,
    #             range=(-1, 1),
    #         )


    ###############
    # Mixture of independent Gaussians
    # Uncomment this section to use GMM with independent distributions
    # Modify the value of k
    ###############

    variance_hp = 0.8
    k_hp = 100

    size = latents[0].shape[0]
    means = []
    variances = []
    weights = []

    # Adding all the distributions around the input images
    for latent in latents:
        means.append(latent.detach().reshape(-1).cpu().numpy())
        variances.append(np.random.rand(latent.shape[0]) * variance_hp)
        weights.append(1.0 / len(latents))

    
    for i in range(args.pics):
        latent = sample_mixture(means, variances, weights, size, k_hp)
        latent = torch.from_numpy(latent).float().to(device)
        # Generate the image associated to the vector
        with torch.no_grad():        
            img_gen, _ = g_ema([latent.reshape(1, -1)], input_is_latent=True, truncation=1)            

            utils.save_image(
                img_gen,
                f'generated_images/gmm_ind_{i}.png',
                nrow=1,
                normalize=True,
                range=(-1, 1),
            )
    

    ###############
    # Linear points interpolation
    # Uncomment this section to traverse the linear space between two given images
    ###############

    # start_point = {"latent": latents[0], "noises": np.array(noises[0])}        
    # end_point = {"latent": latents[1], "noises": np.array(noises[1])}        
    
    # moving_vector = {
    #     "latent": end_point["latent"] - start_point["latent"],
    #     "noises": end_point["noises"] - start_point["noises"]
    # }

    # increment = 1 / args.pics
    # point = start_point
    # for i in range(args.pics):
    #     point = {
    #         "latent": point["latent"] + increment * moving_vector["latent"],
    #         "noises": point["noises"] + increment * moving_vector["noises"],
    #     }

    #     # Generate the image associated to the vector
    #     with torch.no_grad():        
    #         img_gen, _ = g_ema([point["latent"].reshape(1, -1)], input_is_latent=True, noise=point["noises"])
    #         utils.save_image(
    #             img_gen,
    #             f'generated_images/test_{i}.png',
    #             nrow=1,
    #             normalize=True,
    #             range=(-1, 1),
    #         )
