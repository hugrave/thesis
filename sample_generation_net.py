import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as tdist

import lpips
from model import Generator, PixelNorm, EqualLinear

import argparse
from tqdm import tqdm
import numpy as np

from torchviz import make_dot

device = torch.device("cuda")
dimension = 512

log_norm_constant = -0.5 * np.log(2 * np.pi)

class GaussianMixture:
    def __init__(self, data, n_components, n_iter=100):
        m = data.size(0)
        idxs = torch.from_numpy(np.random.choice(m, n_components, replace=False))
        self.mu = data[idxs]

        self.logvar = torch.Tensor(n_components, data.shape[1]).fill_(0.1).log().to(device)

        self.pi = torch.empty(n_components).fill_(1. / n_components).to(device)
        self.data = data
        self.n_iter = n_iter
        self.taken_distributions = []

    def log_gaussian(self, x, mean=0, logvar=0.):
        """
        Returns the density of x under the supplied gaussian. Defaults to
        standard gaussian N(0, I)
        :param x: (*) torch.Tensor
        :param mean: float or torch.FloatTensor with dimensions (*)
        :param logvar: float or torch.FloatTensor with dimensions (*)
        :return: (*) elementwise log density
        """

        if type(logvar) == 'float':
            logvar = x.new(1).fill_(logvar)

        a = (x - mean) ** 2
        log_p = -0.5 * (logvar + a / logvar.exp())
        log_p = log_p + log_norm_constant

        return log_p
    

    def get_likelihoods(self, log=True):
        """
        :param X: design matrix (examples, features)
        :param mu: the component means (K, features)
        :param logvar: the component log-variances (K, features)
        :param log: return value in log domain?
            Note: exponentiating can be unstable in high dimensions.
        :return likelihoods: (K, examples)
        """

        # get feature-wise log-likelihoods (K, examples, features)
        log_likelihoods = self.log_gaussian(
            self.data[None, :, :], # (1, examples, features)
            self.mu[:, None, :], # (K, 1, features)
            self.logvar[:, None, :] # (K, 1, features)
        )

        # sum over the feature dimension
        log_likelihoods = log_likelihoods.sum(-1)

        if not log:
            log_likelihoods.exp_()

        return log_likelihoods
    
    def get_posteriors(self, log_likelihoods):
        """
        Calculate the the posterior probabilities log p(z|x), assuming a uniform prior over
        components.
        :param likelihoods: the relative likelihood p(x|z), of each data point under each mode (K, examples)
        :return: the log posterior p(z|x) (K, examples)
        """
        posteriors = log_likelihoods - torch.logsumexp(log_likelihoods, dim=0, keepdim=True)
        return posteriors

    def update_params(self, log_posteriors, eps=1e-6, min_var=1e-6):
        """
        :param X: design matrix (examples, features)
        :param log_posteriors: the log posterior probabilities p(z|x) (K, examples)
        :returns mu, var, pi: (K, features) , (K, features) , (K)
        """

        posteriors = log_posteriors.exp()

        # compute `N_k` the proxy "number of points" assigned to each distribution.
        K = posteriors.size(0)
        N_k = torch.sum(posteriors, dim=1) # (K)
        N_k = N_k.view(K, 1, 1)

        # get the means by taking the weighted combination of points
        # (K, 1, examples) @ (1, examples, features) -> (K, 1, features)
        mu = posteriors[:, None] @ self.data[None,]
        mu = mu / (N_k + eps)

        # compute the diagonal covar. matrix, by taking a weighted combination of
        # the each point's square distance from the mean
        A = self.data - mu
        var = posteriors[:, None] @ (A ** 2) # (K, 1, features)
        var = var / (N_k + eps)
        logvar = torch.clamp(var, min=min_var).log()

        # recompute the mixing probabilities
        m = self.data.size(1) # nb. of training examples
        pi = N_k / N_k.sum()

        self.mu = mu.squeeze(1)
        self.logvar = logvar.squeeze(1)
        self.pi = pi.squeeze(1)

    
    def fit(self, n_iter=100):
        for i in range(n_iter):
            log_likelihoods = self.get_likelihoods()
            posteriors = self.get_posteriors(log_likelihoods)
            self.update_params(posteriors)

    # Utility methods
    def get_closest(self, mean_vector):
        if len(self.taken_distributions) == self.mu.shape[0]:
            self.taken_distributions = []

        dist = torch.norm(self.mu - mean_vector.reshape((1,-1)), 2, dim=1)
        dist_idx = dist.argsort() # Sorting it in ascending order and get the indexes

        idx = dist_idx[0].item()
        for i in dist_idx:
            if i not in self.taken_distributions:
                idx = i
                break
        
        self.taken_distributions.append(idx)

        return (
            self.mu[idx],
            self.logvar[idx].exp()
        )




def lerp(a, b, t):
    return a + (b - a) * t


class Net(nn.Module):
    def __init__(self, n_mlp=8):
        super().__init__()

        layers = [PixelNorm()]

        for _i in range(n_mlp):
            layers.append(
                EqualLinear(
                    dimension, dimension, lr_mul=0.01, activation='fused_lrelu'
                )
            )

        self.net = nn.Sequential(*layers)


    def forward(self, x):
        return self.net(x)

class PPL():
    def __init__(self, g, num_samples=5, eps=1e-4):

        # Initializing the perceptual distance
        self.percept = lpips.PerceptualLoss(
            model='net-lin', net='vgg', use_gpu=True
        )
 
        self.num_samples = num_samples
        self.eps = eps
        self.g = g

    
    # Compute the loss
    def __call__(self, latent):
        dist = tdist.Normal(latent, torch.ones_like(latent) * 0.1)

        samples = dist.sample((self.num_samples,)).to(device)
        noise = self.g.make_noise()
        base_image, _ = self.g([latent], input_is_latent=True, noise=noise)  

        distances = torch.empty(0).to(device)

        for sample in samples:
            comparison = lerp(latent, sample, self.eps)
            del sample
            noise = self.g.make_noise()
            image, _ = self.g([comparison], input_is_latent=True, noise=noise)

            # Compute the distance
            perceptual_dist = self.percept(base_image, image).reshape(latent.shape[0])
            euclidean_dist = self.eps ** 2
            distance = perceptual_dist / euclidean_dist
            
            distances = torch.cat((distances, distance))
        
        loss = distances.mean() 

        return loss

class KL():
    def __init__(self):
        pass
    
    def __call__(self, source_mean, source_cov, target_mean, target_cov):
        # Compute the KL divergence
        term_1 = source_cov / target_cov
        term_2 = ((target_mean - source_mean) ** 2) / target_cov
        term_3 = torch.log(target_cov / source_cov)
        
        return (term_1 + term_2 + term_3).sum()
        

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--g_ckpt', type=str, required=True)
    parser.add_argument('--ckpt', type=str)
    parser.add_argument('--it', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--virtual_batch_size', type=int, default=16)
    parser.add_argument('--fitting_samples', type=int, default=2000)
    parser.add_argument('--checkpoint_it', type=int, default=1000)
    parser.add_argument('--kl_it', type=int, default=5000) # Number of iterations for the KL divergence minimization
    parser.add_argument('--ppl_it', type=int, default=1000) # Number of iterations for the PPL minimization
    parser.add_argument('projected_files', metavar='FILES', nargs='+')
    
    args = parser.parse_args()
    
    kl_loss_value = 0
    ppl_loss_value = 0

    # Loading the generator model
    g_ema = Generator(256, 512, 8)
    g_ema.load_state_dict(torch.load(args.g_ckpt)['g_ema'], strict=False)
    g_ema.eval()
    g_ema = g_ema.to(device)
    g = g_ema

    # Init the network
    net = Net().to(device)
    net.net.load_state_dict(g.style.state_dict())

    # Init the optimizer
    optim = torch.optim.Adam(net.parameters())
    optim.zero_grad()
    
    # Load the checkpoint
    if args.ckpt:
        net.load_state_dict(torch.load(args.ckpt)['net'])
        optim.load_state_dict(torch.load(args.ckpt)['optim'])

    # Extract the referent latent
    latents = []
    for projected_file in args.projected_files:
        projected = torch.load(projected_file)
        for key in projected:
            if key != "noises":
                latents.append(projected[key]["latent"])
    latent = latents[0].to(device)

    latents = [l.to(device) for l in latents]

    # Define the target distribution
    target_dist_mean = latents
    target_dist_cov = torch.ones((len(latents), latents[0].shape[0])).to(device) * 0.1

    # Init Loss function
    ppl_loss = PPL(g) 
    kl_loss = KL()
    
    # Train loop
    counter = 0
    kl_train = True
    pbar = tqdm(range(args.it))
    for i in pbar:
        
        # Alternate training
        if kl_train and counter == args.kl_it:
            counter = 0
            kl_train = False
        elif not kl_train and counter == args.ppl_it:
            counter = 0
            kl_train = True

        elif kl_train and counter < args.kl_it:
            # Prepare for the KL divergence
            inputs = torch.randn(args.fitting_samples, dimension).to(device)
            samples = net(inputs)

            # Fit a mixture of gaussian
            n_components = len(latents)
            model = GaussianMixture(samples, n_components=n_components)
            model.fit()


            # # Compute KL Loss for each pair of distributions
            loss = torch.zeros(1,).to(device)
            for index, target_mean_vector in enumerate(target_dist_mean):
                fitted_dist_mean, fitted_dist_cov = model.get_closest(target_mean_vector)
                loss += kl_loss(
                    fitted_dist_mean,
                    fitted_dist_cov,
                    target_dist_mean[index],
                    target_dist_cov[index]
                )
            loss = loss / len(target_dist_mean)

            kl_loss_value = loss.item()

            # Update step
            loss.backward()
            optim.step()
            optim.zero_grad()
            
            counter += 1

        else:
            # Compute the PPL loss
            inputs = torch.randn((args.batch_size, dimension)).to(device)                
            latent_out = net(inputs)
            loss = ppl_loss(latent_out)

            (loss / args.virtual_batch_size).backward()
            ppl_loss_value = loss.item()

            if i != 0 and i % args.virtual_batch_size == 0:
                optim.step()
                optim.zero_grad()

            counter += 1
        
        # Debug train status
        pbar.set_description(
            f"KL loss: {kl_loss_value:.2f} "
            f"PPL loss: {ppl_loss_value:.2f}"
        )

        # Checkpointing
        if i % args.checkpoint_it == 0:
            torch.save(
                {
                    'net': net.state_dict(),
                    'optim': optim.state_dict(),
                },
                f'mapping_net_checkpoint/{str(i).zfill(6)}.pt',
            )
        