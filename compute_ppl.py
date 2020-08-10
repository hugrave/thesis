import torch
import numpy as np
import argparse
from tqdm import tqdm

from sample_generation_net import PPL
from model import Generator



if __name__ == '__main__':
    device = 'cuda'

    parser = argparse.ArgumentParser()

    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--n_sample', type=int, default=2000)
    parser.add_argument('--size', type=int, default=256) 
    parser.add_argument('ckpt', metavar='CHECKPOINT', nargs="+")

    args = parser.parse_args()


    for ckpt_file in args.ckpt:

        # Load the generator
        ckpt = torch.load(ckpt_file)
        g = Generator(args.size, 512, 8).to(device)
        g.load_state_dict(ckpt['g_ema'])
        g.eval()

        # Load the ppl loss class
        ppl_loss = PPL(g, num_samples=5)


        # Divide the computation in batches
        n_batch = args.n_sample // args.batch
        resid = args.n_sample - (n_batch * args.batch)
        batch_sizes = [args.batch] * n_batch + [resid]

        ppl_values = []

        for batch in tqdm(batch_sizes):
            # Compute the average PPL for each batch
            with torch.no_grad():
                input_latents = torch.randn((batch, 512)).to(device)
                latents = g.get_latent(input_latents)
                ppl = ppl_loss(latents)
            ppl_values.append(ppl.to("cpu"))


        ppl_values = np.array(ppl_values)
        final_ppl = ppl_values.mean()

        print("Checkpoint:", ckpt_file, "PPL:", final_ppl)




