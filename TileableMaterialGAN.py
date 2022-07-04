import os.path as pth
import argparse
import torch
from models.stylegan2 import Generator


class TileableMaterialGAN:
    def __init__(self, args):
        self.args = args
        self.optim = args.optim
        self.device = args.device

        self.generator_params = get_generator_hyperparameters()
        self.generator = self.load_generator()

        init_latent = pth.join(args.in_dir, 'latent/init_latent.pt')
        init_noises = pth.join(args.in_dir, 'latent/init_noises.pt')
        if pth.exists(init_latent) and pth.exists(init_noises) and not args.rand_init:
            self.latent, self.noises = self.load(init_latent, init_noises)
            self.latent_type = 'preload'
        else:
            self.latent, self.noises = self.initialize_latent_and_noise()
            self.latent_type = 'random_mean'

    def initialize_latent_and_noise(self):
        # initialize latent
        mean_w = self.generator.mean_latent(n_latent=5000)
        if self.optim == 'w' or self.optim == 'wn':
            latent = mean_w.detach().clone()
            latent.requires_grad = True
        elif self.optim == 'w+' or self.optim == 'w+n':
            if self.generator_params.starting_height_size == 32:
                num_w = 9
            elif self.generator_params.starting_height_size == 4:
                num_w = 15 if self.generator_params.scene_size[0] == 512 else 13
            else:
                raise NotImplementedError(f'Unsupported starting_height_size: {self.generator_params.starting_height_size}')
            latent = mean_w.repeat(1, num_w, 1).detach().clone()
            latent.requires_grad = True
        else:
            latent = mean_w.detach().clone()

        # initialize noise
        noises = self.generator.make_noise()
        if 'n' in self.optim:
            for noise in noises:
                noise.requires_grad = True

        return latent, noises

    def save(self, out_dir, prefix=''):
        saved_latent = pth.join(out_dir, f'{prefix}latent.pt')
        saved_noises = pth.join(out_dir, f'{prefix}noises.pt')

        torch.save(self.latent, saved_latent)
        torch.save(self.noises, saved_noises)

    def load(self, init_latent, init_noises):
        latent = torch.load(init_latent).to(self.device)
        noises = torch.load(init_noises)
        for noise in noises:
            noise.to(self.device)
        return latent, noises

    def load_generator(self):
        g_ema = Generator(self.generator_params, self.device).to(self.device)
        print(f'Loading pretrained model from {self.args.gan_dir}')
        ckpt = torch.load(self.args.gan_dir, map_location=lambda storage, loc: storage)
        g_ema.load_state_dict(ckpt["g_ema"])

        return g_ema

    def get_optimizable(self):
        if self.optim == 'w' or self.optim == 'w+':
            optimizable = [self.latent]
        elif self.optim == 'n':
            optimizable = self.noises
        else:  # ['wn', 'w+n']
            optimizable = [self.latent] + self.noises
        return optimizable

    def __call__(self):
        dummy = torch.empty((1, 1, 1, 1), dtype=torch.float32, device=self.latent.device)
        if self.args.optim == 'n':
            output = self.generator(dummy, noise=self.noises)['image']
        elif self.args.optim == 'wn' or self.args.optim == 'w':
            output = self.generator(dummy, noise=self.noises, styles=self.latent, input_type='w')['image']
        else:  # ['w+n','w+']
            output = self.generator(dummy, noise=self.noises, styles=self.latent, input_type='w+')['image']

        return output


def get_generator_hyperparameters():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_size", type=tuple, default=(512, 512))
    parser.add_argument("--nc", type=int, default=9)
    parser.add_argument("--style_dim", type=int, default=512)
    parser.add_argument("--n_mlp", type=int, default=8)
    parser.add_argument("--channel_multiplier", type=int, default=2)
    parser.add_argument("--starting_height_size", type=int, default=4)
    parser.add_argument("--condv", type=str, default='1')
    parser.add_argument('--truncate_z', type=float, default=1.0)
    parser.add_argument('--no_cond', type=bool, default=True)
    parser.add_argument('--circular', type=bool, default=True)
    parser.add_argument('--circular2', type=bool, default=False)
    args = parser.parse_args()
    return args