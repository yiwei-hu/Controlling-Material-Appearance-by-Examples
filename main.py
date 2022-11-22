import os.path as pth
import random
import argparse

from transfer import transfer
from renderer import VideoGenerator, render_material_maps
from utils import Timer, make_tiled_material_maps

# AdobePatentID=“P11336-US

def get_steps_and_learning_rate(in_dir):
    # default
    steps, lr, proj_steps, proj_lr = 500, 0.02, 1000, 0.08

    if pth.basename(in_dir) == 'leather':
        '''
        If results from above steps/learning rate are not satisfactory (usually because material maps and target images
        have conflicting material properties, we recommend using 300/0.004
        '''
        steps, lr = 300, 0.004
    if pth.basename(in_dir) in ['brick0', 'brick1']:
        '''
        If projected material maps have artifacts, we recommend using a smaller learning rate
        '''
        proj_steps, proj_lr = 1500, 0.02

    return steps, lr, proj_steps, proj_lr


def get_light_and_camera(in_dir):
    # default
    light_and_camera = 'point'

    if pth.basename(in_dir) in ['grass', 'pavement']:
        '''
        Sometimes, using different light setting can get better results
        '''
        light_and_camera = 'dir'  # directional light

    return light_and_camera


def main():
    timer = Timer()
    in_dir = './sample/leather'  # brick0|floor|leather|grass|metal|wall|brick1|tile|pavement
    vgg_dir = './pretrained/vgg_conv.pt'
    gan_dir = './pretrained/250000.pt'

    steps, lr, proj_steps, proj_lr = get_steps_and_learning_rate(in_dir)
    light_and_camera = get_light_and_camera(in_dir)

    parser = argparse.ArgumentParser()
    parser.add_argument('--in_dir', type=str, default=in_dir, help='Location of input material maps and target images.')
    parser.add_argument('--out_dir', type=str, default=None, help='Location of output results.')
    parser.add_argument('--light_and_camera', type=str, default=light_and_camera, help='Renderer Configurations: path or predefined [point|dir]')
    parser.add_argument('--im_res', type=int, default=512, help='Resolutions of material map and target images.')
    parser.add_argument('--loss_fn', type=str, default='wasserstein', help='Loss function: cramer|wasserstein')
    parser.add_argument('--steps', type=int, default=steps, help='Number of optimization steps')
    parser.add_argument('--lr', type=float, default=lr, help='Learning rate')
    parser.add_argument('--proj_steps', type=int, default=proj_steps, help='Number of optimization steps')
    parser.add_argument('--proj_lr', type=float, default=proj_lr, help='Learning rate')
    parser.add_argument('--proj_normal_weight', type=float, default=10.0, help='Weight of normal map during projection') # 5
    parser.add_argument('--vis_every', type=int, default=100, help='Frequency of visualizing intermediate results.')
    parser.add_argument('--invert_normal', type=bool, default=False, help='Invert Y channel when loading normal maps.')
    parser.add_argument('--vgg_dir', type=str, default=vgg_dir, help='Location of pretrained VGG19')
    parser.add_argument('--gan_dir', type=str, default=gan_dir, help='Location of pretrained tileable MaterialGAN prior')
    parser.add_argument('--rand_init', type=bool, default=False, help='Optimize from random initialization.')
    parser.add_argument('--optim', type=str, default="w+n", help='Optimizable latent space.')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device')
    parser.add_argument('--seed', type=int, default=None, help='Seed for repeatability.')
    args = parser.parse_args()

    if args.light_and_camera in ['point', 'dir']:
        args.light_and_camera = './light/light_point' if args.light_and_camera == 'point' else './light/light_dir'

    if args.out_dir is None:
        args.out_dir = pth.join(in_dir, f'{pth.basename(args.light_and_camera)}_{args.loss_fn}')

    if args.seed is None:
        args.seed = random.randint(0, 2 ** 31 - 1)

    # optimization
    timer.begin()
    transfer(args)
    timer.end('Material transfer finished in ')

    # render dynamic lighting and tiled material maps
    VideoGenerator.animate_lighting(args.out_dir, args.out_dir, res=512, suffix='_optim', r=24, n_round=2, n_seconds=3, frame_per_seconds=30)

    make_tiled_material_maps(args.out_dir, suffixes=['_optim', '_init', '_input'])
    render_material_maps(pth.join(args.out_dir, 'tiled_optim'))
    render_material_maps(pth.join(args.out_dir, 'tiled_init'))
    render_material_maps(pth.join(args.out_dir, 'tiled_input'))


if __name__ == '__main__':
    main()