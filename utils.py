import os.path as pth
import time
import imageio
import torch
from torchvision.transforms import ToTensor
import warnings
import cv2
import json
import matplotlib.pyplot as plt
from utils_base import *


def save_args(args, save_path):
    with open(save_path, 'w') as f:
        json.dump(args.__dict__, f, indent=2)


def record_loss(loss_list, losses):
    for name in losses.keys():
        if name not in loss_list:
            loss_list[name] = []
        loss_list[name].append(losses[name])


def plot_and_save(loss_list, save_dir):
    plt.figure(figsize=(8,4))
    for name in loss_list.keys():
        plt.plot(loss_list[name], label=name)
    plt.legend()
    plt.savefig(save_dir)
    plt.close()


def render_and_save(svbrdf, save_dir, tmp_dir, step, renderers, suffix=''):
    fn = os.path.join(save_dir, f'tex{suffix}.png')
    fn2 = os.path.join(save_dir, f'rendered{suffix}.png')
    png = TextureOps.tex2png(svbrdf, fn)

    renders = []
    for renderer in renderers:
        render = renderer(svbrdf, format='MaterialGAN')
        renders.append(render)

    render_all = None
    for render in renders:
        render = gyTensor2Array(render[0].permute(1, 2, 0))
        render = gyArray2PIL(render)
        render_all = gyConcatPIL_h(render_all, render)
        png = gyConcatPIL_h(png, render)

    render_all.save(fn2)
    png.save(os.path.join(tmp_dir, f'step_{step:05d}{suffix}.jpg'))


def examine_novel_view(svbrdf, save_dir, renderer, suffix=''):
    fn = os.path.join(save_dir, f'novel_view{suffix}.png')

    renders = renderer(svbrdf, format='MaterialGAN')
    n_renders = renders.shape[0]

    render_all = None
    for i in range(n_renders):
        render = gyTensor2Array(renders[i].permute(1, 2, 0))
        render = gyArray2PIL(render)
        render_all = gyConcatPIL_h(render_all, render)

    render_all.save(fn)


def save_material_maps(svbrdf, out_dir, suffix=''):
    albedo, normal, roughness, metallic = TextureOps.tex2map(svbrdf)
    normal = (normal + 1) / 2
    save_image(albedo, pth.join(out_dir, f'albedo{suffix}.png'))
    save_image(normal, pth.join(out_dir, f'normal{suffix}.png'))
    save_image(roughness, pth.join(out_dir, f'roughness{suffix}.png'))
    save_image(metallic, pth.join(out_dir, f'specular{suffix}.png'))


class TextureOps:
    @staticmethod
    def image_to_numpy(image):
        return image.detach().cpu().squeeze().permute(dims=(1, 2, 0)).numpy()

    @staticmethod
    def tex2map(tex):
        eps = 1e-6

        albedo = ((tex[:, 0:3, :, :].clamp(-1, 1) + 1) / 2)  # (-1, 1) -> (0, 1)

        normal_x = tex[:, 3, :, :].clamp(-1, 1)
        normal_y = tex[:, 4, :, :].clamp(-1, 1)
        normal_xy = (normal_x ** 2 + normal_y ** 2).clamp(min=0, max=1 - eps)
        normal_z = (1 - normal_xy).sqrt()
        normal = torch.stack((normal_x, normal_y, normal_z), 1)
        normal = normal.div(normal.norm(2.0, 1, keepdim=True))  # (-1, 1)

        # rough = ((tex[:, 5, :, :].clamp(-0.3, 1) + 1) / 2)  # (-0.3, 1) -> (0.35, 1)
        rough = ((tex[:, 5, :, :].clamp(-1, 1) + 1) / 2)  # (-1, 1) -> (0, 1)
        rough = rough.clamp(min=eps).unsqueeze(1).expand(-1, 3, -1, -1)

        if tex.shape[1] == 9:
            specular = ((tex[:, 6:9, :, :].clamp(-1, 1) + 1) / 2)  # (-1, 1) -> (0, 1)
            return albedo, normal, rough, specular
        else:
            return albedo, normal, rough

    @staticmethod
    def tex2png(tex, fn, isVertical=False):
        isSpecular = False
        if tex.size(1) == 9:
            isSpecular = True

        if isSpecular:
            albedo, normal, rough, specular = TextureOps.tex2map(tex)
        else:
            albedo, normal, rough = TextureOps.tex2map(tex)

        albedo = gyTensor2Array(albedo[0, :].permute(1, 2, 0))
        normal = gyTensor2Array((normal[0, :].permute(1, 2, 0) + 1) / 2)  # (-1, 1) -> (0, 1)
        rough = gyTensor2Array(rough[0, :].permute(1, 2, 0))

        albedo = gyArray2PIL(albedo)
        normal = gyArray2PIL(normal)
        rough = gyArray2PIL(rough)

        if isVertical:
            png = gyConcatPIL_v(gyConcatPIL_v(albedo, normal), rough)
        else:
            png = gyConcatPIL_h(gyConcatPIL_h(albedo, normal), rough)

        if isSpecular:
            specular = gyTensor2Array(specular[0, :].permute(1, 2, 0))
            specular = gyArray2PIL(specular)
            if isVertical:
                png = gyConcatPIL_v(png, specular)
            else:
                png = gyConcatPIL_h(png, specular)

        if fn is not None:
            png.save(fn)
        return png

    @staticmethod
    def substance2map(tex):
        # unpack
        basecolor = tex['baseColor']
        normal = tex['normal']
        roughness = tex['roughness']
        metallic = tex['metallic']

        # Remove alpha channels and convert grayscale/color if necessary
        if basecolor.shape[1] == 4:
            basecolor = basecolor[:, :3, :, :]
        elif basecolor.shape[1] == 1: # basecolor should always have 3 channels
            basecolor = basecolor.expand(-1, 3, -1, -1)

        if normal.shape[1] == 4:
            normal = normal[:, :3, :, :]
        elif normal.shape[1] == 1: # normal should always have 3 channels
            normal = normal.expand(-1, 3, -1, -1)
        if roughness.shape[1] != 1: # roughness should always have 1 channel (also ignore the alpha channel)
            roughness = roughness[:, :3, :, :].mean(dim=1, keepdim=True)
        if metallic.shape[1] != 1: # metallic should always have 1 channel (also ignore the alpha channel)
            metallic = metallic[:, :3, :, :].mean(dim=1, keepdim=True)

        # reparametrized normal
        normal = normal * 2.0 - 1.0 # (0, 1) -> (-1, 1)

        return basecolor, normal, roughness, metallic

    @staticmethod
    def tex2im(tex):
        maps = list(TextureOps.tex2map(tex))
        maps[1] = maps[1] * 0.5 + 0.5  # normal map: (-1, 1) -> (0, 1)
        return maps


class MaterialLoader:
    def __init__(self, res, device):
        self.res = res
        self.device = device

    def load(self, data_path, use_default=True, suffix='', invert_normal=False):
        config_file = pth.join(data_path, 'config.json')
        if not pth.exists(config_file):
            return self.load_(data_path, use_default=use_default, suffix=suffix, invert_normal=invert_normal)

        with open(config_file) as f:
            config = json.load(f)

        albedo = self.load_image(data_path, f'albedo{suffix}')
        normal = self.load_image(data_path, f'normal{suffix}')
        roughness = self.load_image(data_path, f'roughness{suffix}', use_default, 0.5)
        specular = self.load_image(data_path, f'specular{suffix}', use_default, 0.04)

        if invert_normal:
            normal[:, 1, :, :] = 1 - normal[:, 1, :, :]

        num_targets = config['num_targets']
        if config['num_targets'] == 1:
            try:
                target_images = self.load_image(data_path, 'target')
            except FileNotFoundError:
                warnings.warn('Cannot find target images')
                target_images = None
        else:
            target_images = torch.zeros((num_targets, 3, self.res, self.res), device=self.device)
            for i in range(num_targets):
                target_images[i, ...] = self.load_image(data_path, f'target{i}')

        svbrdf = torch.cat((albedo, normal[:, 0:2, ...], roughness.mean(dim=1, keepdim=True), specular), dim=1)

        if 'labels' in config:
            material_label = self.load_label(data_path, f'mat_mask{suffix}')
            if num_targets == 1:
                target_label = self.load_label(data_path, f'target_mask{suffix}')
                target_labels = [target_label]
            else:
                target_labels = []
                for i in range(num_targets):
                    target_label = self.load_label(data_path, f'target_mask{suffix}{i}')
                    target_labels.append(target_label)
            label_maps = (material_label, target_labels, config['labels'])
        else:
            label_maps = None

        return svbrdf, target_images, label_maps

    def load_(self, data_path, num_target_images=0, use_default=True, suffix='', invert_normal=False):
        albedo = self.load_image(data_path, f'albedo{suffix}')
        normal = self.load_image(data_path, f'normal{suffix}')
        roughness = self.load_image(data_path, f'roughness{suffix}', use_default, 0.5)
        specular = self.load_image(data_path, f'specular{suffix}', use_default, 0.04)

        if invert_normal:
            normal[:, 1, :, :] = 1 - normal[:, 1, :, :]

        target_images = None

        if num_target_images == 0:
            try:
                target_images = self.load_image(data_path, 'target')
            except FileNotFoundError:
                warnings.warn('Cannot find target images')
        else:
            target_images = torch.zeros((num_target_images, 3, self.res, self.res))
            for i in range(num_target_images):
                target_images[i, ...] = self.load_image(data_path, f'target{i}')
            target_images = target_images.cuda()

        svbrdf = torch.cat((albedo, normal[:, 0:2, ...], roughness.mean(dim=1, keepdim=True), specular), dim=1)

        # load a binary label map if exist
        try:
            material_mask = self.load_label(data_path, f'mat_mask{suffix}')
            image_mask = self.load_label(data_path, f'target_mask{suffix}')
            labels = (material_mask, image_mask)
        except FileNotFoundError:
            warnings.warn('Failed to find label maps!')
            labels = None

        return svbrdf, target_images, labels

    @staticmethod
    def get_filename(data_path, map_name, exts=('.png', '.jpg', '.jpeg', '.bmp')):
        for ext in exts:
            filename = pth.join(data_path, map_name + ext)
            if pth.exists(filename):
                return filename
        raise FileNotFoundError

    def load_image(self, data_path, map_name, use_default=False, default_value=0.0):
        try:
            filename = self.get_filename(data_path, map_name)
        except FileNotFoundError as e:
            if use_default:
                warnings.warn(f"Use default value {default_value} for {map_name} map")
                im = torch.ones((1, 3, self.res, self.res), device=self.device)*default_value
                return im
            else:
                raise e

        im = Image.open(filename).convert('RGB')
        if im.width != self.res:
            im = im.resize((self.res, self.res), resample=Image.LANCZOS)
        im = ToTensor()(im).unsqueeze(0).to(self.device)
        return im

    def load_label(self, data_path, map_name):
        mask_filename = self.get_filename(data_path, map_name)
        mask = np.asarray(Image.open(mask_filename))
        if mask.shape[0] != self.res:
            mask = cv2.resize(mask, dsize=(self.res, self.res), interpolation=cv2.INTER_NEAREST)
        return mask

    @staticmethod
    def remapping(svbrdf):
        svbrdf = svbrdf * 2.0 - 1.0  # (0, 1) -> (-1, 1)
        return svbrdf


class Timer:
    def __init__(self):
        self.start_time = []

    def begin(self, output=''):
        if output != '':
            print(output)
        self.start_time.append(time.time())

    def end(self, output=''):
        if len(self.start_time) == 0:
            raise RuntimeError("Timer stack is empty!")
        t = self.start_time.pop()
        elapsed_time = time.time() - t
        print(output, time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

    def lap(self, output=''):
        if len(self.start_time) == 0:
            raise RuntimeError("Timer stack is empty!")
        t = self.start_time[-1]
        elapsed_time = time.time() - t
        print(output, time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))


# save a torch tensor to image file
def save_image(im, filename):
    assert(im.ndim == 4 and im.shape[0] == 1 and (im.shape[1] == 1 or im.shape[1] == 3))
    im = im.squeeze(0).permute(1, 2, 0).clamp(0, 1).detach().cpu().numpy()

    if im.shape[2] == 1:
        im = np.concatenate((im, im, im), axis=2)
    im = Image.fromarray((im*255).astype(np.uint8))
    im.save(filename)


def read_image(filename: str):

    # See the following link for installing the OpenEXR plugin for imageio:
    # https://imageio.readthedocs.io/en/stable/format_exr-fi.html

    img = imageio.imread(filename)
    if img.dtype == np.float32:
        return img
    elif img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    elif img.dtype == np.uint16:
        return img.astype(np.float32) / 65535.0
    else:
        raise RuntimeError('Unexpected image data type.')


def write_image(filename: str, img, check=False):

    # See the following link for installing the OpenEXR plugin for imageio:
    # https://imageio.readthedocs.io/en/stable/format_exr-fi.html

    if check:
        assert (np.all(img >= 0) and np.all(img <= 1))

    extension = os.path.splitext(filename)[1]
    if extension == '.exr':
        imageio.imwrite(filename, img)
    elif extension in ['.png', '.jpg']:
        imageio.imwrite(filename, (img * 255.0).astype(np.uint8))
    else:
        raise RuntimeError(f'Unexpected image filename extension {extension}.')


def resample(m, shape, erosion, width=3):
    m0 = cv2.resize(m, dsize=shape, interpolation=cv2.INTER_NEAREST)
    m1 = cv2.erode(m0, np.ones((width, width), np.uint8)) if erosion else m0
    return m1


def make_tile(image, n):
    if n <= 1:
        return image

    return np.tile(image, (n, n, 1))


def make_tiled_material_maps(in_dir, suffixes, n_tile=2):
    def tile_it(map_name):
        material_map = read_image(pth.join(in_dir, f'{map_name}{suffix}.png'))
        tiled = make_tile(material_map, n_tile)
        write_image(pth.join(out_dir, f'{map_name}.png'), tiled)

    for suffix in suffixes:
        out_dir = pth.join(in_dir, f'tiled{suffix}')
        os.makedirs(out_dir, exist_ok=True)

        tile_it('albedo')
        tile_it('normal')
        tile_it('roughness')
        tile_it('specular')


def make_material_grid(input_path, suffixes=('', )):
    for suffix in suffixes:
        albedo = read_image(pth.join(input_path, f'albedo{suffix}.png'))
        normal = read_image(pth.join(input_path, f'normal{suffix}.png'))
        roughness = read_image(pth.join(input_path, f'roughness{suffix}.png'))
        if roughness.ndim == 2:
            roughness = np.stack((roughness, roughness, roughness), axis=2)
        specular = read_image(pth.join(input_path, f'specular{suffix}.png'))

        im0 = np.concatenate((albedo, normal), axis=1)
        im1 = np.concatenate((roughness, specular), axis=1)
        im = np.concatenate((im0, im1), axis=0)
        write_image(pth.join(input_path, f'grid{suffix}.png'), im)