import os.path as pth
import torch
import torch.nn as nn
import kornia
import cv2
import imageio
from utils import read_image, save_image, MaterialLoader, TextureOps
from utils_base import *

default_dir_light_config_path = './light/light_dir'
default_point_light_config_path = './light/light_point'


def load_light_config(in_dir, verbose=False):
    light_type = np.loadtxt(os.path.join(in_dir, 'light_type.txt'), delimiter=',')
    light_type = int(light_type)

    light_pos = np.loadtxt(os.path.join(in_dir, 'light_pos.txt'), delimiter=',').astype(np.float32)
    camera_pos = np.loadtxt(os.path.join(in_dir, 'camera_pos.txt'), delimiter=',').astype(np.float32)

    im_size = np.loadtxt(os.path.join(in_dir, 'image_size.txt'), delimiter=',')
    im_size = float(im_size)
    
    light = np.loadtxt(os.path.join(in_dir, 'light_power.txt'), delimiter=',').astype(np.float32)

    light_pos = light_pos.reshape((-1, 3))
    camera_pos = camera_pos.reshape((-1, 3))

    light_config = {'light_pos': light_pos, 'camera_pos': camera_pos, 'im_size': im_size, 'light': light, 'light_type': light_type}

    if verbose:
        print('Load camera position from ', os.path.join(in_dir, 'camera_pos.txt'))
        print('Load light position from ', os.path.join(in_dir, 'light_pos.txt'))

    return light_config


class Renderer(nn.Module):
    def print_light_config(self):
        print('Renderer Configuration: ')
        print('Is light optimizable:', self.optimizable)
        print('Is directional light: ', self.is_dir)
        if self.is_dir:
            print('Light Direction:', self.lp)
        else:
            print('Light Position:', self.lp)
        print('Camera Position:', self.cp)
        print('Image size:', self.im_size)
        print('Light power:', self.li)

    def __init__(self, res, light_config=None, preload='point', optimizable=False, device=torch.device('cuda:0')):
        super(Renderer, self).__init__()

        # load default light setting if not specified
        if light_config is None:
            if preload == 'dir':
                light_config = load_light_config(default_dir_light_config_path)
            elif preload == 'point':
                light_config = load_light_config(default_point_light_config_path)
            else:
                raise RuntimeError('Preload config not specified.')
        elif isinstance(light_config, str):
            light_config = load_light_config(light_config)
        else:
            assert(isinstance(light_config, dict))

        self.lp = torch.as_tensor(light_config['light_pos'], dtype=torch.float32, device=device)
        self.cp = torch.as_tensor(light_config['camera_pos'], dtype=torch.float32, device=device)
        self.im_size = light_config['im_size']
        self.li = torch.as_tensor(light_config['light'][0], dtype=torch.float32, device=device)

        self.optimizable = optimizable
        if optimizable:
            self.lp = nn.Parameter(self.lp)
            self.cp = nn.Parameter(self.cp)
            self.li = nn.Parameter(self.li)
            
        light_type = light_config['light_type']
        if light_type == 0: # point light
            self.is_dir = False
        elif light_type == 1:
            self.is_dir = True
        else:
            raise RuntimeError("Unknown light type")

        self.n_light = self.lp.shape[0]
        self.shader = Microfacet(res=res, size=self.im_size)

    def forward(self, svbrdfs, format):
        return self.render(svbrdfs, format)

    def render(self, svbrdfs, format):
        render_targets = []
        for i in range(self.n_light):
            render = self.shader.eval(svbrdfs, lightPos=self.lp[i, :], cameraPos=self.cp[i, :], light=self.li, is_dir=self.is_dir, format=format)
            render_targets.append(render)

        render_targets = torch.cat(render_targets, dim=0)
        return render_targets

    def regularize(self):
        # if optimizable, clamp light_pos and camera_pos to make them above the sample
        if self.optimizable:
            self.lp[:, 2].clamp_(0)
            self.cp[:, 2].clamp_(0)
    

class Microfacet:
    def __init__(self, res, size, f0=0.04):
        self.res = res
        self.size = size
        self.f0 = f0
        self.eps = 1e-6

        self.initGeometry()

    def initGeometry(self):
        tmp = torch.arange(self.res, dtype=torch.float32).cuda()
        tmp = ((tmp + 0.5) / self.res - 0.5) * self.size
        y, x = torch.meshgrid((tmp, tmp))
        self.pos = torch.stack((x, -y, torch.zeros_like(x)), 2)
        self.pos_norm = self.pos.norm(2.0, 2, keepdim=True)

    def GGX(self, cos_h, alpha):
        c2 = cos_h ** 2
        a2 = alpha ** 2
        den = c2 * a2 + (1 - c2)
        return a2 / (np.pi * den**2 + self.eps)

    def Beckmann(self, cos_h, alpha):
        c2 = cos_h ** 2
        t2 = (1 - c2) / c2
        a2 = alpha ** 2
        return torch.exp(-t2 / a2) / (np.pi * a2 * c2 ** 2)

    def Fresnel(self, cos, f0):
        return f0 + (1 - f0) * (1 - cos)**5

    def Fresnel_S(self, cos, specular):
        sphg = torch.pow(2.0, ((-5.55473 * cos) - 6.98316) * cos)
        return specular + (1.0 - specular) * sphg

    def Smith(self, n_dot_v, n_dot_l, alpha):
        def _G1(cos, k):
            return cos / (cos * (1.0 - k) + k)

        k = alpha * 0.5 + self.eps
        return _G1(n_dot_v, k) * _G1(n_dot_l, k)

    def normalize(self, vec):
        assert(vec.size(0)==self.N)
        assert(vec.size(1)==3)
        assert(vec.size(2)==self.res)
        assert(vec.size(3)==self.res)

        vec = vec / (vec.norm(2.0, 1, keepdim=True))
        return vec

    def getDir(self, pos, is_dir=False):
        if is_dir:
            vec = pos.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand(self.N, -1, self.res, self.res)
            return self.normalize(vec), 1
        else:
            vec = (pos - self.pos).permute(2,0,1).unsqueeze(0).expand(self.N,-1,-1,-1)
            return self.normalize(vec), (vec**2).sum(1, keepdim=True).expand(-1,3,-1,-1)

    def AdotB(self, a, b):
        ab = (a*b).sum(1, keepdim=True).clamp(min=0).expand(-1,3,-1,-1)
        return ab

    def eval(self, textures, lightPos, cameraPos, light, is_dir, format):
        # unpack material maps
        if format == 'MaterialGAN':
            albedo, normal, rough, specular = TextureOps.tex2map(textures)
        elif format == 'Images':
            albedo, normal, rough, specular = textures
            normal = normal * 2 - 1 # (0, 1) -> (-1, 1)
        else:
            raise RuntimeError('Unknown texture format')

        self.N = albedo.shape[0]

        # assume albedo in gamma space
        albedo = albedo ** 2.2

        light = torch.stack((light, light, light))
        light = light.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(albedo)

        v, _ = self.getDir(cameraPos)
        l, dist_l_sq = self.getDir(lightPos, is_dir=is_dir)
        h = self.normalize(l + v)
        
        n_dot_v = self.AdotB(normal, v)
        n_dot_l = self.AdotB(normal, l)
        n_dot_h = self.AdotB(normal, h)
        v_dot_h = self.AdotB(v, h)

        geom = n_dot_l / dist_l_sq

        D = self.GGX(n_dot_h, rough**2)
        # D = self.Beckmann(n_dot_h, rough**2)
        F = self.Fresnel_S(v_dot_h, specular)

        G = self.Smith(n_dot_v, n_dot_l, rough**2)

        # lambert brdf
        f1 = albedo / np.pi * (1 - specular)

        # cook-torrence brdf
        f2 = D * F * G / (4 * n_dot_v * n_dot_l + self.eps)

        # brdf
        kd = 1; ks = 1
        f = kd * f1 + ks * f2

        # rendering
        img = f * geom * light

        return img.clamp(self.eps, 1) ** (1/2.2)


class Height2Normal(nn.Module):
    def __init__(self, intensity):
        super(Height2Normal, self).__init__()
        self.intensity = intensity

    @staticmethod
    def normalize(m):
        return m / torch.sqrt(torch.sum(m ** 2, dim=1, keepdim=True))

    def forward(self, height):
        gradient = kornia.spatial_gradient(height, normalized=False)
        dx = gradient[:, :, 0, :, :]
        dy = gradient[:, :, 1, :, :]

        x = -dx*self.intensity
        y = dy*self.intensity
        z = torch.ones_like(dx)
        normal = torch.cat((x, y, z), dim=1)
        normal = Height2Normal.normalize(normal)  # (-1, 1)

        return normal


class VideoGenerator:
    @staticmethod
    def animate_lighting(in_dir, out_dir, res, suffix='', r=12, phi=np.pi/12,
                         n_round=1, n_seconds=5, frame_per_seconds=30):
        if not pth.exists(out_dir):
            os.makedirs(out_dir)

        material_loader = MaterialLoader(res, device=torch.device('cuda:0'))
        svbrdf, _, _ = material_loader.load(in_dir, use_default=True, suffix=suffix)
        svbrdf = material_loader.remapping(svbrdf)

        # compute light sequence
        n_frames = n_seconds*frame_per_seconds
        frames_per_round = int(n_frames / n_round)
        theta = np.linspace(0, 2*np.pi, num=frames_per_round, endpoint=False)
        theta = np.tile(theta, n_round)
        phi = np.ones_like(theta)*phi
        x = r*np.cos(theta)*np.sin(phi)
        y = r*np.sin(theta)*np.sin(phi)
        z = r*np.cos(phi)

        lp = np.stack((y, x, z), axis=1).astype(np.float32)
        cp = np.zeros_like(lp)
        cp[:, 2] = r

        # init lighting
        im_size = 30
        li = np.asarray([3300, 3300, 3300], dtype=np.float32)
        light_config = {'light_pos': lp, 'camera_pos': cp, 'im_size': im_size,
                        'light': li, 'light_type': 0}

        renderer = Renderer(res, light_config=light_config)
        with torch.no_grad():
            render_targets = renderer.render(svbrdf, format='MaterialGAN')

        VideoGenerator.save_video(render_targets, out_dir, suffix)

    @staticmethod
    def save_video(images, out_dir, suffix):
        n_frame = images.shape[0]
        video = imageio.get_writer(f'{out_dir}/render{suffix}.mp4', mode='I', fps=30, codec='libx264', bitrate='16M')
        for i in range(n_frame):
            im = TextureOps.image_to_numpy(images[i])
            im255 = (im * 255.0).astype(np.uint8)
            video.append_data(im255)
        video.close()


def render_material_maps(input_path, preload='point', suffixes=('', ), invert_normal=False):
    def read_and_rescale_image(filepath, res=0, device=torch.device('cuda:0')):
        im = read_image(filepath)
        if res != 0 and im.shape[0] != res:
            im = cv2.resize(im, dsize=(res, res))
        if im.ndim == 2:
            im = np.expand_dims(im, axis=2)
        if im.shape[2] == 4:
            im = im[:, :, :3]
        im = torch.as_tensor(im, dtype=torch.float32, device=device).permute(2, 0, 1).unsqueeze(0)
        return im

    for suffix in suffixes:
        albedo = read_and_rescale_image(pth.join(input_path, f'albedo{suffix}.png'))
        normal = read_and_rescale_image(pth.join(input_path, f'normal{suffix}.png'))
        roughness = read_and_rescale_image(pth.join(input_path, f'roughness{suffix}.png'))
        specular = read_and_rescale_image(pth.join(input_path, f'specular{suffix}.png'))

        if invert_normal:
            normal[:, 1, :, :] = 1 - normal[:, 1, :, :]

        svbrdfs = (albedo, normal, roughness, specular)
        renderer = Renderer(res=albedo.shape[2], preload=preload)
        render = renderer.render(svbrdfs, format='Images')
        save_image(render, pth.join(input_path, f'render{suffix}.png'))