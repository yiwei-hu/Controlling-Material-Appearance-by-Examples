import torch.nn.functional as F
import vgg
from utils import resample, write_image
from renderer import *
from wasserstein import WassersteinLoss, CramerLoss


# used for projecting material maps into latent space of MaterialGAN
class EmbeddingLosses:
    def __init__(self, args, svbrdf_ref, renderer=None, scales=(0, )):
        self.args = args
        self.svbrdf_ref = svbrdf_ref
        self.svbrdf_maps = TextureOps.tex2im(svbrdf_ref)

        self.criterion = nn.L1Loss().cuda()
        self.renderer = renderer
        self.scales = scales

        # define losses
        self.vgg19 = vgg.FeatureExtractor(args.vgg_dir, args.device)
        self.feature_layers = ['r12', 'r22', 'r32', 'r42']
        self.feature_weights = [1e-3, 1e-3, 1e-3, 1e-3]
        self.criterion_feature = WeightedLoss(self.feature_weights, 'l1')

        self.map_weights = [1, args.proj_normal_weight, 1, 1]
        self.render_loss_weight = 1.0

    def schedule_weights(self, alpha):
        weights = self.map_weights.copy()
        if alpha is not None:
            weights[1] = 1 + alpha * (self.map_weights[1] - 1)
        return weights

    def eval(self, svbrdf, alpha=None):
        svbrdf_maps = TextureOps.tex2im(svbrdf)

        weights = self.schedule_weights(alpha)
        pixel_loss, feature_loss = 0, 0
        for svbrdf_map, svbrdf_map_ref, weight in zip(svbrdf_maps, self.svbrdf_maps, weights):
            pixel_l = []
            feature_l = []
            for scale in self.scales:
                x = F.interpolate(svbrdf_map, scale_factor=1. / 2 ** scale, mode='bilinear') if scale > 0 else svbrdf_map
                y = F.interpolate(svbrdf_map_ref, scale_factor=1. / 2 ** scale, mode='bilinear') if scale > 0 else svbrdf_map_ref
                feat = self.vgg19(x, self.feature_layers)
                ref = self.vgg19(y, self.feature_layers, detach=True)
                pixel_l.append(self.criterion(x, y))
                feature_l.append(self.criterion_feature(feat, ref))

            pixel_loss += sum(pixel_l) / len(pixel_l) * weight
            feature_loss += sum(feature_l) / len(feature_l) * weight

        pixel_loss = pixel_loss / sum(weights)
        feature_loss = feature_loss / sum(weights)

        loss = pixel_loss + feature_loss

        if self.renderer:
            render = self.renderer.render(svbrdf, format='MaterialGAN')
            render_ref = self.renderer.render(self.svbrdf_ref, format='MaterialGAN')

            render_feat = self.vgg19(render, self.feature_layers)
            render_feat_ref = self.vgg19(render_ref, self.feature_layers, detach=True)

            render_loss = self.criterion(render_ref, render) + self.criterion_feature(render_feat, render_feat_ref)
            loss += render_loss * self.render_loss_weight
        else:
            render_loss = torch.tensor([0])

        return loss, {'pixel': pixel_loss.item(), 'feature': feature_loss.item(), 'render': render_loss.item(), 'tot': loss.item()}


# used for material transfer optimization
class StyleTransferLosses:
    def __init__(self, args, svbrdf_ref, target_images, labels, renderer, guidance_maps=None, scales=(0, )):
        self.args = args
        self.svbrdf_ref = svbrdf_ref
        self.svbrdf_maps = TextureOps.tex2im(svbrdf_ref)
        self.target_images = target_images
        self.guidance_maps = guidance_maps if guidance_maps else compute_guidance_maps(labels)
        self.scales = scales

        # renderer settings
        self.renderer = renderer

        # VGG19 settings
        self.vgg19 = vgg.FeatureExtractor(args.vgg_dir, args.device)
        self.style_layers = ['r11', 'r21', 'r31', 'r41']
        self.style_weights = [5, 5, 5, 0.5]
        self.content_layers = ['r42']
        self.content_weights = [1]

        # pre-compute features
        self.content_targets = []
        for svbrdf_map in self.svbrdf_maps:
            content_target = self.vgg19(svbrdf_map, self.content_layers, detach=True)
            self.content_targets.append(content_target)

        # define loss functions
        self.criterion_feature = WeightedLoss(self.content_weights, metric='l1')
        if args.loss_fn == 'wasserstein':
            self.criterion_style = WassersteinLoss(nb_dirs=0, metric='l1')
        elif args.loss_fn == 'cramer':
            self.criterion_style = CramerLoss(nb_dirs=0, p=1)
        else:
            raise RuntimeError(f"Unknown loss functions for transfer: {args.loss_fn}")

    def content_loss(self, svbrdf_maps, weights):
        content_loss = 0
        for svbrdf_map, ref, weight in zip(svbrdf_maps, self.content_targets, weights):
            feat = self.vgg19(svbrdf_map, self.content_layers)
            content_loss = content_loss + self.criterion_feature(feat, ref) * weight
        content_loss /= len(weights)
        return content_loss

    def get_style_layers_and_weights(self, scale):
        if scale == 0:
            return self.style_layers, self.style_weights
        else:
            return ['r11', 'r21'], [5, 5]

    def multi_scale_style_loss(self, svbrdf, scales=(0,)):
        renders = self.renderer(svbrdf, format='MaterialGAN')

        multi_scale_losses = []
        for scale in scales:
            x = F.interpolate(renders, scale_factor=1./2**scale, mode='bilinear') if scale > 0 else renders
            y = F.interpolate(self.target_images, scale_factor=1./2**scale, mode='bilinear') if scale > 0 else self.target_images
            style_layers, style_weights = self.get_style_layers_and_weights(scale)
            loss, loss_vals = self.style_loss(x, y, style_layers, style_weights)
            multi_scale_losses.append(loss)

        acc_loss = sum(multi_scale_losses) / len(multi_scale_losses)
        return acc_loss

    def style_loss(self, x, y, style_layers, style_weights):
        render_features = self.vgg19(x, style_layers)
        target_features = self.vgg19(y, style_layers, detach=True)

        losses = []
        loss_vals = []

        if len(self.guidance_maps) == 0:
            erosion, width = False, [0, 0]
        else:
            erosion, width = True, [5, 5]

        for render_feature, target_feature, weight in zip(render_features, target_features, style_weights):
            # downsample guidance maps and generate index maps
            _, _, h, w = render_feature.shape
            index_maps = []
            for mat_map, i, target_map in self.guidance_maps:
                mat_map_ = resample(mat_map, (h, w), erosion=erosion, width=width[0]).flatten()
                mat_idx = np.where(mat_map_ == 1)[0]
                if mat_idx.size <= 0:
                    continue
                mat_idx = torch.as_tensor(mat_idx, device=render_feature.device)

                target_map_ = resample(target_map, (h, w), erosion=erosion, width=width[1]).flatten()
                target_idx = np.where(target_map_ == 1)[0]
                if target_idx.size <= 0:
                    continue
                target_idx = torch.as_tensor(target_idx, device=target_feature.device)

                # resampling
                if mat_idx.shape[0] != target_idx.shape[0] and self.args.loss_fn == 'wasserstein':
                    size = min(mat_idx.shape[0], target_idx.shape[0])
                    if mat_idx.shape[0] != size:
                        index = torch.randint(mat_idx.shape[0], (size, ))
                        mat_idx = mat_idx[index]
                    if target_idx.shape[0] != size:
                        index = torch.randint(target_idx.shape[0], (size, ))
                        target_idx = target_idx[index]

                # mat_idx: index on material maps,
                # i: ith target images
                # target_idx: index on target images
                index_maps.append((mat_idx, i, target_idx))

            if len(index_maps) == 0:
                index_maps = None

            # if we have guidance maps, but the no pixel left after erosion, we do not compute loss at this layer
            if index_maps is None and len(self.guidance_maps) > 0:
                continue

            loss = self.criterion_style(render_feature, target_feature, index_maps) * weight
            losses.append(loss)
            loss_vals.append(loss.item())

        style_loss = sum(losses)

        return style_loss, loss_vals

    def eval(self, svbrdf):
        svbrdf_maps = TextureOps.tex2im(svbrdf)

        content_loss = self.content_loss(svbrdf_maps, weights=(1, 1, 1, 1))
        style_loss = self.multi_scale_style_loss(svbrdf, scales=self.scales)
        loss = content_loss + style_loss

        return loss, {'content': content_loss.item(), 'style': style_loss.item(), 'tot': loss.item()}


# weighted feature loss
class WeightedLoss(nn.Module):
    def __init__(self, weights, metric='l2'):
        super(WeightedLoss, self).__init__()
        self.weights = weights
        assert metric in ['l1', 'l2']
        self.criterion = nn.L1Loss() if metric == 'l1' else nn.MSELoss()

    def forward(self, x, y):
        tot_loss = 0
        for w, x_, y_ in zip(self.weights, x, y):
            loss = self.criterion(x_, y_)
            tot_loss = tot_loss + w * loss

        return tot_loss


# covert labels to guidance channels
def compute_guidance_maps(labels, out_dir=None):
    if labels is None:
        return []
    mat_label, target_labels, mappings = labels
    labels = np.unique(mat_label)
    print(f'Found {len(labels)} labels: {labels}')

    guidance_maps = []
    for mat_idx, (i, target_idx) in mappings.items():
        mat_idx, i, target_idx = int(mat_idx), int(i), int(target_idx)
        target_label = target_labels[i]
        mat_map = np.zeros(mat_label.shape, dtype=mat_label.dtype)
        mat_map[mat_label == mat_idx] = 1
        target_map = np.zeros(target_label.shape, dtype=target_label.dtype)
        target_map[target_label == target_idx] = 1
        guidance_maps.append((mat_map, i, target_map))

        if out_dir:
            tmp = np.concatenate((mat_map, target_map), axis=1)
            write_image(pth.join(out_dir, f'label{mat_idx}_{i}.png'), tmp)

    return guidance_maps