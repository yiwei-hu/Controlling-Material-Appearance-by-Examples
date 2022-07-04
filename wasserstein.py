import torch
import torch.nn as nn


class WassersteinLoss(nn.Module):
    def __init__(self, nb_dirs=0, metric='l2'):
        super(WassersteinLoss, self).__init__()
        assert metric in ['l1', 'l2']

        self.nb_dirs = nb_dirs
        self.criterion = nn.MSELoss() if metric == 'l2' else nn.L1Loss()

    @staticmethod
    def random_dir(nb_dim, nb_dirs, device):
        directions = torch.normal(mean=0.0, std=1.0, size=(nb_dirs, nb_dim), device=device)
        norm = torch.sqrt(torch.sum(torch.square(directions), dim=-1, keepdim=True))
        return directions / norm

    @staticmethod
    def slicing(features, directions):
        # project each pixel feature onto directions
        b, c, h, w = features.shape
        features = features.view(b, c, h*w)
        proj = torch.matmul(directions, features) # (b, n_dir, h*w)
        return proj

    def forward(self, x, y, index_maps=None):
        assert x.shape[0] == 1 and x.shape[1:] == y.shape[1:]
        n_features, h, w = x.shape[1:]
        nb_dirs = self.nb_dirs if self.nb_dirs != 0 else n_features
        directions = self.random_dir(n_features, nb_dirs, x.device).unsqueeze(0)

        proj_x, proj_y = self.slicing(x, directions), self.slicing(y, directions)
        proj_x = proj_x.view(nb_dirs, h * w)

        if index_maps is None:
            proj_y = proj_y.view(nb_dirs, h * w)
            return self.wasserstein_distance(proj_x, proj_y)
        else:
            dists = []
            for x_idx, i, y_idx in index_maps:
                proj_x_ = proj_x[:, x_idx]
                proj_y_ = proj_y[i]
                proj_y_ = proj_y_[:, y_idx]
                dists.append(self.wasserstein_distance(proj_x_, proj_y_))

            return sum(dists) / len(dists)

    def wasserstein_distance(self, u_values, v_values):
        assert u_values.shape == v_values.shape and u_values.ndim == 2 and v_values.ndim == 2
        u_sorted, _ = torch.sort(u_values, dim=-1)
        v_sorted, _ = torch.sort(v_values, dim=-1)

        return self.criterion(u_sorted, v_sorted)


class CramerLoss(nn.Module):
    def __init__(self, nb_dirs=0, p=1):
        super(CramerLoss, self).__init__()
        assert p == 1 or p == 2

        self.nb_dirs = nb_dirs
        self.p = p

    @staticmethod
    def random_dir(nb_dim, nb_dirs, device):
        directions = torch.normal(mean=0.0, std=1.0, size=(nb_dirs, nb_dim), device=device)
        norm = torch.sqrt(torch.sum(torch.square(directions), dim=-1, keepdim=True))
        return directions / norm

    @staticmethod
    def slicing(features, directions):
        # project each pixel feature onto directions
        b, c, h, w = features.shape
        features = features.view(b, c, h*w)
        proj = torch.matmul(directions, features)  # (b, n_dir, h*w)
        return proj

    def forward(self, x, y, index_maps=None):
        assert x.shape[0] == 1 and x.shape[1:] == y.shape[1:]
        n_features, h, w = x.shape[1:]
        nb_dirs = self.nb_dirs if self.nb_dirs != 0 else n_features
        directions = self.random_dir(n_features, nb_dirs, x.device).unsqueeze(0)

        proj_x, proj_y = self.slicing(x, directions), self.slicing(y, directions)
        proj_x = proj_x.view(nb_dirs, h * w)

        if index_maps is None:
            proj_y = proj_y.view(-1, h*w)
            return torch.mean(self.cramer_distance(proj_x, proj_y, self.p))
        else:
            dists = []
            for x_idx, i, y_idx in index_maps:
                proj_x_ = proj_x[:, x_idx]
                proj_y_ = proj_y[i]
                proj_y_ = proj_y_[:, y_idx]
                dists.append(torch.mean(self.cramer_distance(proj_x_, proj_y_, self.p)))

            return sum(dists) / len(dists)

    @staticmethod
    def cramer_distance(u_values, v_values, p):
        assert p == 1 or p == 2
        assert u_values.shape[0] == v_values.shape[0] and u_values.ndim == 2 and v_values.ndim == 2
        u_sorted, _ = torch.sort(u_values, dim=1)
        v_sorted, _ = torch.sort(v_values, dim=1)

        all_values = torch.cat((u_values, v_values), dim=1)
        all_values, _ = all_values.sort(dim=1)

        # Compute the differences between pairs of successive values of u and v.
        deltas = torch.diff(all_values, dim=1)

        # Get the respective positions of the values of u and v among the values of both distributions.
        u_cdf_indices = torch.searchsorted(u_sorted, all_values[:, :-1], right=True)
        v_cdf_indices = torch.searchsorted(v_sorted, all_values[:, :-1], right=True)

        u_cdf = u_cdf_indices / u_values.shape[1]
        v_cdf = v_cdf_indices / v_values.shape[1]

        if p == 1:
            return torch.sum(torch.multiply(torch.abs(u_cdf - v_cdf), deltas), dim=1)
        elif p == 2:
            return torch.sum(torch.multiply(torch.square(u_cdf - v_cdf), deltas), dim=1)
        else:
            raise NotImplementedError(f'Unsupported norm: {p}')