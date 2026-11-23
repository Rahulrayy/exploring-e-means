import math

import torch


def latent(n, m, dist, g, nu=6):
    # every coordinate has mean 0 and variance 1
    if dist == "gaussian":
        return torch.randn(n, m, generator=g)
    if dist == "uniform":
        return (torch.rand(n, m, generator=g) * 2 - 1) * math.sqrt(3)
    if dist == "student":
        # multivariate t: gaussian / sqrt(chi2 / nu), one chi2 draw per point
        z = torch.randn(n, m, generator=g)
        chi2 = (torch.randn(n, nu, generator=g) ** 2).sum(1)
        z = z / torch.sqrt(chi2 / nu)[:, None]
        return z * math.sqrt((nu - 2) / nu)
    if dist == "laplace":
        # independent laplace coordinates, scale 1/sqrt(2) gives variance 1
        u = torch.rand(n, m, generator=g) - 0.5
        return -torch.sign(u) * torch.log1p(-2 * u.abs()) / math.sqrt(2)
    raise ValueError("unknown dist " + dist)


def swiss_roll(n, seed=0, n_val=50_000, n_test=50_000, device="cpu"):
    # 2-d sheet rolled up in 3-d (intrinsic dim 2, curved), total variance scaled to 1
    g = torch.Generator().manual_seed(seed)
    total = n + n_val + n_test
    t = 1.5 * math.pi * (1 + 2 * torch.rand(total, generator=g))
    h = 21 * torch.rand(total, generator=g)
    x = torch.stack([t * torch.cos(t), h, t * torch.sin(t)], 1)
    x = x - x.mean(0)
    x = x / x.var(0).sum().sqrt()
    x = x.to(device)
    return x[:n], x[n:n + n_val], x[n + n_val:]


def random_basis(D, m, seed=0):
    # D x m matrix with orthonormal columns
    g = torch.Generator().manual_seed(seed)
    Q, _ = torch.linalg.qr(torch.randn(D, m, generator=g, dtype=torch.float64))
    return Q


def make_data(n, m, dist="gaussian", D=None, sigma=0.0, seed=0,
              n_val=50_000, n_test=50_000, device="cpu"):
    # train / val / test drawn from the same distribution
    # signal scaled so the total variance is 1 for every m
    # D=None keeps the data in m dims. without noise that's the same problem as
    # putting it in a bigger space, just cheaper
    g = torch.Generator().manual_seed(seed)
    total = n + n_val + n_test
    x = latent(total, m, dist, g) / math.sqrt(m)
    if D is not None:
        Q = random_basis(D, m, seed=1000 * m + D).float()
        x = x @ Q.T
    if sigma > 0:
        x = x + sigma * torch.randn(x.shape, generator=g) / math.sqrt(x.shape[1])
    x = x.to(device)
    return x[:n], x[n:n + n_val], x[n + n_val:]


def zador_init(k, m, seed=0, device="cpu"):
    # start centroids from p^(m/(m+2)) instead of p (gaussian latent, D=m only)
    # for N(0, I) that is N(0, (m+2)/m I), then the same 1/sqrt(m) scaling as make_data
    g = torch.Generator().manual_seed(seed + 12345)
    c = torch.randn(k, m, generator=g) * math.sqrt((m + 2) / m) / math.sqrt(m)
    return c.to(device)
