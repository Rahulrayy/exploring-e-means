import torch


@torch.no_grad()
def nearest(X, C, chunk=8192):
    # index of the closest centroid for every row of X
    # |x|^2 is the same for all centroids so we skip it
    c_sq = (C * C).sum(1)
    idx = torch.empty(X.shape[0], dtype=torch.long, device=X.device)
    for s in range(0, X.shape[0], chunk):
        xb = X[s:s + chunk]
        idx[s:s + chunk] = (c_sq[None, :] - 2 * (xb @ C.T)).argmin(1)
    return idx


@torch.no_grad()
def exact_mse(X, C, chunk=8192):
    # assign in fp32 but recompute the distance in float64
    # (the expansion above loses precision when |x| is large, e.g. sift)
    a = nearest(X, C, chunk)
    total = 0.0
    for s in range(0, X.shape[0], chunk):
        d = X[s:s + chunk].double() - C[a[s:s + chunk]].double()
        total += (d * d).sum().item()
    return total / X.shape[0]
