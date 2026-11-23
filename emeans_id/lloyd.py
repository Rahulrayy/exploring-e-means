import torch

from .assign import nearest


@torch.no_grad()
def split_empty(C, counts, g, delta=2**-7):
    # faiss trick: an empty cluster takes over half of a big one
    empty = (counts == 0).nonzero().flatten().tolist()
    if not empty:
        return 0
    w = counts.to(C.dtype).clone()
    sign = torch.ones(C.shape[1], device=C.device, dtype=C.dtype)
    sign[1::2] = -1
    for ci in empty:
        cj = int(torch.multinomial(w, 1, generator=g))
        c = C[cj].clone()
        C[ci] = c * (1 + delta * sign)
        C[cj] = c * (1 - delta * sign)
        w[ci] = w[cj] / 2
        w[cj] = w[cj] / 2
    return len(empty)


@torch.no_grad()
def lloyd(X, k, iters=25, init=None, seed=0):
    dev = X.device
    n = X.shape[0]
    g = torch.Generator(device=dev).manual_seed(seed)
    # same seed gives the same starting centroids as emeans()
    if init is None:
        C = X[torch.randperm(n, generator=g, device=dev)[:k]].clone()
    else:
        C = init.to(device=dev, dtype=X.dtype).clone()

    n_split = 0
    for it in range(iters):
        a = nearest(X, C)
        counts = torch.bincount(a, minlength=k)
        sums = torch.zeros_like(C).index_add_(0, a, X)
        has = counts > 0
        C[has] = sums[has] / counts[has, None].to(X.dtype)
        n_split += split_empty(C, counts, g)
    return C, {"split": n_split}
