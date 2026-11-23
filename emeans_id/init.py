import math

import torch


@torch.no_grad()
def kmeanspp(X, k, seed=0):
    # greedy k-means++ like scikit-learn: at every step draw a few candidates with
    # probability ~ squared distance to the closest centre so far, keep the one that
    # lowers the total cost the most
    dev = X.device
    n = X.shape[0]
    g = torch.Generator(device=dev).manual_seed(seed)
    trials = 2 + int(math.log(k))

    first = torch.randint(n, (1,), generator=g, device=dev)
    centers = [X[first[0]]]
    closest = ((X - X[first[0]]) ** 2).sum(1)
    for _ in range(1, k):
        cand = torch.multinomial(closest, trials, replacement=True, generator=g)
        d = torch.cdist(X, X[cand]) ** 2           # n x trials
        new = torch.minimum(closest[:, None], d)   # cost if we picked each candidate
        best = new.sum(0).argmin()
        closest = new[:, best]
        centers.append(X[cand[best]])
    return torch.stack(centers)
