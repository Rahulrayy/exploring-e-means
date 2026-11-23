import math

import torch

from .assign import nearest


def cosine_alpha(t, M, alpha0, eps=0.1):
    # eq 5, goes from alpha0 down to eps * alpha0 over the M updates
    return alpha0 * (eps + (1 - eps) * 0.5 * (1 + math.cos(math.pi * t / M)))


@torch.no_grad()
def revive_dead(S, N, eligible, tau, delta, g):
    # clusters with a tiny count take over half of a big cluster (same idea as faiss)
    dead = ((N < tau * N.mean()) & eligible).nonzero().flatten()
    if len(dead) == 0:
        return 0
    w = N.clone()
    w[dead] = 0
    dead = dead[:int((w > 0).sum())]  # can't split more donors than we have
    if len(dead) == 0:
        return 0
    donors = torch.multinomial(w, len(dead), replacement=False, generator=g)

    c = S[donors] / N[donors, None]
    sign = torch.ones(S.shape[1], device=S.device, dtype=S.dtype)
    sign[1::2] = -1  # +1 -1 +1 -1 ...
    half = N[donors] / 2
    N[dead] = half
    S[dead] = c * (1 + delta * sign) * half[:, None]
    N[donors] = half
    S[donors] = c * (1 - delta * sign) * half[:, None]
    return len(dead)


@torch.no_grad()
def emeans(X, k, T, B, alpha0, tau=0.1, revive_every=5, delta=2**-7,
           finalize=True, grace=True, init=None, seed=0):
    dev = X.device
    n = X.shape[0]
    # revival gets its own generator, otherwise runs with different alpha0
    # would end up with different batch orders after the first revival
    g_data = torch.Generator(device=dev).manual_seed(seed)
    g_rev = torch.Generator(device=dev).manual_seed(seed + 1000003)

    steps = n // B
    M = T * steps

    if init is None:
        C = X[torch.randperm(n, generator=g_data, device=dev)[:k]].clone()
    else:
        C = init.to(device=dev, dtype=X.dtype).clone()
    # tiny starting count so the init has almost no weight (paper sec 3.1)
    N = torch.full((k,), 1e-10, device=dev, dtype=X.dtype)
    S = C * N[:, None]

    ever_hit = torch.zeros(k, dtype=torch.bool, device=dev)
    last = torch.full((n,), -1, dtype=torch.long, device=dev)
    perm_head = []
    n_revived = 0
    t = 0

    for ep in range(T):
        perm = torch.randperm(n, generator=g_data, device=dev)
        perm_head.append(perm[:8].tolist())
        for b in range(steps):
            idx = perm[b * B:(b + 1) * B]
            xb = X[idx]
            C = S / N[:, None]
            a = nearest(xb, C)

            nj = torch.bincount(a, minlength=k).to(X.dtype)
            sj = torch.zeros_like(S).index_add_(0, a, xb)
            ever_hit |= nj > 0

            alpha = cosine_alpha(t, M, alpha0)
            N = (1 - alpha) * N + alpha * nj
            S = (1 - alpha) * S + alpha * sj
            # a cluster that gets nothing for a long time (or any empty cluster when
            # alpha = 1) ends up with N = 0 and S/N = nan, so keep it where it was
            tiny = N < 1e-20
            N = N.clamp(min=1e-20)
            S = torch.where(tiny[:, None], C * 1e-20, S)
            t += 1

            if revive_every and t % revive_every == 0:
                if grace and t < steps:
                    # first epoch: clusters that never got a point aren't dead yet,
                    # their points just haven't come up
                    eligible = ever_hit
                else:
                    eligible = torch.ones_like(ever_hit)
                n_revived += revive_dead(S, N, eligible, tau, delta, g_rev)

            if ep == T - 1:
                last[idx] = a

    ema = S / N[:, None]
    C = ema.clone()
    if finalize:
        # last epoch mean: each centroid becomes the mean of the points it got in the last epoch
        ok = last >= 0
        cnt = torch.bincount(last[ok], minlength=k).to(X.dtype)
        sums = torch.zeros_like(S).index_add_(0, last[ok], X[ok])
        has = cnt > 0
        C[has] = sums[has] / cnt[has, None]

    stats = {"revived": n_revived, "updates": t, "last": last, "ema": ema,
             "perm_head": perm_head}
    return C, stats
