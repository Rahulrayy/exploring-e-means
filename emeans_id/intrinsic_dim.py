import torch


@torch.no_grad()
def knn(Q, X, K, self_idx=None, budget=5e7):
    # sorted distances from every row of Q to its K nearest rows of X.
    # candidates come from fp32, then the distances are redone in float64
    # (the fp32 expansion is too rough for the very smallest distances)
    x_sq = (X * X).sum(1)
    step = max(1, int(budget // X.shape[0]))
    out = []
    for s in range(0, Q.shape[0], step):
        q = Q[s:s + step]
        d2 = x_sq[None, :] - 2 * (q @ X.T)
        if self_idx is not None:
            # don't count the query itself
            d2[torch.arange(len(q), device=X.device), self_idx[s:s + step]] = float("inf")
        cand = d2.topk(K + 5, dim=1, largest=False).indices  # a few spare in case fp32 got the order wrong
        diff = X[cand].double() - q[:, None, :].double()
        d = diff.pow(2).sum(-1).sqrt()
        d, _ = d.sort(1)
        out.append(d[:, :K])
    return torch.cat(out)


def sample_queries(X, n_query, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randperm(X.shape[0], generator=g)[:n_query].to(X.device)


def pca_dims(X, n_max=200_000):
    # participation ratio and number of components for 95% of the variance
    Xs = X[:n_max].double()
    Xs = Xs - Xs.mean(0)
    cov = Xs.T @ Xs / (Xs.shape[0] - 1)
    ev = torch.linalg.eigvalsh(cov).flip(0).clamp(min=0)
    pr = (ev.sum() ** 2 / (ev ** 2).sum()).item()
    frac = ev.cumsum(0) / ev.sum()
    n95 = int((frac < 0.95).sum()) + 1
    return pr, n95


def twonn(X, n_query=10_000, seed=0):
    # facco et al 2017: mu = r2 / r1 is pareto with exponent d -> mle
    qi = sample_queries(X, n_query, seed)
    d = knn(X[qi], X, 2, self_idx=qi)
    ok = d[:, 0] > 0  # exact duplicates break this, they should be removed before
    mu = d[ok, 1] / d[ok, 0]
    return ((ok.sum() - 1) / torch.log(mu).sum()).item()


def levina_bickel(X, K, n_query=5000, seed=0):
    # levina & bickel 2004 with K neighbours, averaged the mackay-ghahramani way
    # (average 1/m over points, then invert)
    qi = sample_queries(X, n_query, seed)
    d = knn(X[qi], X, K, self_idx=qi).clamp(min=1e-30)
    inv = torch.log(d[:, -1:] / d[:, :-1]).mean(1)
    return (1.0 / inv.mean()).item()


def all_estimates(X, k, seed=0):
    # k = number of clusters, sets the "cluster scale" K = n / k
    pr, n95 = pca_dims(X)
    K_cell = max(3, int(round(X.shape[0] / k)))
    return {
        "pr": pr,
        "pca95": n95,
        "twonn": twonn(X, seed=seed),
        "lb20": levina_bickel(X, 20, seed=seed),
        "lb_cell": levina_bickel(X, K_cell, seed=seed),
        "K_cell": K_cell,
    }
