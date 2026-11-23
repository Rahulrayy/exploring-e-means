import torch

from emeans_id.intrinsic_dim import knn, levina_bickel, pca_dims, twonn
from emeans_id.synthetic import make_data

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def test_knn_vs_brute_force():
    g = torch.Generator().manual_seed(0)
    X = torch.randn(2000, 8, generator=g).to(DEV)
    qi = torch.arange(50, device=DEV)
    d = knn(X[qi], X, 5, self_idx=qi).cpu()
    full = torch.cdist(X[qi].double(), X.double()).cpu()
    full[torch.arange(50), torch.arange(50)] = float("inf")
    ref, _ = full.sort(1)
    assert torch.allclose(d, ref[:, :5], atol=1e-9)


def test_estimators_on_a_4d_subspace():
    X, _, _ = make_data(50_000, 4, D=64, seed=1, n_val=10, n_test=10, device=DEV)
    pr, n95 = pca_dims(X)
    assert abs(pr - 4) < 0.1
    assert n95 == 4
    assert abs(twonn(X, n_query=5000) - 4) < 0.5
    assert abs(levina_bickel(X, 20, n_query=3000) - 4) < 0.5
