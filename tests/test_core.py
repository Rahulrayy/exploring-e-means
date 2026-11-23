import torch

from emeans_id import laws
from emeans_id.assign import nearest, exact_mse
from emeans_id.emeans import emeans
from emeans_id.lloyd import lloyd
from emeans_id.synthetic import make_data, random_basis

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def small_data(n, m, seed=0, dist="gaussian"):
    X, _, _ = make_data(n, m, dist=dist, seed=seed, n_val=10, n_test=10)
    return X


# S1
def test_nearest_vs_brute_force():
    g = torch.Generator().manual_seed(0)
    X = torch.randn(3000, 16, generator=g)
    C = torch.randn(200, 16, generator=g)
    a = nearest(X.to(DEV), C.to(DEV), chunk=512).cpu()
    d = ((X.double()[:, None, :] - C.double()[None, :, :]) ** 2).sum(-1)
    ref = d.argmin(1)
    # fp32 vs fp64 can flip a near tie now and then
    assert (a != ref).sum() <= 3


# S2
def test_init_comes_back_exactly():
    X = small_data(5000, 8, seed=1)
    C, _ = emeans(X, 64, T=0, B=256, alpha0=0.1, seed=3)
    C0 = X[torch.randperm(5000, generator=torch.Generator().manual_seed(3))[:64]]
    assert torch.allclose(C, C0, rtol=1e-6, atol=1e-7)


# S3
def test_last_epoch_mean_never_hurts():
    X = small_data(20000, 4, seed=2).double()
    for seed in range(5):
        C, st = emeans(X, 128, T=2, B=512, alpha0=0.2, seed=seed)
        last = st["last"]
        ok = last >= 0
        cost_ema = ((X[ok] - st["ema"][last[ok]]) ** 2).sum()
        cost_fin = ((X[ok] - C[last[ok]]) ** 2).sum()
        assert cost_fin <= cost_ema * (1 + 1e-12)


# S4 (exact version, float64 and no revival)
def test_rotation_invariance():
    Z = small_data(4000, 4, seed=5).double()
    Q = random_basis(32, 4, seed=0)
    X = Z @ Q.T
    Cz, _ = emeans(Z, 64, T=3, B=256, alpha0=0.3, revive_every=0, seed=0)
    Cx, _ = emeans(X, 64, T=3, B=256, alpha0=0.3, revive_every=0, seed=0)
    assert torch.allclose(Cx, Cz @ Q.T, atol=1e-9)


# S8: changing alpha0 must not change the init or the batch order
def test_same_batches_for_every_alpha():
    X = small_data(8000, 2, seed=7)
    heads = []
    for a0 in [0.02, 0.9]:
        C, st = emeans(X, 256, T=3, B=256, alpha0=a0, tau=0.5, seed=11)
        assert st["revived"] > 0  # otherwise the test doesn't test anything
        heads.append(st["perm_head"])
    assert heads[0] == heads[1]


def test_far_away_centroid_gets_revived():
    X = small_data(10000, 2, seed=3)
    init = X[:64].clone()
    init[0] = 100.0  # no point will ever pick this one
    C, st = emeans(X, 64, T=3, B=500, alpha0=0.1, init=init, seed=0)
    counts = torch.bincount(nearest(X, C), minlength=64)
    assert st["revived"] >= 1
    assert counts[0] > 0


def test_alpha_one_gives_no_nan():
    # with alpha0 = 1 the first update wipes the count of every cluster without points
    X = small_data(5000, 3, seed=4)
    C, _ = emeans(X, 256, T=2, B=256, alpha0=1.0, seed=0)
    assert torch.isfinite(C).all()


def test_emeans_vs_lloyd():
    # what the paper claims, on a small scale: same epochs -> e-means wins,
    # and a few more epochs beat lloyd@25
    X = small_data(20000, 8, seed=6)
    C3, _ = emeans(X, 256, T=3, B=1024, alpha0=0.2, seed=0)
    C10, _ = emeans(X, 256, T=10, B=1024, alpha0=0.2, seed=0)
    L3, _ = lloyd(X, 256, iters=3, seed=0)
    L25, _ = lloyd(X, 256, iters=25, seed=0)
    assert exact_mse(X, C3) < exact_mse(X, L3)
    assert exact_mse(X, C10) < exact_mse(X, L25)


def test_lloyd_goes_down():
    X = small_data(20000, 4, seed=8)
    C1, _ = lloyd(X, 256, iters=1, seed=0)
    C10, _ = lloyd(X, 256, iters=10, seed=0)
    assert exact_mse(X, C10) < exact_mse(X, C1)


def test_laws_match_hand_numbers():
    assert laws.our_batch(1024, 100_000, 2) == 1024
    assert abs(laws.alpha_law(1024, 1024, 100_000, 2) - 0.091) < 0.001
    assert laws.our_batch(4096, 1_000_000, 2) == 4096
    assert abs(laws.alpha_law(4096, 4096, 1_000_000, 2) - 0.059) < 0.001
    assert laws.knee_batch(1024, 1_000_000, 5) == 65536
    assert laws.alpha_law(1024, 65536, 1_000_000, 5) == 1.0  # capped
    assert len(laws.ladder(0.1)) == 15


def test_laplace_and_swiss_roll_have_unit_variance():
    from emeans_id.synthetic import swiss_roll
    X = small_data(200_000, 2, seed=9, dist="laplace")
    assert abs(X.var(0).sum().item() - 1) < 0.03
    R, _, _ = swiss_roll(100_000, n_val=10, n_test=10)
    assert R.shape[1] == 3 and abs(R.var(0).sum().item() - 1) < 0.01


def test_student_t_has_unit_variance():
    X = small_data(200_000, 4, seed=9, dist="student")
    # total variance should be ~1 (heavy tails so be a bit loose)
    assert abs(X.var(0).sum().item() - 1) < 0.05
