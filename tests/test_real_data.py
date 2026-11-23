import numpy as np

from emeans_id.real_data import normalize_rows, row_hashes


def test_row_hashes_find_duplicates():
    rng = np.random.default_rng(0)
    a = rng.standard_normal((5000, 100)).astype(np.float32)  # 400 bytes per row
    b = rng.standard_normal((300, 100)).astype(np.float32)
    b[:50] = a[100:150]  # 50 exact copies
    hit = np.isin(row_hashes(b), row_hashes(a))
    assert hit[:50].all() and not hit[50:].any()


def test_row_hashes_odd_row_length():
    # 3 floats = 12 bytes per row, needs padding to 16
    a = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 4]], dtype=np.float32)
    h = row_hashes(a)
    assert h[0] == h[1] and h[0] != h[2]


def test_normalize_rows_in_chunks():
    rng = np.random.default_rng(1)
    a = rng.standard_normal((2500, 20)).astype(np.float32)
    ref = a / np.linalg.norm(a, axis=1, keepdims=True)
    normalize_rows(a, chunk=1000)
    assert np.array_equal(a, ref)
