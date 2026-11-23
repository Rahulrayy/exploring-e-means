import os

import h5py
import numpy as np
import torch

# datasets live outside the repo (they're big, and the repo folder is synced by onedrive)
DATA_DIR = os.environ.get("EMEANS_DATA", os.path.join(os.path.expanduser("~"), "emeans_data"))

FILES = {
    "sift": "sift-128-euclidean.hdf5",
    "fashion": "fashion-mnist-784-euclidean.hdf5",
    "glove": "glove-100-angular.hdf5",
    "deep": "deep-image-96-angular.hdf5",
}
# full sizes in bytes, so a half-downloaded file doesn't count as there
SIZES = {
    "sift": 525128288,
    "fashion": 227528288,
    "glove": 485413888,
    "deep": 3848008288,
}
ANGULAR = ("glove", "deep")


def row_hashes(a, chunk=100_000):
    # one 64-bit number per row made from its raw bytes, to spot exact duplicate rows.
    # way less memory than a python set of a million byte strings (8 bytes per row)
    nbytes = a.shape[1] * a.itemsize
    pad = (-nbytes) % 8
    mult = np.random.default_rng(12345).integers(1, 2**63, size=(nbytes + pad) // 8, dtype=np.uint64)
    mult |= np.uint64(1)
    out = np.empty(len(a), dtype=np.uint64)
    for s in range(0, len(a), chunk):
        raw = np.ascontiguousarray(a[s:s + chunk]).view(np.uint8).reshape(-1, nbytes)
        if pad:
            raw = np.pad(raw, ((0, 0), (0, pad)))
        words = np.ascontiguousarray(raw).view(np.uint64)
        out[s:s + chunk] = (words * mult).sum(axis=1)  # uint64 overflow just wraps, that's fine
    return out


def normalize_rows(a, chunk=100_000):
    # unit length, in place and in chunks (np.linalg.norm on the whole thing makes a full-size temp)
    for s in range(0, len(a), chunk):
        part = a[s:s + chunk]
        part /= np.maximum(np.linalg.norm(part, axis=1, keepdims=True), 1e-12)


def load_ann(name, n_train=None, n_val=10_000, max_rows=None, seed=0, device="cpu"):
    # ann-benchmarks files have a 'train' and a 'test' array.
    # their test queries are our test set, val is cut out of train.
    # written to keep only one full copy of the data in RAM (the old version peaked ~2 GB on sift)
    path = os.path.join(DATA_DIR, FILES[name])
    with h5py.File(path, "r") as f:
        if max_rows is None:
            train = f["train"][:]
        else:
            train = f["train"][:max_rows]  # deep has 10M rows, only read what we need
        test = f["test"][:]
    train = train.astype(np.float32, copy=False)  # no copy when it's float32 already
    test = test.astype(np.float32, copy=False)

    if name in ANGULAR:
        # unit length so euclidean distance ranks points like cosine does
        normalize_rows(train)
        normalize_rows(test)

    # same split as before: a random permutation, first n_val rows are val
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(train))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:] if n_train is None else perm[n_val:n_val + n_train]

    # throw out val/test rows that are exact copies of a train row
    train_h = row_hashes(train)[train_idx]
    val = train[val_idx]
    val = val[~np.isin(row_hashes(val), train_h)]
    test = test[~np.isin(row_hashes(test), train_h)]

    # pick the train rows on the target device, so RAM never holds two big copies
    X = torch.from_numpy(train).to(device)
    del train
    X = X[torch.from_numpy(train_idx).to(device)]
    return X, to_torch(val, device), to_torch(test, device)


def to_torch(a, device):
    return torch.from_numpy(np.ascontiguousarray(a)).to(device)


def load_pixels(n_train=1_000_000, n_val=10_000, n_test=10_000, seed=0, device="cpu", images=None):
    # rgb values from the colour photos that come with scikit-image, scaled to [0, 1].
    # no dedup against train here: nearly every colour appears many times, that's what pixels are like
    import skimage.data as d
    if images is None:
        images = ["astronaut", "chelsea", "coffee", "rocket", "hubble_deep_field", "immunohistochemistry"]
    ims = [getattr(d, name)() for name in images]
    px = np.concatenate([im.reshape(-1, 3) for im in ims]).astype(np.float32) / 255
    px = px[np.random.default_rng(seed).permutation(len(px))]
    val = px[:n_val]
    test = px[n_val:n_val + n_test]
    train = px[n_val + n_test:n_val + n_test + n_train]
    return to_torch(train, device), to_torch(val, device), to_torch(test, device)


def load_real(name, device="cpu"):
    # the settings used in block F
    if name == "pixels":
        return load_pixels(device=device)
    if name == "deep":
        return load_ann("deep", n_train=1_000_000, max_rows=1_010_000, device=device)
    return load_ann(name, n_train=1_000_000, device=device)


def have(name):
    if name == "pixels":
        return True
    path = os.path.join(DATA_DIR, FILES[name])
    return os.path.exists(path) and os.path.getsize(path) == SIZES[name]
