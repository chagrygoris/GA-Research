"""ModelNet10-SO(3) and SYMSOL data loading (image2sphere's real dataset classes, used as-is).

Both return the same {img, rot, cls} dict shape as Pascal3D, at a fixed 224x224, so no
resizing/collation changes are needed elsewhere. `cfg.run.path_to_datasets` must contain a
`modelnet10/` (modelnet10_{train,test}.npz) or `symsol/` ({train,test}/rotations.npz + images/)
subfolder depending on cfg.run.dataset.

ModelNet10Dataset already loads its whole .npz eagerly in __init__ (torch.from_numpy on the full
array), so it's RAM-resident the same way Pascal3D+'s Features.ram_memory cache is -- no wrapping
needed. SymsolDataset is not: it opens and decodes a PNG from disk on every __getitem__, every
epoch, so left as-is it is much slower per sample than Pascal3D+'s cached path. Wrapping it in the
same InMemoryDataset Pascal3D+ uses (dataset-agnostic: it only needs a base Dataset whose
__getitem__ returns {img, rot, cls} of the shapes read off sample 0, which SymsolDataset already
matches) puts it on equal footing.
"""

import time

from image2sphere.dataset import ModelNet10Dataset, SymsolDataset

from pose3d.datasets.cache import InMemoryDataset
from pose3d.datasets.pascal import _num_builder


def _cache(base, split, cfg):
    """Mirrors pascal.py's own ram_memory behavior: cache once at startup, same class, same
    build_workers rule, so a symsol run's per-epoch data path costs the same as Pascal3D+'s."""
    if not cfg.features.ram_memory:
        return base
    t0 = time.time()
    ds = InMemoryDataset(base, build_workers=_num_builder(cfg), include_cls=cfg.features.fisher_prior)
    print(f"[timing] {split}: built {len(ds)} samples from {cfg.run.dataset} in {time.time() - t0:.1f}s")
    return ds


def create_benchmark_datasets(cfg):
    path = cfg.run.path_to_datasets
    if cfg.run.dataset == "modelnet10":
        return ModelNet10Dataset(path, train=True), ModelNet10Dataset(path, train=False)
    train = SymsolDataset(path, train=True, set_number=cfg.data.symsol_set)
    val = SymsolDataset(path, train=False, set_number=cfg.data.symsol_set)
    return _cache(train, "train", cfg), _cache(val, "val", cfg)
