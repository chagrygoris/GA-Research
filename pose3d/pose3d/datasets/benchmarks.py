"""ModelNet10-SO(3) and SYMSOL data loading (image2sphere's real dataset classes, used as-is).

Both return the same {img, rot, cls} dict shape as Pascal3D, at a fixed 224x224, so no
resizing/collation changes are needed elsewhere. `cfg.run.path_to_datasets` must contain a
`modelnet10/` (modelnet10_{train,test}.npz) or `symsol/` ({train,test}/rotations.npz + images/)
subfolder depending on cfg.run.dataset.
"""

from image2sphere.dataset import ModelNet10Dataset, SymsolDataset


def create_benchmark_datasets(cfg):
    path = cfg.run.path_to_datasets
    if cfg.run.dataset == "modelnet10":
        return ModelNet10Dataset(path, train=True), ModelNet10Dataset(path, train=False)
    train = SymsolDataset(path, train=True, set_number=cfg.data.symsol_set)
    val = SymsolDataset(path, train=False, set_number=cfg.data.symsol_set)
    return train, val
