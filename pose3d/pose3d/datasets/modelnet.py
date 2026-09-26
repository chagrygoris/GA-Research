"""ModelNet10 point-cloud stub (a single mesh, random rotations); not a real benchmark pipeline yet."""

import pathlib

import pandas as pd
import torch
from torch.utils.data import Dataset

from pose3d.engine.metrics import project_to_orthogonal_manifold


class DummyPointCloudDataset(Dataset):
    def __init__(self, cfg=None, path: str = None, size: int = 42, num_points=2048):
        super().__init__()
        from pose3d.pointcloud import MeshProcessor

        path = cfg.run.path_to_datasets if cfg else path
        self.size = size
        self.base_path = pathlib.Path(path)
        self.meta = pd.read_csv(self.base_path / "metadata_modelnet10.csv")
        self.base_path /= "ModelNet10"
        self.point_cloud = MeshProcessor.to_point_cloud_array(file_path=self.base_path / "bed/train/bed_0001.off", num_points=num_points)
        self.num_points = num_points
        self.rotmats = project_to_orthogonal_manifold(torch.rand(self.size, 3, 3))

    def __len__(self):
        return self.size

    def __getitem__(self, i):
        return {
            "img" : torch.tensor(self.point_cloud, dtype=torch.float32) @ self.rotmats[i],
            "rot" : self.rotmats[i]
        }
