"""Point-cloud utilities: mesh sampling, Depth-Anything depth maps, plotting, and the DummyNet debug model."""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import plotly.graph_objects as go
import torch
import torch.nn as nn
from clifford.algebra.cliffordalgebra import CliffordAlgebra
from clifford.models.modules.linear import MVLinear
from transformers import pipeline

from pose3d.models.ga_layers import TralaleroTralala


def show_torch_img(img : torch.Tensor):
    if len(img.shape) == 4:
        img = img.squeeze(0)
    img = img.cpu().permute(1, 2, 0).numpy()
    plt.imshow(img)


def draw_clouds(points, colors=None, size=1):
    if colors is not None:
        fig = go.Figure(data=[go.Scatter3d(
            x=points[:, 0], y=points[:, 1], z=points[:, 2],
            mode='markers',
            marker=dict(size=size, color=colors)
        )])
    else:
        fig = go.Figure(data=[go.Scatter3d(
            x=points[:, 0], y=points[:, 1], z=points[:, 2],
            mode='markers',
            marker=dict(size=size)
        )])
    
    fig.update_layout(
        scene=dict(xaxis_title='X (pixels)', yaxis_title='Y (pixels)', zaxis_title='Depth'),
        title='3D Point Cloud'
    )
    fig.show()


class PointCloudProcessor:
    def __init__(self, model_size: str = "base", device: Optional[str] = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        model_name = f"depth-anything/Depth-Anything-V2-{model_size.capitalize()}-hf"
        self.pipe = pipeline(task="depth-estimation", model=model_name, device=self.device)
        self.model = self.pipe.model
        self.image_processor = self.pipe.image_processor
        self.model.eval()


    def __call__(self, x : torch.tensor):
        '''
        Arguments:
            x : torch.Tensor of size (B, C, H, W)
        '''
        return self.model(x).predicted_depth.detach()

    def visualize_plotly(self, x : torch.Tensor, raw_img=True):
        '''Draw an interactive 3D point cloud of given object'''
        colors = x.cpu().numpy().reshape(-1, 3)
        if raw_img:
            x = self.model(x.unsqueeze(0)).squeeze(0)
        x = x.detach().cpu().numpy()
        v, u = np.indices((224, 224))
        points = np.stack((v, u, x), axis=-1).reshape(-1, 3)
        draw_clouds(points, colors=colors)

    def visualize_matplotlib(self, pcd: o3d.geometry.PointCloud, sample_size: int = 10000):
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        
        if len(points) > sample_size:
            indices = np.random.choice(len(points), sample_size, replace=False)
            points = points[indices]
            colors = colors[indices]
        
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=1)
        
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        ax.set_zlabel('Depth')
        ax.set_title('3D Point Cloud')


class DummyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.algebra = CliffordAlgebra((1, 1, 1))
        self.tralalero = TralaleroTralala(self.algebra, in_features=1, hidden_dim=[8, 16, 32], out_features=64)
        self.head = MVLinear(self.algebra, in_features=64, out_features=9)

    def forward(self, x : torch.tensor):
        '''
        Args:
            x : torch.tensor, point cloud of size (B, N, 3)
        Returns:
            x : rotation matrices of size (B, 3, 3)
        '''
        batch_size = x.shape[0]
        x = self.algebra.embed_grade(x, 1)
        x = x.reshape(-1, 1, 8)
        x = self.tralalero(x)
        x = x.reshape(batch_size, -1, 64, 8)
        x = x.max(dim=1).values
        x = self.head(x)
        x = self.algebra.get_grade(x, 0)
        # x = x.flatten(1, -1)
        # x = self.mlp(x)
        return x.reshape(-1, 3, 3)


class PoseNet(nn.Module):
    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, hidden_dim),
            nn.ReLU(inplace=True),
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 9),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: point cloud of shape (B, N, 3)
        Returns:
            rotation matrices of shape (B, 3, 3)
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)
        if x.dim() != 3 or x.size(-1) != 3:
            raise ValueError(f"Expected input of shape (B, N, 3), got {tuple(x.shape)}")

        feat = self.encoder(x)
        feat = feat.max(dim=1).values
        R = self.head(feat).view(-1, 3, 3)
        return R


class MeshProcessor:
    """
    A utility class to visualize .off files in headless environments 
    and convert 3D meshes to point cloud arrays.
    """

    @staticmethod
    def visualize_in_notebook(file_path, color='lightpink'):
        """Loads a mesh and renders it interactively using Plotly."""
        mesh = o3d.io.read_triangle_mesh(file_path)
        if not mesh.has_vertices():
            raise ValueError(f"Could not load mesh from {file_path}")

        verts = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)

        fig = go.Figure(data=[
            go.Mesh3d(
                x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                i=triangles[:, 0], j=triangles[:, 1], k=triangles[:, 2],
                color=color, opacity=0.6, flatshading=True
            )
        ])

        fig.update_layout(
            scene=dict(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False),
            margin=dict(l=0, r=0, b=0, t=0)
        )
        fig.show()

    @staticmethod
    def to_point_cloud_array(file_path, num_points=2048):
        """
        Loads a mesh and samples it into a point cloud.
        Returns: np.ndarray of shape (num_points, 3)
        """
        mesh = o3d.io.read_triangle_mesh(file_path)
        if not mesh.has_vertices():
            raise ValueError(f"Could not load mesh from {file_path}")
        pcd = mesh.sample_points_uniformly(number_of_points=num_points)
        return np.asarray(pcd.points)
