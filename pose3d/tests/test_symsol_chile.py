"""SYMSOL-CHILE (datasets/symsol_chile.py): pyramid geometry, labels that are true symmetries of the renders, the
camera frame, and loading through datasets/symsol.Symsol (set 5)."""

import math

import numpy as np
import torch

from pose3d.datasets.symsol import Symsol
from pose3d.datasets.symsol_chile import CAM_DIST, HALF_FOV, SIZE, generate, pyramid_faces, pyramid_group, render


def _turn_z(angle):
    c, s = math.cos(angle), math.sin(angle)
    return torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64)


def test_faces_bound_the_pyramid():
    for n in (3, 7, 1000):
        f = pyramid_faces(n)
        assert f.shape == (n + 1, 4) and np.allclose(np.linalg.norm(f[:, :3], axis=1), 1)
        s = 1 / math.sqrt(2)
        ang = 2 * np.pi * np.arange(n) / n
        verts = np.concatenate([[[0, 0, s]], np.stack([np.cos(ang), np.sin(ang), -np.ones(n)], -1) * s])
        assert (verts @ f[:, :3].T <= f[:, 3] + 1e-9).all()                       # every vertex inside every face
        assert (f[:, 3] > 0).all()                                                 # the centre strictly inside
        turned = f[:n, :3] @ pyramid_group(n)[1].T                                 # one step maps the sides onto
        assert np.abs(turned[:, None] - f[None, :n, :3]).sum(-1).min(1).max() < 1e-9   # the sides


def test_equivalent_poses_render_the_same_image():
    torch.manual_seed(0)
    from e3nn import o3
    for n in (7, 10):
        R = o3.rand_matrix(4, dtype=torch.float64)
        G = torch.from_numpy(pyramid_group(n))
        same = render(pyramid_faces(n), torch.cat([R, R @ G[3], R @ G[n - 1]]))
        a, b, c = same.view(3, 4, SIZE, SIZE).int()
        assert (a - b).abs().max() <= 2 and (a - c).abs().max() <= 2
        half = render(pyramid_faces(n), R @ _turn_z(math.pi / n)).int()          # half a step: a different image
        assert ((a - half).abs() > 20).float().mean() > 0.005


def test_camera_frame_columns_right_rows_down():
    """Apex along +e1 lands on the right, along +e2 at the bottom (datasets/symsol.py's camera frame)."""
    to_e1 = torch.tensor([[0.0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=torch.float64)   # R e3 = e1
    to_e2 = torch.tensor([[1.0, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=torch.float64)   # R e3 = e2
    img = render(pyramid_faces(10), torch.stack([to_e1, to_e2])) > 0
    apex = (1 / math.sqrt(2)) / CAM_DIST / math.tan(HALF_FOV)                         # in [-1, 1] image units
    pixel = (apex + 1) / 2 * SIZE
    right = torch.nonzero(img[0].any(0)).max().item()
    bottom = torch.nonzero(img[1].any(1)).max().item()
    assert abs(right - pixel) < 2 and abs(bottom - pixel) < 2, (right, bottom, pixel)


def test_symsol_loader_reads_the_chile_set(tmp_path):
    generate(str(tmp_path), "test", 2, log=lambda *_: None)
    generate(str(tmp_path), "train", 2, log=lambda *_: None)
    test = Symsol(str(tmp_path), train=False, set_number=5, workers=2)
    assert len(test) == 6 and test.shapes == ("pyr7", "pyr10", "pyr13")
    for i, n in zip((0, 2, 4), (7, 10, 13)):
        item = test[i]
        assert int(item["n_eq"]) == n and item["rots_eq"].shape == (720, 3, 3)    # SYMSOL I's padding
        assert torch.allclose(item["rots_eq"][0], item["rot"])
        assert item["img"].shape == (3, SIZE, SIZE) and item["img"].max() > 0.5
    train = Symsol(str(tmp_path), train=True, set_number=5, workers=2)
    rot = train[0]["rot"]                                                              # a random equivalent of pyr7
    eq = torch.from_numpy(np.load(tmp_path / "symsol_chile" / "train" / "rotations.npz")["pyr7"][0])
    assert torch.isclose(eq, rot, atol=1e-6).flatten(1).all(1).any()


def test_padding_grows_for_a_set_with_more_equivalents(tmp_path):
    generate(str(tmp_path), "test", 1, shapes=("pyr1000",), log=lambda *_: None)
    import pose3d.datasets.symsol as symsol
    symsol.SHAPE_SETS[99], symsol.SET_DIRS[99] = ("pyr1000",), "symsol_chile"
    try:
        item = Symsol(str(tmp_path), train=False, set_number=99, workers=1)[0]
        assert int(item["n_eq"]) == 1000 and item["rots_eq"].shape == (1000, 3, 3)
    finally:
        del symsol.SHAPE_SETS[99], symsol.SET_DIRS[99]


def test_adding_shapes_keeps_the_poses_of_the_first_release(tmp_path):
    """pyr7 / pyr10 / pyr13 must get the same poses alone (the first release's set) and next to pyr4..pyr6."""
    generate(str(tmp_path / "old"), "test", 3, shapes=("pyr7", "pyr10", "pyr13"), log=lambda *_: None)
    generate(str(tmp_path / "new"), "test", 3, log=lambda *_: None)
    old = np.load(tmp_path / "old" / "symsol_chile" / "test" / "rotations.npz")
    new = np.load(tmp_path / "new" / "symsol_chile" / "test" / "rotations.npz")
    assert sorted(new.files) == sorted(("pyr4", "pyr5", "pyr6", "pyr7", "pyr10", "pyr13"))
    for s in ("pyr7", "pyr10", "pyr13"):
        np.testing.assert_array_equal(old[s], new[s])
    assert not np.allclose(new["pyr4"][:, 0], new["pyr7"][:, 0])                  # the new shapes have their own poses


def test_set_6_reads_all_six_pyramids(tmp_path):
    generate(str(tmp_path), "test", 2, log=lambda *_: None)
    test = Symsol(str(tmp_path), train=False, set_number=6, workers=2)
    assert test.shapes == ("pyr4", "pyr5", "pyr6", "pyr7", "pyr10", "pyr13") and len(test) == 12
    assert [int(test[2 * c]["n_eq"]) for c in range(6)] == [4, 5, 6, 7, 10, 13]
