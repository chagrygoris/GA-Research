"""engine/distributed.sync_buffers: non-contiguous buffers (Image2Sphere's transposed output_wigners) are broadcast
through a contiguous copy instead of failing with "Tensors must be contiguous", and every rank ends with rank 0's
values. Two real gloo processes; gloo itself accepts non-contiguous tensors, so the broadcast here is wrapped to
refuse them the way NCCL does (ValueError "Tensors must be contiguous", seen on Kaggle 4xL4)."""

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _worker(rank, store_path, out_path):
    dist.init_process_group("gloo", store=dist.FileStore(store_path, 2), rank=rank, world_size=2)
    try:
        from pose3d.engine import distributed
        real = dist.broadcast

        def nccl_like(tensor, src):
            if not tensor.is_contiguous():
                raise ValueError("Tensors must be contiguous")
            return real(tensor, src=src)
        distributed.dist.broadcast = nccl_like
        sync_buffers = distributed.sync_buffers
        m = torch.nn.Module()
        m.register_buffer("w", (torch.arange(12.0) + 100 * rank).view(3, 4).t())   # transposed: non-contiguous
        m.register_buffer("b", torch.full((5,), float(rank)))
        assert not m.w.is_contiguous()
        sync_buffers(m)
        torch.save({"w": m.w.clone(), "b": m.b.clone()}, f"{out_path}.{rank}")
    finally:
        dist.destroy_process_group()


def test_non_contiguous_buffer_reaches_every_rank(tmp_path):
    mp.spawn(_worker, args=(str(tmp_path / "store"), str(tmp_path / "out")), nprocs=2, join=True)
    for rank in (0, 1):
        got = torch.load(f"{tmp_path / 'out'}.{rank}")
        torch.testing.assert_close(got["w"], torch.arange(12.0).view(3, 4).t())      # rank 0's values everywhere
        torch.testing.assert_close(got["b"], torch.zeros(5))
