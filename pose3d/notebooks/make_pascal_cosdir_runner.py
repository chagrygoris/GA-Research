"""Runner: the cosine-attention GATr with the larger feature map on Pascal3D+.

Model: the bottom-view 7-gon recipe's model (make_bottom_grid14_variant.py: plain spherecloud GATr, 8 heads, GATr's
multivector dot product replaced by s_h * cos(angle) between query and key directions, s_h learnable per head, image
tokens lifted from a 14x14 ResNet map onto 368 hemisphere points, --dircloud_nside 8). Data and training: the Pascal3D+
reference recipe of config.py (ResNet-101, warp augmentation + 3 synthetic renders per real image from the mounted
syfry5suvzovvakmuj/pascal3d-synth-pack, EMA, batch 64, AdamW 1e-4 sqrt-scaled, 5 warmup epochs). No Pascal3D+ run of
this head exists yet.

TIMING mode (EPOCHS = 0): a patch prints "TIMING step=<k> t=<seconds since the first step>" every 50 optimizer steps on
rank 0, and the watchdog stops the kernel after WATCH minutes, so the time per step (and per epoch: about 1,149 steps
of 64) is measured before a long run is planned.

NOTE (2026-11-22): the synth pack is read into RAM by every DDP rank (datasets/packed.py _read_bin, about 18 GB each),
which does not fit the ~29 GB of a 2xT4 machine: the first timing kernel stalled before step 0. NOSYNTH=1 drops the
synthetic renders (--no-use_synth: real images with warp augmentation, a small part of the pack).

usage: python make_pascal_cosdir_runner.py EPOCHS WATCHDOG_MIN START_S_H OUT.ipynb [NOSYNTH(0|1) [GPUS]]   (EPOCHS 0 = timing run)
"""
import json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_bottom_grid14_variant import make
from make_sh_trace_variant import GRID28   # its GATr block checkpointing (POSE3D_GATR_CHECKPOINT=1); --dircloud_grid 14 is unchanged by it

def args(epochs, synth=True):
    return ['--model', 'clifford_flow', '--encoder', 'resnet101', '--dataset', 'pascal', '--vector_field', 'gatr',
            '--cond_tokens', 'spherecloud', '--pose_tokens', 'frame_pin', '--condition_head', 'none',
            '--dircloud_scalars', '64', '--dircloud_nside', '8', '--dircloud_grid', '14',
            '--gatr_mv_channels', '8', '--gatr_s_channels', '32', '--no-gatr_zero_readout', '--gatr_heads', '8',
            *([] if synth else ['--no-use_synth']), '--num_workers', '4', '--batch_size', '64', '--optimizer', 'adamw', '--lr', '1e-4', '--lr_scaling', 'sqrt',
            '--n_epochs', str(max(epochs, 1)), '--ddp', '--num_gpus', '2', '--sync_bn', '--seed', '0']

PATHS_PATCH = '''

def apply_pascal_paths(root):
    """Image2Sphere's get_pascal_paths calls os.path.exists for every image in the VOC lists (tens of thousands of
    calls); on Kaggle's lazily mounted dataset that took over 12 minutes in every DDP rank before training (the
    stack dumps of avgurpetrovich/pascal-cosdir-grid14-diag2-t4x2). Same result from one directory listing per
    category."""
    f = os.path.join(root, "pose3d", "datasets/pascal.py")
    edit(f, "from image2sphere.pascal_dataset import Pascal3D\\n",
         "from image2sphere.pascal_dataset import Pascal3D\\n"
         "import image2sphere.pascal_dataset as _i2s_pd\\n\\n\\n"
         "def _fast_pascal_paths(folder, cate, mode):\\n"
         "    import os as _os\\n"
         "    d = _os.path.join(folder, f'Images/{cate}_pascal')\\n"
         "    have = set(_os.listdir(d)) if _os.path.isdir(d) else set()\\n"
         "    img_paths, annot_paths = [], []\\n"
         "    with open(_os.path.join(folder, f'PASCAL/VOCdevkit/VOC2012/ImageSets/Main/{cate}_{mode}.txt')) as fh:\\n"
         "        for l in fh.readlines():\\n"
         "            num = l.split(' ')[0]\\n"
         "            if f'{num}.jpg' in have:\\n"
         "                img_paths.append(_os.path.join(folder, f'Images/{cate}_pascal/{num}.jpg'))\\n"
         "                annot_paths.append(_os.path.join(folder, f'Annotations/{cate}_pascal/{num}.mat'))\\n"
         "    return img_paths, annot_paths\\n\\n\\n"
         "_i2s_pd.get_pascal_paths = _fast_pascal_paths\\n")

apply_pascal_paths('/kaggle/working/GA-Research/pose3d')
print('patched: fast Pascal3D+ path listing', flush=True)
'''

TIMING_PATCH = '''

def apply_timing(root):
    """Print the wall time every 50 optimizer steps on rank 0 (TIMING lines), to measure the time per step."""
    f = os.path.join(root, "pose3d", "engine/trainer.py")
    anchor = "        if ema is not None:\\n            ema.update_parameters(model)\\n"
    edit(f, anchor, anchor +
         "        if is_main() and batch_i % 50 == 0:\\n"
         "            import time as _t\\n"
         "            if batch_i == 0: globals()['_T0'] = _t.time()\\n"
         "            print(f'TIMING step={batch_i} t={_t.time() - globals()[\\"_T0\\"]:.1f}', flush=True)\\n")

    # Diagnosis (2026-11-22): two timing kernels hung between data loading and the first step. Every rank prints its
    # Python stack every 4 minutes (faulthandler), which shows where it waits (DataLoader, NCCL, forward pass).
    edit(f, "    n_epochs = cfg.train.n_epochs\\n",
         "    n_epochs = cfg.train.n_epochs\\n"
         "    import faulthandler as _fh, sys as _sys\\n"
         "    _fh.dump_traceback_later(240, repeat=True, file=_sys.stdout)\\n")
    edit(f, "    for batch_i, data in enumerate(tqdm(loader, disable=not is_main() or cfg.run.platform == \\\"kaggle\\\")):\\n        optimizer.zero_grad(set_to_none=True)\\n",
         "    for batch_i, data in enumerate(tqdm(loader, disable=not is_main() or cfg.run.platform == \\\"kaggle\\\")):\\n"
         "        if batch_i < 3: print(f'BATCH {batch_i} loaded', flush=True)\\n"
         "        optimizer.zero_grad(set_to_none=True)\\n")
    # The second diagnostic kernel printed no stack at all: the hang is before train() (after the val data loaded).
    # Arm the stack dumps at the start of main() instead, every 3 minutes, in every process.
    edit(os.path.join(root, "pose3d", "train.py"), "    cfg = parse_args(argv)\\n",
         "    cfg = parse_args(argv)\\n"
         "    import faulthandler as _fh0, sys as _sys0\\n"
         "    _fh0.dump_traceback_later(180, repeat=True, file=_sys0.stdout)\\n")
    edit(f, "    _fh.dump_traceback_later(240, repeat=True, file=_sys.stdout)\\n", "")

apply_timing('/kaggle/working/GA-Research/pose3d')
print('patched: TIMING lines every 50 steps, stack dumps every 4 min', flush=True)
'''

DATA_CELL = '''# Pascal3D+: the annotations / images dataset is --path_to_datasets; the synth pack and the RAM cache are found by
# config.py from their mount points (SYNTH_PACK_DIRS, PRE_CACHE_DIRS).
DATA = next(p for p in ['/kaggle/input/datasets/syfry5suvzovvakmuj/pascal3d', '/kaggle/input/pascal3d'] if os.path.exists(p))
print('Pascal3D+ root:', DATA, [p for p in glob.glob('/kaggle/input/**/pascal3d*', recursive=False)] +
      glob.glob('/kaggle/input/datasets/*/pascal3d*'), flush=True)
r = subprocess.run([sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', 'tests/test_resume.py'],
                   env={**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '4'}, capture_output=True, text=True)
print(r.stdout[-2000:], r.stderr[-1000:], flush=True)
assert r.returncode == 0, 'CPU preflight failed'
'''

def build(epochs, watchdog, s=60, synth=True, gpus=2):
    nb, _ = make(7, 60, likelihood=False)
    cells = nb["cells"]
    name = f"pascal_gatr_cosdir_grid14_nside8_s{s}_" + ("" if synth else "nosynth_") + ("timing" if epochs == 0 else f"e{epochs}")
    c0 = "".join(cells[0]["source"])
    c0 = re.sub(r"RUN_NAME = '[^']*'", f"RUN_NAME = '{name}'", c0, count=1)
    c0, k = re.subn(r"ARGS = \[.*?\]\n", "ARGS = " + json.dumps(args(epochs, synth)).replace('"--num_gpus", "2"', f'"--num_gpus", "{gpus}"') + "\n", c0, count=1, flags=re.S); assert k == 1
    c0, k = re.subn(r"WATCHDOG_MINUTES = \([^)]*\)", f"WATCHDOG_MINUTES = ({watchdog}, {watchdog})", c0, count=1); assert k == 1
    head = ("# COSINE-ATTENTION GATr, 14x14 MAP, ON PASCAL3D+ (make_pascal_cosdir_runner.py). Requested by a human (2026-11-22). "
            + ("TIMING RUN: measures the time per step of the reference recipe (warp + synth, ResNet-101) with this head, "
               f"stopped by the watchdog after {watchdog} min; approved as a short run before planning the long one.\n"
               if epochs == 0 else f"{epochs} epochs.\n")
            + "# Everything below the next line describes the base runner this one was built from; its ARGS were replaced.\n")
    cells[0]["source"] = head + c0
    t = "".join(cells[2]["source"]); k = "apply_grid('/kaggle/working/GA-Research/pose3d')"
    assert t.count(k) == 1; cells[2]["source"] = t.replace(k, k + GRID28 + PATHS_PATCH + (TIMING_PATCH if epochs == 0 else ""))   # no stack dumps in long runs
    t = "".join(cells[5]["source"])
    a = t.index("DATA = '/kaggle/working/data'"); b = t.index("name = RUN_NAME")
    t = t[:a] + DATA_CELL + t[b:]
    t = t.replace("os.environ['SYMSOL_SHARED_CACHE'] = '/kaggle/temp/symsol-shared-pixels'   # one image copy for all ranks\n", "")
    t = t.replace("os.environ['POSE3D_SYMSOL_COS_SCALE'] = '60'", f"os.environ['POSE3D_SYMSOL_COS_SCALE'] = '{s}'")
    # ResNet-101 + 14x14 map + 368 tokens ran out of CUDA memory on a T4 in the first backward pass (timing3 kernel):
    # GATr block checkpointing, as for the 28x28 SYMSOL runs.
    t = t.replace("os.environ['POSE3D_SYMSOL_COS_SCALE'] = ", "os.environ['POSE3D_GATR_CHECKPOINT'] = '1'\nos.environ['POSE3D_SYMSOL_COS_SCALE'] = ", 1)
    cells[5]["source"] = t
    return nb, name

if __name__ == "__main__":
    a = sys.argv[1:]
    nb, name = build(int(a[0]), int(a[1]), int(a[2]), synth=not (len(a) > 4 and a[4] == '1'), gpus=int(a[5]) if len(a) > 5 else 2)
    json.dump(nb, open(a[3], "w")); print(name)
