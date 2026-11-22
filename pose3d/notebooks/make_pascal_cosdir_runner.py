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

usage: python make_pascal_cosdir_runner.py EPOCHS WATCHDOG_MIN START_S_H OUT.ipynb [NOSYNTH(0|1)]   (EPOCHS 0 = timing run)
"""
import json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_bottom_grid14_variant import make

def args(epochs, synth=True):
    return ['--model', 'clifford_flow', '--encoder', 'resnet101', '--dataset', 'pascal', '--vector_field', 'gatr',
            '--cond_tokens', 'spherecloud', '--pose_tokens', 'frame_pin', '--condition_head', 'none',
            '--dircloud_scalars', '64', '--dircloud_nside', '8', '--dircloud_grid', '14',
            '--gatr_mv_channels', '8', '--gatr_s_channels', '32', '--no-gatr_zero_readout', '--gatr_heads', '8',
            *([] if synth else ['--no-use_synth']), '--num_workers', '4', '--batch_size', '64', '--optimizer', 'adamw', '--lr', '1e-4', '--lr_scaling', 'sqrt',
            '--n_epochs', str(max(epochs, 1)), '--ddp', '--num_gpus', '2', '--sync_bn', '--seed', '0']

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

def build(epochs, watchdog, s=60, synth=True):
    nb, _ = make(7, 60, likelihood=False)
    cells = nb["cells"]
    name = f"pascal_gatr_cosdir_grid14_nside8_s{s}_" + ("" if synth else "nosynth_") + ("timing" if epochs == 0 else f"e{epochs}")
    c0 = "".join(cells[0]["source"])
    c0 = re.sub(r"RUN_NAME = '[^']*'", f"RUN_NAME = '{name}'", c0, count=1)
    c0, k = re.subn(r"ARGS = \[.*?\]\n", "ARGS = " + json.dumps(args(epochs, synth)) + "\n", c0, count=1, flags=re.S); assert k == 1
    c0, k = re.subn(r"WATCHDOG_MINUTES = \([^)]*\)", f"WATCHDOG_MINUTES = ({watchdog}, {watchdog})", c0, count=1); assert k == 1
    head = ("# COSINE-ATTENTION GATr, 14x14 MAP, ON PASCAL3D+ (make_pascal_cosdir_runner.py). Requested by a human (2026-11-22). "
            + ("TIMING RUN: measures the time per step of the reference recipe (warp + synth, ResNet-101) with this head, "
               f"stopped by the watchdog after {watchdog} min; approved as a short run before planning the long one.\n"
               if epochs == 0 else f"{epochs} epochs.\n")
            + "# Everything below the next line describes the base runner this one was built from; its ARGS were replaced.\n")
    cells[0]["source"] = head + c0
    t = "".join(cells[2]["source"]); k = "apply_grid('/kaggle/working/GA-Research/pose3d')"
    assert t.count(k) == 1; cells[2]["source"] = t.replace(k, k + TIMING_PATCH)
    t = "".join(cells[5]["source"])
    a = t.index("DATA = '/kaggle/working/data'"); b = t.index("name = RUN_NAME")
    t = t[:a] + DATA_CELL + t[b:]
    t = t.replace("os.environ['SYMSOL_SHARED_CACHE'] = '/kaggle/temp/symsol-shared-pixels'   # one image copy for all ranks\n", "")
    t = t.replace("os.environ['POSE3D_SYMSOL_COS_SCALE'] = '60'", f"os.environ['POSE3D_SYMSOL_COS_SCALE'] = '{s}'")
    cells[5]["source"] = t
    return nb, name

if __name__ == "__main__":
    a = sys.argv[1:]
    nb, name = build(int(a[0]), int(a[1]), int(a[2]), synth=not (len(a) > 4 and a[4] == '1'))
    json.dump(nb, open(a[3], "w")); print(name)
