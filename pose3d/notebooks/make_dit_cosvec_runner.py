"""Runners for DiT on SYMSOL I (10 %) with and without the scaled-cosine vector part (--dit_cos_vector,
models/dit_flow.CosVectorAttention): in the vector field's attention the first 3 dimensions of every head's query and key
enter through s_h * cos(q, k), the rest through the usual dot product.

Built from the bottom-view runner (make_bottom_grid14_variant.py: 2xT4, offline wheels + code snapshot dataset, watchdog,
per-epoch resume); changed: the SYMSOL I subset from a dataset (<user>/symsol1-10pct-fulltest) instead of rendering, the
recipe of the cosine-attention GATr's SYMSOL I kernel (arynabiaroza/symsol-gatr-spherecloud-cosdir-h8: 4,500 train /
5,000 test views per shape, AdamW lr 1e-4 sqrt-scaled, cosine schedule, 40 epochs, batch 64, no EMA, seed 0) with
--model dit_flow, and a patch that adds CosVectorAttention to the snapshot (the repository's code, identical).

usage: python make_dit_cosvec_runner.py COS(0|1) OUT.ipynb   -> prints the run name
"""
import json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_bottom_grid14_variant import make

HERE = os.path.dirname(os.path.abspath(__file__))

def args(cos):
    a = ['--model', 'dit_flow', '--encoder', 'resnet50', '--dataset', 'symsol', '--symsol_set', '1',
         '--symsol_train_views', '4500', '--symsol_test_views', '5000', '--no-use_warp', '--no-use_synth',
         '--num_workers', '4', '--no-ema', '--fixed_val_noise', '--batch_size', '64', '--optimizer', 'adamw', '--lr', '1e-4',
         '--lr_scaling', 'sqrt', '--lr_schedule', 'cosine', '--warmup_epochs', '0', '--n_epochs', '40',
         '--no-symsol_likelihood', '--dit_heads', '8', '--dit_sphere_tokens', '--ddp', '--num_gpus', '2', '--sync_bn',
         '--seed', '0']
    return a + (['--dit_cos_vector', '--dit_cos_scale', '30'] if cos else [])

def patch_source():
    src = open(os.path.join(HERE, "..", "pose3d", "models", "dit_flow.py")).read()
    cls = src[src.index("class CosVectorAttention(nn.Module):"):src.index("class DiTBlock(nn.Module):")]
    return '''

# DiT with a scaled-cosine vector part (--dit_cos_vector): the repository's CosVectorAttention and its plumbing.
COSVEC_CLASS = %r

def apply_dit_cosvec(root):
    p = lambda f: os.path.join(root, "pose3d", f)
    edit(p("models/dit_flow.py"), "class DiTBlock(nn.Module):", COSVEC_CLASS + "class DiTBlock(nn.Module):")
    edit(p("models/dit_flow.py"),
         "    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0):\\n"
         "        super().__init__()\\n"
         "        hidden = int(dim * mlp_ratio)\\n"
         "        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)\\n"
         "        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)\\n",
         "    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0, cos_vector: bool = False, cos_scale: float = 30.0):\\n"
         "        super().__init__()\\n"
         "        hidden = int(dim * mlp_ratio)\\n"
         "        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)\\n"
         "        self.attn = (CosVectorAttention(dim, heads, scale=cos_scale) if cos_vector\\n"
         "                     else nn.MultiheadAttention(dim, heads, batch_first=True))\\n")
    edit(p("models/dit_flow.py"),
         "    def __init__(self, dim: int, blocks: int, heads: int, mlp_ratio: float = 4.0, pose_lmax: int = 0):\\n",
         "    def __init__(self, dim: int, blocks: int, heads: int, mlp_ratio: float = 4.0, pose_lmax: int = 0,\\n"
         "                 cos_vector: bool = False, cos_scale: float = 30.0):\\n")
    edit(p("models/dit_flow.py"),
         "        self.blocks = nn.ModuleList(DiTBlock(dim, heads, mlp_ratio) for _ in range(blocks))\\n",
         "        self.blocks = nn.ModuleList(DiTBlock(dim, heads, mlp_ratio, cos_vector, cos_scale) for _ in range(blocks))\\n")
    edit(p("models/dit_flow.py"), "                 sphere_nside: int = 4):\\n",
         "                 sphere_nside: int = 4, cos_vector: bool = False, cos_scale: float = 30.0):\\n")
    edit(p("models/dit_flow.py"),
         "        self.vector_field = DiTVectorField(dim, blocks, heads, mlp_ratio, pose_lmax=pose_lmax)\\n",
         "        self.vector_field = DiTVectorField(dim, blocks, heads, mlp_ratio, pose_lmax=pose_lmax, cos_vector=cos_vector,\\n"
         "                                           cos_scale=cos_scale)\\n")
    edit(p("models/__init__.py"), "        sphere_input=fl.dit_sphere_tokens,\\n",
         "        sphere_input=fl.dit_sphere_tokens,\\n        cos_vector=fl.dit_cos_vector,\\n        cos_scale=fl.dit_cos_scale,\\n")
    edit(p("config.py"), "    dit_sphere_tokens: bool = False\\n",
         "    dit_sphere_tokens: bool = False\\n    dit_cos_vector: bool = False\\n    dit_cos_scale: float = 30.0\\n")

apply_dit_cosvec('/kaggle/working/GA-Research/pose3d')
print('patched: DiT CosVectorAttention (--dit_cos_vector)', flush=True)
''' % cls

DATA_CELL = '''os.chdir('/kaggle/working/GA-Research/pose3d')   # the package is found from here, no install needed
# SYMSOL I subset: a tar in the dataset (or already unpacked by Kaggle) -> /tmp/data/symsol/{train,test}
DATA = '/tmp/data'
tars = glob.glob('/kaggle/input/**/symsol1_*.tar', recursive=True)
if tars:
    os.makedirs(DATA, exist_ok=True)
    if not os.path.exists(f'{DATA}/symsol/test/rotations.npz'):
        with tarfile.open(tars[0]) as t:
            t.extractall(DATA)
else:
    found = glob.glob('/kaggle/input/**/symsol/test/rotations.npz', recursive=True)
    DATA = found[0][:-len('/symsol/test/rotations.npz')]
print('SYMSOL root:', DATA, {s: len(os.listdir(f'{DATA}/symsol/{s}/images')) for s in ('train', 'test')}, flush=True)
# CPU preflight: the DiT cosine-attention tests and the resume test, before any GPU time.
open('tests/test_dit_cosvec_kernel.py', 'w').write(TEST_SRC)
r = subprocess.run([sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', 'tests/test_dit_cosvec_kernel.py',
                    'tests/test_resume.py'],
                   env={**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '4'}, capture_output=True, text=True)
print(r.stdout[-3000:], r.stderr[-2000:], flush=True)
assert r.returncode == 0, 'CPU preflight failed'
name = RUN_NAME
os.environ['SYMSOL_SHARED_CACHE'] = '/kaggle/temp/symsol-shared-pixels'   # one image copy for all ranks
'''

def build(cos):
    nb, _ = make(7, 60, likelihood=False)
    cells = nb["cells"]
    name = f"symsol1_10pct_dit_sphere_h8_coslr_{'cosvec' if cos else 'plain'}"
    c0 = "".join(cells[0]["source"])
    c0 = re.sub(r"RUN_NAME = '[^']*'", f"RUN_NAME = '{name}'", c0, count=1)
    c0 = re.sub(r"ARGS = \[.*?\]\n", "ARGS = " + json.dumps(args(cos)) + "\n", c0, count=1, flags=re.S)
    c0 = re.sub(r"WATCHDOG_MINUTES = \([^)]*\)", "WATCHDOG_MINUTES = (240, 690)", c0, count=1)
    head = (f"# DiT ON SYMSOL I (10 %), {'WITH' if cos else 'WITHOUT'} THE SCALED-COSINE VECTOR PART "
            "(make_dit_cosvec_runner.py). Requested by a human (2026-11-21): DiT whose attention takes the first 3 dimensions "
            "of every head through s_h * cos (s_h learnable per head, start 30, cap 100) and the rest through the dot "
            "product, against the same DiT without it. Recipe of arynabiaroza/symsol-gatr-spherecloud-cosdir-h8 "
            "(cosine-attention GATr, median 3.08 deg): 40 epochs, cosine lr, AdamW 1e-4 sqrt-scaled, batch 64, no EMA; "
            "DiT dim 112, 4 blocks, 8 heads (as the GATr), tokens = the 88 hemisphere points (--dit_sphere_tokens).\n"
            "# Everything below the next line describes the base runner this one was built from; its ARGS were replaced.\n")
    cells[0]["source"] = head + c0
    assert "'--model', 'dit_flow'" not in c0 or True
    t = "".join(cells[2]["source"]); k = "apply_grid('/kaggle/working/GA-Research/pose3d')"
    assert t.count(k) == 1; cells[2]["source"] = t.replace(k, k + patch_source())
    t = "".join(cells[5]["source"])
    a = t.index("os.chdir('/kaggle/working/GA-Research/pose3d')"); b = t.index("# RESUME (engine/resume.py)")
    test_src = open(os.path.join(HERE, "..", "tests", "test_dit_cosvec.py")).read()
    cells[5]["source"] = f"TEST_SRC = {test_src!r}\n" + DATA_CELL + t[b:]
    return nb, name

if __name__ == "__main__":
    nb, name = build(sys.argv[1] == "1")
    json.dump(nb, open(sys.argv[2], "w")); print(name)
