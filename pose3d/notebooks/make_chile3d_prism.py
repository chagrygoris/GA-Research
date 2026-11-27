"""Runner: the 3D 7-gonal PRISM ("7-gonal cylinder", requested 2026-11-27) with the recipe of the 3D 7-gonal pyramid runs.

Same model and recipe as make_chile3d_runner.py (ImageNet ResNet-50, 14x14 map, 368 sphere points, start s_h 100,
constant lr, 4,500 train / 1,000 test views uniform on SO(3), 180 epochs, 2xT4); only the shape changes. The prism
(probes/cosdir_interp/prism_pool.py) is rendered with the SYMSOL-CHILE renderer: regular 7-gon cross-section, circumradius
and half-height 1/sqrt 2, rotation group D7 (14 equivalent poses: 7 turns, each also with the end-swapping half-turn).
A model that knows the axis but not the turn has median error 90/7 = 12.9 deg, as for the pyramid; unlike the pyramid,
the 7 rectangular sides and the two 7-gon ends make the turn visible from most directions.
Code-snapshot patch: --symsol_set 15 = ("prism7",) read from symsol_chile_prism/.

usage: python make_chile3d_prism.py SEED EPOCHS WATCHDOG_MIN OUT.ipynb   -> prints the run name
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_chile3d_runner import build

HERE = os.path.dirname(os.path.abspath(__file__))
PRISM_SRC = open(os.path.join(HERE, "..", "probes", "cosdir_interp", "prism_pool.py")).read()

PRISM_PATCH = r'''

def apply_prism(root):
    """--symsol_set 15: the 7-gonal prism, poses uniform on SO(3), from symsol_chile_prism/."""
    f = os.path.join(root, "pose3d", "datasets/symsol.py"); t = open(f).read()
    assert t.count("SHAPE_SETS = {\n") == 1 and t.count("SET_DIRS = {") == 1, "symsol.py layout changed"
    t = t.replace("SHAPE_SETS = {\n", "SHAPE_SETS = {\n    15: (\"prism7\",),   # 7-gonal prism, 3D poses, D7\n")
    t = t.replace("SET_DIRS = {", "SET_DIRS = {15: \"symsol_chile_prism\", ")
    open(f, "w").write(t)

apply_prism('/kaggle/working/GA-Research/pose3d')
print('patched: --symsol_set 15 (3D 7-gonal prism)', flush=True)
'''

def build_prism(seed=0, epochs=180, watchdog=690):
    nb, name = build(seed, epochs, watchdog); cells = nb["cells"]
    def sub(old, new):
        hits = [c for c in cells if old in "".join(c["source"])]; assert len(hits) == 1, (old, len(hits))
        t = "".join(hits[0]["source"]); assert t.count(old) == 1, old; hits[0]["source"] = t.replace(old, new)
    sub("'--symsol_set', '14'", "'--symsol_set', '15'")
    sub("if not os.path.exists(f'{DATA}/symsol_chile/train/rotations.npz'):\n",
        "if not os.path.exists(f'{DATA}/symsol_chile_prism/train/rotations.npz'):\n"
        "    open('/kaggle/working/prism_pool.py', 'w').write(PRISM_SRC)\n"
        "    _gen = '/kaggle/working/GA-Research/pose3d/pose3d/datasets'\n"
        "    subprocess.run([sys.executable, '-u', '/kaggle/working/prism_pool.py', _gen, DATA, 'symsol_chile_prism', 'test', '1000', '1007'], check=True)\n"
        "    subprocess.run([sys.executable, '-u', '/kaggle/working/prism_pool.py', _gen, DATA, 'symsol_chile_prism', 'train', '4500', '7'], check=True)\n"
        "if False:\n")
    t = "".join(cells[2]["source"]); k = "apply_chile3d('/kaggle/working/GA-Research/pose3d')"
    assert t.count(k) == 1; cells[2]["source"] = t.replace(k, k + PRISM_PATCH)
    new = name.replace("chile3d_pyr7", "chile3d_prism7")
    sub(f"RUN_NAME = '{name}'", f"RUN_NAME = '{new}'\nPRISM_SRC = {PRISM_SRC!r}")
    cells[0]["source"] = (f"# 3D 7-GONAL PRISM (2026-11-27, human-requested, 2xT4, 1 seed): recipe of the 3D 7-gonal pyramid runs, only the "
                          f"shape changed; {epochs} epochs; ETA ~10 h (191 s/epoch measured for this recipe on 2xT4).\n") + "".join(cells[0]["source"])
    return nb, new

if __name__ == "__main__":
    a = sys.argv[1:]
    nb, name = build_prism(int(a[0]), int(a[1]), int(a[2]))
    json.dump(nb, open(a[3], "w")); print(name)
