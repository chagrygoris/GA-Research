"""Runner: the bottom-view recipe of the cosine-attention GATr applied to the ORIGINAL (3D) SYMSOL-CHILE 7-gonal pyramid.

"Original 3D" means poses drawn uniformly from all rotations SO(3) (datasets/symsol_chile.py --view random, written to
<root>/symsol_chile/), instead of the bottom view, where only the turn about the pyramid's axis varies. Equivalent poses
are still the 7 turns about the axis, so a model that does not resolve the turn has median error about 90/7 = 12.9 deg
(the "guessing level"; Image2Sphere measured 12.67 deg on this shape, reports/symmetry_spectrum.md on the branch
claude/symsol-symsol2-differences-2dukos).

Built from the s_h-trace runner (make_sh_trace_variant.build: the bottom-view 14x14-map runner plus a per-epoch s_h log
and model snapshots) with the recipe that learned the bottom-view 7-gon: 14x14 feature map, 368 sphere points
(--dircloud_nside 8), start s_h 100, learnable, constant learning rate, 4,500 train / 1,000 test views. Changed:
  * a patch of the code snapshot adds --symsol_set 14 = ("pyr7",) read from symsol_chile/ (the 3D views; set 7 is
    the bottom view of the same shape);
  * the data cell renders --view random instead of --view bottom.

usage: python make_chile3d_runner.py SEED EPOCHS WATCHDOG_MIN OUT.ipynb   -> prints the run name
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_sh_trace_variant import build as build_sh

PATCH3D = '''

def apply_chile3d(root):
    """--symsol_set 14: the 7-gonal pyramid with poses uniform on SO(3), from symsol_chile/ (3D SYMSOL-CHILE)."""
    f = os.path.join(root, "pose3d", "datasets/symsol.py"); t = open(f).read()
    assert t.count("SHAPE_SETS = {\\n") == 1 and t.count("SET_DIRS = {") == 1, "symsol.py layout changed"
    t = t.replace("SHAPE_SETS = {\\n", "SHAPE_SETS = {\\n    14: (\\"pyr7\\",),   # SYMSOL-CHILE 7-gon, 3D poses (set 7 = its bottom view)\\n")
    t = t.replace("SET_DIRS = {", "SET_DIRS = {14: \\"symsol_chile\\", ")
    open(f, "w").write(t)

apply_chile3d('/kaggle/working/GA-Research/pose3d')
print('patched: --symsol_set 14 (3D 7-gon)', flush=True)
'''

def build(seed, epochs, watchdog, s=100):
    tag = f"s{s}_e{epochs}" + (f"_seed{seed}" if seed else "")
    nb, _ = build_sh(s, True, 1, 1000, epochs, [epochs], watchdog, tag, n=7, seed=seed,
                     why="does the bottom-view recipe of the 7-gon work on the 3D SYMSOL-CHILE 7-gon?")
    cells = nb["cells"]
    name = f"chile3d_pyr7_gatr_cosdir_grid14_nside8_{tag}"
    def sub(old, new):
        hits = [c for c in cells if old in "".join(c["source"])]
        assert len(hits) == 1, (old, len(hits))
        t = "".join(hits[0]["source"]); assert t.count(old) == 1, old; hits[0]["source"] = t.replace(old, new)
    sub("RUN_NAME = 'bottom7_gatr_cosdir_grid14_nside8_" + tag + "'", f"RUN_NAME = '{name}'")
    sub("'--symsol_set', '7'", "'--symsol_set', '14'")
    sub("f'{DATA}/symsol_chile_bottom/train/rotations.npz'", "f'{DATA}/symsol_chile/train/rotations.npz'")
    sub("'--view', 'bottom', '--shapes', 'pyr7'", "'--view', 'random', '--shapes', 'pyr7'")
    t = "".join(cells[2]["source"]); k = "apply_grid('/kaggle/working/GA-Research/pose3d')"
    assert t.count(k) == 1; cells[2]["source"] = t.replace(k, k + PATCH3D)
    cells[0]["source"] = (f"# 3D SYMSOL-CHILE 7-GON (make_chile3d_runner.py), seed {seed}: the bottom-view recipe (14x14 map, 368 "
                          f"sphere points, start s_h {s}, constant lr) on poses uniform on SO(3), {epochs} epochs. Requested by a "
                          "human (2026-11-22): do the recipes for the 2D SYMSOL-CHILE work on the original 3D dataset?\n"
                          + "".join(cells[0]["source"]))
    return nb, name

if __name__ == "__main__":
    a = sys.argv[1:]
    nb, name = build(int(a[0]), int(a[1]), int(a[2]))
    json.dump(nb, open(a[3], "w")); print(name)
