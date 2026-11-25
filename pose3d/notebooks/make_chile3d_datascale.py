"""Runners for the data-scaling test on the 3D SYMSOL-CHILE 7-gon (2026-11-25): the recipe of make_chile3d_runner.py
(ImageNet ResNet-50, 14x14 map, 368 sphere points, start s_h 100, constant lr, seed 0) with N training views and a
fixed budget of about 20,000 optimizer steps of 64 images (epochs = ceil(20000 / (N / 64))), on 4xL4. Only the
number of distinct training views changes; the failed runs had N = 4,500 (180 epochs = 12,650 steps).

usage: python make_chile3d_datascale.py N OUT.ipynb   -> prints the run name"""
import json, math, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_chile3d_runner import build

def build_n(n, steps=20000):
    ep = math.ceil(steps / (n / 64)); nb, name = build(0, ep, 690); cells = nb["cells"]
    def sub(old, new):
        hits = [c for c in cells if old in "".join(c["source"])]; assert len(hits) == 1, (old, len(hits))
        t = "".join(hits[0]["source"]); assert t.count(old) == 1, old; hits[0]["source"] = t.replace(old, new)
    sub("'--symsol_train_views', '4500'", f"'--symsol_train_views', '{n}'")
    sub("'--train', '4500'", f"'--train', '{n}'")
    sub("'--num_gpus', '2'", "'--num_gpus', '4'")
    new = name.replace("chile3d_pyr7", f"chile3d_pyr7_n{n // 1000 if n % 1000 == 0 else n / 1000}k")
    sub(f"RUN_NAME = '{name}'", f"RUN_NAME = '{new}'")
    cells[0]["source"] = (f"# DATA-SCALING TEST (2026-11-25, human-approved): 3D 7-gon, {n} training views, {ep} epochs = ~{steps} steps "
                          "of 64 (the failed runs: 4,500 views, 180 epochs = 12,650 steps); recipe otherwise unchanged (ImageNet "
                          "ResNet-50). 4xL4, ETA ~5 h.\n") + "".join(cells[0]["source"])
    return nb, new

if __name__ == "__main__":
    nb, name = build_n(int(sys.argv[1])); json.dump(nb, open(sys.argv[2], "w")); print(name)
