"""Variants of bottom7-gatr-cosdir-grid14-nside8-runner.ipynb: another pyramid (n = 7, 10, 13) or another start value of
the cosine attention's inverse temperature s_h.

usage: python make_bottom_grid14_variant.py N S OUT.ipynb [--no-likelihood]
"""
import json, os, sys

SETS = {7: "7", 10: "8", 13: "9"}          # --symsol_set of the bottom view of pyr<N> (datasets/symsol.py)

def make(n, s, likelihood=True):
    here = os.path.dirname(os.path.abspath(__file__))
    nb = json.load(open(os.path.join(here, "bottom7-gatr-cosdir-grid14-nside8-runner.ipynb")))
    name = f"bottom{n}_gatr_spherecloud_cosdir_h8_grid14_nside8_s{s}_constlr"
    def sub(text, old, new):
        assert text.count(old) == 1, old
        return text.replace(old, new)
    c0 = "".join(nb["cells"][0]["source"])
    c0 = sub(c0, "RUN_NAME = 'bottom7_gatr_spherecloud_cosdir_h8_grid14_nside8_s60_constlr'", f"RUN_NAME = '{name}'")
    c0 = sub(c0, "'--symsol_set', '7'", f"'--symsol_set', '{SETS[n]}'")
    if not likelihood:
        c0 = sub(c0, "'--n_epochs', '120',", "'--n_epochs', '120', '--no-symsol_likelihood',")
    c0 = (f"# VARIANT (make_bottom_grid14_variant.py): the {n}-gonal pyramid, cosine attention started at s_h = {s}"
          f"{'' if likelihood else ', no final log-likelihood'}; guessed-turn level {90 / n:.1f} deg.\n") + c0
    nb["cells"][0]["source"] = c0
    for cell in nb["cells"][1:]:
        t = "".join(cell["source"])
        if "'--shapes', 'pyr7'" in t:
            t = sub(t, "'--shapes', 'pyr7'", f"'--shapes', 'pyr{n}'")
        if "POSE3D_SYMSOL_COS_SCALE'] = '60'" in t:
            t = sub(t, "POSE3D_SYMSOL_COS_SCALE'] = '60'", f"POSE3D_SYMSOL_COS_SCALE'] = '{s}'")
        cell["source"] = t
    return nb, name

if __name__ == "__main__":
    nb, name = make(int(sys.argv[1]), int(sys.argv[2]), "--no-likelihood" not in sys.argv)
    json.dump(nb, open(sys.argv[3], "w")); print(name)
