"""Runners for the s_h (attention temperature) trace experiments on the 7-gon bottom view, built from the 14x14 runner
(make_bottom_grid14_variant.py). Adds to it:
  * a patch of the snapshot: POSE3D_SYMSOL_COS_LEARN=0 freezes s_h (a buffer instead of a parameter);
    POSE3D_SYMSOL_COS_LRMULT=F stores log s_h / F as the parameter, so Adam moves log s_h F times faster;
    POSE3D_SYMSOL_COS_MAX raises the cap on s_h (100 in the code);
  * a watcher thread in the notebook kernel: after every epoch it reads the run's resume checkpoint and appends to
    /kaggle/working/s_h_trajectory.jsonl all 32 log s_h (4 blocks x 8 heads) and, if s_h is learnable, Adam's moments of
    those parameters; at chosen epochs it saves the model state in half precision (snap_eNNN.pt) for CPU analysis.

usage: python make_sh_trace_variant.py S LEARN LRMULT MAX EPOCHS SNAP_EPOCHS(comma) WATCHDOG_MIN TAG OUT.ipynb
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_bottom_grid14_variant import make

PATCH = '''

def apply_sh(root):
    """s_h experiments: freeze it, move it faster (parameter = log s_h / F), raise its cap. Defaults change nothing."""
    p = lambda f: os.path.join(root, "pose3d", f)
    edit(p("models/gatr_denoiser.py"), 'scale=float(os.environ.get("POSE3D_SYMSOL_COS_SCALE", "30")), fused=True)',
         'scale=float(os.environ.get("POSE3D_SYMSOL_COS_SCALE", "30")), fused=True,\\n'
         '                learn=os.environ.get("POSE3D_SYMSOL_COS_LEARN", "1") == "1",\\n'
         '                max_scale=float(os.environ.get("POSE3D_SYMSOL_COS_MAX", "100")))')
    edit(p("models/cos_attention.py"), "        self.mode, self.max_scale, self.eps, self.fused = mode, max_scale, eps, fused or mode == \\"ktab\\"\\n",
         "        self.mode, self.max_scale, self.eps, self.fused = mode, max_scale, eps, fused or mode == \\"ktab\\"\\n"
         "        import os as _os\\n        self.lrmult = float(_os.environ.get(\\"POSE3D_SYMSOL_COS_LRMULT\\", \\"1\\"))\\n")
    edit(p("models/cos_attention.py"), "            self.log_scale = nn.Parameter(start)\\n",
         "            self.log_scale = nn.Parameter(start / self.lrmult)\\n")
    edit(p("models/cos_attention.py"), "        scale = self.log_scale.clamp(max=math.log(self.max_scale))",
         "        scale = (self.log_scale * self.lrmult).clamp(max=math.log(self.max_scale))")

apply_sh('/kaggle/working/GA-Research/pose3d')
print('patched: s_h freeze / speed / cap', flush=True)
'''

WATCH = '''# s_h TRACE (make_sh_trace_variant.py): a daemon thread reads the run's resume checkpoint after every epoch.
import threading, time, json, os, torch
_RES = f'/kaggle/working/{RUN_NAME}_resume.pth'
_LOG = '/kaggle/working/s_h_trajectory.jsonl'
_SNAP = set(SNAP_EPOCHS)
_KEYS = [f'vector_field.net.blocks.{b}.attention.attention.log_scale' for b in range(4)]
_LS_IDX = [195, 245, 295, 345]   # their indices among model.parameters() (checked on CPU for this model)

def _sh_poll_once(seen):
    if not os.path.exists(_RES):
        return
    mt = os.path.getmtime(_RES)
    if seen.get('mt') == mt:
        return
    st = torch.load(_RES, map_location='cpu', weights_only=False)
    ep = int(st['next_epoch'])            # epochs completed (1-based number of the last one)
    seen['mt'] = mt
    if seen.get('ep') == ep:
        return
    mult = float(os.environ.get('POSE3D_SYMSOL_COS_LRMULT', '1'))
    rec = {'epoch': ep, 'time': time.time(), 'log_s': [(st['model'][k].float() * mult).tolist() for k in _KEYS]}
    if os.environ.get('POSE3D_SYMSOL_COS_LEARN', '1') == '1':
        ost, adam = st['optimizer']['state'], []
        for i in _LS_IDX:
            s = ost.get(i, ost.get(str(i)))
            ok = s is not None and tuple(s['exp_avg'].shape) == (8,)
            adam.append({'m': s['exp_avg'].tolist(), 'v': s['exp_avg_sq'].tolist(), 'step': float(s['step'])} if ok else None)
        rec['adam'] = adam
    with open(_LOG, 'a') as fh:
        fh.write(json.dumps(rec) + '\\n')
    if ep in _SNAP:
        torch.save({k: (v.half() if v.is_floating_point() else v) for k, v in st['model'].items()},
                   f'/kaggle/working/snap_e{ep:03d}.pt')
    seen['ep'] = ep
    del st

def _sh_loop():
    seen = {}
    while True:
        try:
            _sh_poll_once(seen)
        except Exception as e:
            print('s_h watcher:', repr(e), flush=True)
        time.sleep(20)

threading.Thread(target=_sh_loop, daemon=True).start()
print('s_h watcher started; snapshots at epochs', sorted(_SNAP), flush=True)
'''

FINAL = '''# s_h TRACE: one last read after training (the thread may not have seen the final epoch yet).
try:
    _sh_poll_once({})
except Exception as e:
    print('s_h final read:', repr(e))
print(open(_LOG).read()[-2000:] if os.path.exists(_LOG) else 'no s_h trajectory written')
'''

def build(s, learn, lrmult, cap, epochs, snaps, watchdog, tag):
    nb, name = make(7, s, likelihood=False)
    cells = nb["cells"]
    def sub(i, old, new):
        t = "".join(cells[i]["source"]); assert t.count(old) == 1, (i, old); cells[i]["source"] = t.replace(old, new)
    name2 = f"bottom7_gatr_cosdir_grid14_nside8_{tag}"
    sub(0, f"RUN_NAME = '{name}'", f"RUN_NAME = '{name2}'\nSNAP_EPOCHS = {snaps}")
    sub(0, "'--n_epochs', '120',", f"'--n_epochs', '{epochs}',")
    sub(0, "WATCHDOG_MINUTES = (240, 680)", f"WATCHDOG_MINUTES = (240, {watchdog})")
    cells[0]["source"] = (f"# s_h TRACE RUN '{tag}': start s_h = {s}, learnable = {learn}, lr multiplier on log s_h = {lrmult}, "
                          f"cap = {cap}, {epochs} epochs. Launched 2026-11-20 at a human's request (question: is the 7-gon's "
                          f"breakthrough connected with a change of the learned s_h?).\n") + "".join(cells[0]["source"])
    t = "".join(cells[2]["source"]); k = "apply_grid('/kaggle/working/GA-Research/pose3d')"
    assert t.count(k) == 1; cells[2]["source"] = t.replace(k, k + PATCH)
    sub(5, "os.environ['POSE3D_SYMSOL_COS_SCALE'] = ",
        f"os.environ['POSE3D_SYMSOL_COS_LEARN'] = '{1 if learn else 0}'\nos.environ['POSE3D_SYMSOL_COS_LRMULT'] = '{lrmult}'\n"
        f"os.environ['POSE3D_SYMSOL_COS_MAX'] = '{cap}'\nos.environ['POSE3D_SYMSOL_COS_SCALE'] = ")
    code = lambda src: {"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None, "source": src}
    cells.insert(5, code(WATCH)); cells.append(code(FINAL))
    return nb, name2

if __name__ == "__main__":
    a = sys.argv[1:]
    nb, name = build(int(a[0]), a[1] == "1", a[2], a[3], int(a[4]), [int(x) for x in a[5].split(",")], int(a[6]), a[7])
    json.dump(nb, open(a[8], "w")); print(name)
