"""engine/resume.py: an interrupted and resumed run matches an uninterrupted one; SIGTERM exits with 75; a
checkpoint from other training options is refused."""

import dataclasses
import os
import signal
import subprocess
import sys
import time

import pytest
import torch

from pose3d.config import parse_args
from pose3d.engine.resume import RESUMABLE_EXIT, load_resume, resume_path, save_resume


def _cfg(tmp_path, *extra):
    cfg = parse_args(["--path_to_datasets=/tmp", "--run_name=toy", f"--resume_dir={tmp_path}", "--n_epochs", "6", *extra])
    cfg.device = torch.device("cpu")
    return cfg


def _objects(seed=0):
    torch.manual_seed(seed)
    model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 1))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=6)
    return model, opt, sched


def _epoch(model, opt):
    x = torch.randn(32, 4)                      # draws from the global RNG, so the RNG state must be restored too
    loss = (model(x) - x.sum(1, keepdim=True)).pow(2).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()


def test_interrupted_and_resumed_run_matches_an_uninterrupted_one(tmp_path):
    cfg = _cfg(tmp_path)
    model, opt, sched = _objects()
    torch.manual_seed(1)
    for _ in range(6):
        _epoch(model, opt)
        sched.step()
    reference = [p.detach().clone() for p in model.parameters()]
    reference_lr = sched.get_last_lr()

    model, opt, sched = _objects()
    torch.manual_seed(1)
    for i in range(3):                           # "killed" after epoch 3
        _epoch(model, opt)
        sched.step()
        save_resume(cfg, model, opt, sched, None, next_epoch=i + 1)
    torch.manual_seed(12345)                     # the new process starts from another RNG state

    model, opt, sched = _objects(seed=7)         # fresh objects, other initial weights
    cfg_resumed = _cfg(tmp_path, f"--resume={resume_path(cfg)}", "--run_name=toy_cont")
    start = load_resume(cfg_resumed, model, opt, sched, None)
    assert start == 3
    for _ in range(start, 6):
        _epoch(model, opt)
        sched.step()
    for p, q in zip(model.parameters(), reference):
        torch.testing.assert_close(p, q, rtol=0, atol=0)
    assert sched.get_last_lr() == reference_lr
    assert not resume_path(cfg).with_name(resume_path(cfg).name + ".tmp").exists()


def test_a_checkpoint_from_other_training_options_is_refused(tmp_path):
    cfg = _cfg(tmp_path)
    model, opt, sched = _objects()
    save_resume(cfg, model, opt, sched, None, next_epoch=2)
    other = _cfg(tmp_path, f"--resume={resume_path(cfg)}", "--lr", "3e-3")
    with pytest.raises(ValueError, match="lr"):
        load_resume(other, *_objects(), None)


def test_sigterm_exits_with_the_resumable_code(tmp_path):
    code = ("import time\nfrom pose3d.engine.resume import install_sigterm_exit\ninstall_sigterm_exit()\n"
            "print('ready', flush=True)\ntime.sleep(60)\n")
    proc = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True,
                            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    assert proc.stdout.readline().strip() == "ready"
    proc.send_signal(signal.SIGTERM)
    assert proc.wait(timeout=30) == RESUMABLE_EXIT
