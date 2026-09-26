# GA-Research

Geometric-algebra (Clifford) research experiments, one folder per experiment family.

| Folder | Experiment | Status |
|---|---|---|
| [`pose3d/`](pose3d/) | Image-conditioned 3D pose (SO(3)) estimation: Clifford Flow vs Image2Sphere, IPDF, matrix Fisher | active |

Each experiment folder is a self-contained Poetry project with its own README, dependencies,
configuration and experiment log. `main` always holds the reference recipe of every experiment;
experiments in progress live on branches until they are proven (see the workflow below).

## Working on `main`

* **Adopted change** (measured better): merged with its flag switched **on** in the experiment's
  `config.py`, or its value made the new default.
* **Not yet proven**: merged with its flag switched **off**, so it stays one `--flag` away but does
  not change the reference recipe.
* **Turned out worse**: not merged. It is written up in the experiment's `EXPERIMENTS.md` so nobody
  repeats it, and the branch stays available.

Details and the commit conventions are in [`docs/experiment-workflow.md`](docs/experiment-workflow.md).

## Adding an experiment family

Create a sibling folder next to `pose3d/` with its own `pyproject.toml`, `README.md`,
`EXPERIMENTS.md` and a `config.py` whose defaults are that experiment's reference recipe, and add a
row to the table above.
