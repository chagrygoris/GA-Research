# pose3d

Predict an object's 3D rotation (SO(3)) from one RGB image on Pascal3D+. The main model, **Clifford
Flow**, is conditional flow matching on Spin(3) rotors built from geometric-algebra layers.
Reference result: **10.25°** median rotation error (W&B `k5sblpo8`).

```bash
cd pose3d && poetry install
poetry run python -m pose3d --path_to_datasets /path/to/data --run_name my-run
```

Setup and how to run: [`USAGE.md`](USAGE.md). Results: [`reports/`](reports/). Reading material:
[`references/`](references/README.md).

## Idea board

This table is the list of experiments for pose3d. Keep it up to date, in the same commit or pull
request as your work.

1. **Propose.** Add a row: next free ID, one line for the idea, your name under *Proposed by*,
   status `proposed`. Anyone can do this.
2. **Claim.** Put your name under *Developer* and set the status to `in progress`. Work on a branch
   named `idea-<ID>-<short-name>` off `main` and write it in *Branch*. One idea, one flag: the
   default must keep reproducing the reference run (see the root README).
3. **Report.** When you have results, add a Typst report to `reports/`, link it under *Report* and
   set the final status.

| Status | Meaning | What happens on `main` |
|---|---|---|
| `proposed` | nobody works on it yet | nothing |
| `in progress` | has a developer and a branch | nothing |
| `adopted` | measured better than the reference | merged, flag `True` or new default |
| `experimental` | promising, not proven (one run, inside the noise, unfinished) | merged, flag `False` |
| `dropped` | worse, or not worth the complexity | not merged; the report says why |

| ID | Idea | Proposed by | Developer | Status | Branch | Report |
|---|---|---|---|---|---|---|
