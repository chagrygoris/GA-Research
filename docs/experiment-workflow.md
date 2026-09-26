# Experiment workflow

`main` is the reference: running an experiment folder with no flags reproduces its best documented
result. Everything else is a flag away.

## Life of a change

1. **Branch** off `main` (`git switch -c <short-name>`). Implement the change behind a config field
   whose default reproduces the current reference exactly (a boolean in `Features`, or a numeric
   option whose default is the current behaviour). Nothing else in the recipe may move.
2. **Run** it on the reference harness and compare `final_median_rotation_error` against the
   reference number in the experiment's `EXPERIMENTS.md`. Diff the W&B `runInfo.args` of both runs
   first; a comparison between runs with different args is not a comparison.
3. **Decide**, per result:
   * clearly better -> merge with the flag **True** (or the new value as default), update the
     reference row in `EXPERIMENTS.md`;
   * promising but not proven (single run, inside noise, unfinished) -> merge with the flag
     **False**, add a row under *Experimental* with the evidence so far;
   * worse or not worth the complexity -> do not merge; add a row under *Not adopted* with the
     number and the branch name.

## Suggestions

* One change per branch, one flag per change. Combined ablations are run by passing several flags.
* A flag that is False must leave the code path of the reference untouched (no extra ops in the
  forward pass, no changed initialisation).
* Runs are not seeded, so compare against a baseline run from the same commit where you can; a
  single number logged from another commit is a weaker reference.
* Keep numbers in `EXPERIMENTS.md` traceable: W&B run id, branch or commit, the args that differ.
