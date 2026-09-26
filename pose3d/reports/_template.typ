// Report template for one experiment (idea) of the pose3d family.
// Copy to `YYYY-MM-<slug>.typ`, fill it in, compile with `typst compile <file>.typ`.
// Keep it short: what was tried, how it differs from the reference run, the numbers, the decision.

#let meta(idea: "", developer: "", branch: "", date: "", runs: ()) = {
  table(
    columns: (auto, 1fr),
    stroke: none,
    inset: (x: 0pt, y: 3pt),
    [*Idea*], [#idea],
    [*Developer*], [#developer],
    [*Branch*], [#raw(branch)],
    [*Date*], [#date],
    [*W\&B runs*], [#runs.map(r => raw(r)).join(", ")],
  )
}

#set document(title: "Report title", author: "Your name")
#set page(paper: "a4", margin: 2cm, numbering: "1")
#set text(size: 11pt)
#set heading(numbering: "1.")
#show table: set text(size: 10pt)

#align(center)[
  #text(size: 18pt, weight: "bold")[Report title]
]

#meta(
  idea: "ID and title of the row on the idea board (pose3d/README.md)",
  developer: "Your name",
  branch: "idea-00-short-name",
  date: "YYYY-MM-DD",
  runs: ("wandb-run-id",),
)

= Question

What did you want to find out, and why (one paragraph).

= Setup

What differs from the reference run: the flags (`--flag value`), the commit, anything else.
Diff the W\&B `runInfo.args` of your run and the reference run and paste the difference here.

= Results

#figure(
  table(
    columns: (1fr, auto, auto),
    align: (left, right, right),
    table.header[*Run*][*final median error (deg)*][*Acc\@15*],
    [reference `k5sblpo8`], [10.25], [],
    [this experiment], [], [],
  ),
  caption: [Pascal3D+, 32-sample medoid evaluation.],
)

Plots, per-class numbers, qualitative examples.

= Decision

Pick one and say why; it decides what happens on `main`.

- *adopted*: better than the reference; merge with the flag set to `True` (or the value as the new default).
- *experimental*: promising but not proven (one run, inside the noise, unfinished); merge with the flag `False`.
- *dropped*: worse or not worth the complexity; do not merge, keep the branch.

= Next steps

What would settle it, or what this opens up.
