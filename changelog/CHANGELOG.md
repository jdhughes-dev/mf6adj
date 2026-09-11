# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.4.0] - 2026-09-11

### Breaking changes

- The adjoint state grew without bound backward in time on a model holding a
  cell that had gone dry. On a CONUS subdomain over four stress periods the
  composite well sensitivity came back as 3.0e+25, where a streamflow capture
  fraction cannot exceed 1. The previous head enters the specific-yield term
  through the saturation it sets, so the derivative carries the slope of that
  saturation, and MODFLOW 6 rounds the saturation at both ends of a cell where
  that slope falls away. The term asked only whether the saturation was above
  zero. A cell emptied to 3.3e-08 m above the bottom of a cell 20 m thick holds
  a saturation of 3e-19, which is above zero, and was handed the whole coupling
  of 16.2 against a matrix diagonal of 3e-04. The slope is now used.

- A cell holding no water then still carried a state of its own. Nothing flows
  through such a cell, so no sensitivity reaches through it, but under the
  Newton-Raphson formulation its row keeps a coefficient for every neighbor
  while losing its diagonal. The state was multiplied by the ratio of the
  storage term to that diagonal once per time step, 6.57 at the cell this was
  found on, and the composite came back as 1.5548 where the largest value
  anywhere else in the model was exactly 1.0000. The state is held at zero
  there now, as it already was for a constant head. With both terms corrected
  the composite on that subdomain is 1.0000. A sensitivity from an earlier
  release, for a transient model with cells that go dry, should be run again.

- The storage term that carries the adjoint state backward in time was formed
  over the length of the step being solved rather than the step it comes from,
  so a sensitivity was wrong wherever the time step changes length. Against a
  central difference of -0.54945055 taken by re-running the flow model, a run
  with a `tsmult` of 10 reported -0.55567957. A run that uses one time-step
  length throughout is unaffected, which is every case the tests had covered.

- `perturbation_method` moved a boundary value and did not put it back.
  MODFLOW 6 rereads a boundary only where the stress period data gives it
  again, so a rate given once and carried forward kept the perturbed value from
  the step it was made at through to the end of the run, and each step answered
  for every step after it. Over two time steps the method reported -1.097837
  against an independent difference of -0.549449, which is twice the right
  answer and grows with the number of steps. A single-step run is unaffected.

- A measure that reaches a time step whose matrix holds no solution stops there
  rather than solving on. The state of such a step is carried into the right
  side of every earlier one, so a sound matrix later in the recursion returns
  no number either, and a run used to write a whole file of values that are not
  numbers with nothing said. The measures of a run are independent, so the rest
  are still solved. The ones that stopped are named at the end of the run with
  the stress period and time step each reached, the step that stopped them is
  written so the cells holding no number can be read from it, and they are
  absent from the dictionary `solve_adjoint` returns.

- A run writes a log file where it wrote none before, taking the stem of the
  adjoint file when `logging_filename` is not given, and appends to that file
  rather than opening it to write. A run that reported something is commonly
  rerun to look into it, and opening the file to write erased the report being
  looked for. Pass `logging_filename=False` for no log file.

### Changes

- docs(release): describe the release process the workflow actually has (#130)
- feat(pm): write and read a performance measure from columns (#135)
- fix(pm): map a measure onto the reduced nodes without scanning the model (#136)
- docs: point the documentation links at the site that serves them (#137)
- fix(pm): carry the adjoint storage term over the next step's length (#142)
- feat(pm): report an adjoint matrix that cannot carry a sensitivity (#143)
- fix(sto): carry the storage term back with the slope of the smoothed saturation (#148)
- fix(adj): put back what a perturbation replaced, once the step is solved (#144)
- fix(pm): hold the adjoint state at zero where a cell holds no water (#150)
- docs(pm): say which form carries a quantity of the flow model alone (#146)
- fix(pm): report an adjoint solve that returned no numbers (#151)
- perf(pm): fall back on the point Jacobi preconditioner, not the block (#152)
- docs(examples): build the performance measures with the writer (#154)

## [1.3.0] - 2026-09-09

### Breaking changes

- Python 3.11 is now the minimum. modflowapi 1.0.0, which this release
  requires, does not support Python 3.9 or 3.10. `pyemu` is no longer a
  dependency, and `pandas` 3 is now allowed.

- A sensitivity to recharge was short by the area of the cell, because MODFLOW 6
  treats recharge as a rate over that area and the sensitivity was reported as
  the adjoint state alone. On a 250 m grid the reported value was too small by a
  factor of 62,500, and by a varying factor on a grid with variable cell sizes.
  Any recharge sensitivity from an earlier release is wrong by that factor.

- A sensitivity to hydraulic conductivity was formed from the conductance the
  grid describes rather than the one the model used, so a model with a
  horizontal flow barrier was wrong at every cell a barrier touched. On a
  confined test model the reported value was 694 times a finite-difference
  derivative. Cells away from a barrier were unaffected, and a model with no
  barrier is unchanged.

- A package given `AUXMULTNAME` scales the values it applies by an auxiliary
  variable, and MODFLOW applies that where it forms its terms rather than
  folding it into the values it keeps. The multiplier was not carried, so
  recharge, well, drain, river and general-head sensitivities were reported for
  the unscaled values.

- Storage terms are now selected the way MODFLOW 6 selects them, on the cell
  saturation, and under `SS_CONFINED_ONLY` the specific-storage term is dropped
  for a cell that is not full rather than scaled. Specific storage was
  previously applied at full cell thickness everywhere, so a partially saturated
  cell carried a storage term the forward model does not have.

- A specific-yield sensitivity is reported beside the specific-storage one, as
  `sy` in the composite results. A convertible cell releases water both ways and
  only one was reported before, so a storage sensitivity for such a cell was
  incomplete rather than merely differently named.

- A drain sitting on its activation threshold is dropped from the
  performance-measure derivative. A drain without a drainage depth switches on
  and off at its elevation and a converged solution leaves drains sitting on
  that corner, where the derivative is the conductance from one side and zero
  from the other. The full conductance was taken regardless.

- Three cases that used to run are now refused, each because the answer they
  produced could not be right:

  - a flow model that used XT3D, whose flow between two cells is not the
    conductance the sensitivity differentiates;
  - a performance measure of a specified flow, a well rate or a recharge rate,
    which has no derivative and was reported as zero everywhere;
  - a performance measure naming a package the adjoint forms no terms for,
    which used to fail later with a `KeyError` naming an object in an HDF file.

- A model whose matrix is not the derivative of its equations is now reported.
  Under the standard formulation the derivative of the transmissivity with
  respect to the head is lagged rather than assembled, so a model with
  convertible cells holds the transmissivity fixed in the sensitivity. On a
  single-layer unconfined model the error was 5.9 percent with the pumped cell
  near the middle of its thickness and 10.2 percent near the bottom. The term is
  not recovered; the condition is reported.

### Changes

- feat(pm): add lak6 performance measures and reject matrix-coupled packages (#79) (@jdhughes-dev)
- feat(pm): solve the lake water balance with the flow equations (#80) (@jdhughes-dev)
- feat(sfr): solve the reach routing with the flow equations (#81) (@jdhughes-dev)
- fix(pm): drop drain entries sitting on their activation threshold (#84) (@jdhughes-dev)
- fix(adj): select the storage terms the way MODFLOW 6 does (#87) (@jdhughes-dev)
- fix(pm): scale the recharge sensitivity by the cell area (#88) (@jdhughes-dev)
- fix(adj): reject a performance measure of a specified flow (#91) (@jdhughes-dev)
- feat(adj): report specific storage and specific yield separately (#93) (@jdhughes-dev)
- feat(sfr): differentiate a reach with a cross section (#94) (@jdhughes-dev)
- feat(sfr): differentiate the flow a diversion takes (#95) (@jdhughes-dev)
- docs(suppinfo): add supplemental technical information (#89) (@jdhughes-dev)
- fix(pm): carry the auxiliary multiplier into the recharge sensitivity (#96) (@jdhughes-dev)
- refactor: move standard package terms out of the solve loop (#98) (@jdhughes-dev)
- fix(packages): carry the auxiliary multiplier into the remaining sensitivities (#100) (@jdhughes-dev)
- feat(adj): report a model whose matrix is not the derivative of its equations (#103) (@jdhughes-dev)
- fix(adj): refuse an unsupported package while the input is read (#105) (@jdhughes-dev)
- refactor(adj): locate the adjoint matrix with the solution's sparsity (#107) (@jdhughes-dev)
- feat(maw): measure the exchange between a multi-aquifer well and the aquifer (#108) (@jdhughes-dev)
- fix(npf): carry a horizontal flow barrier into the conductivity sensitivity (#114) (@jdhughes-dev)
- feat(hfb): report the sensitivity to a barrier's hydraulic characteristic (#116) (@jdhughes-dev)
- fix(adj): refuse a flow model that used XT3D (#121) (@jdhughes-dev)
- build: drop Intel macOS, which MODFLOW 6 no longer builds for (#119) (@jdhughes-dev)
- build: let a task that runs other tasks fail when one of them does (#120) (@jdhughes-dev)
- build: lift the pandas bound in the environment files too (#122) (@jdhughes-dev)

## [1.2.0] - 2026-08-03

### Breaking changes

- `Mf6Adj.solve_adjoint()` and `PerfMeas.solve_adjoint()` no longer accept
  `skip_solve`. The flag applied to every performance measure form, but a
  transient `direct` or `residual` measure carries information backward from
  one time step to the next, so skipping a time step returned incorrect
  sensitivities with no indication that anything was wrong. Time steps with no
  entries are now skipped automatically, and only for the `instantaneous` form,
  where each time step is solved on its own and skipping is correct.

### Changes

- post v1.1.0 updates (#63) (@jdhughes-dev)
- Bump actions/checkout from 6 to 7 (#65) (@app/dependabot)
- Bump prefix-dev/setup-pixi from 0.9.6 to 0.10.0 (#67) (@app/dependabot)
- Bump actions/setup-python from 6 to 7 (#68) (@app/dependabot)
- fix(adj): detect IHIGHCELLSAT instead of comparing version strings (#69) (@jdhughes-dev)
- feat(pm)!: Add instantaneous performance measure type and remove skip_solve (#64) (@jdhughes-dev)
- ci(release): start a release from dropdowns and add rehearsal modes (#70) (@jdhughes-dev)
- ci(release): only allow a release to be cut from main (#71) (@jdhughes-dev)


## [1.1.0] - 2026-06-02

### Changes

- Release 1.0.0 (#53) (@app/github-actions)
- release: resync develop with main (#54) (@jdhughes-dev)
- release: develop resync after release (#55) (@jdhughes-dev)
- Bump prefix-dev/setup-pixi from 0.9.4 to 0.9.5 (#56) (@app/dependabot)
- Bump dawidd6/action-download-artifact from 19 to 20 (#57) (@app/dependabot)
- Bump dawidd6/action-download-artifact from 20 to 21 (#58) (@app/dependabot)
- Bump prefix-dev/setup-pixi from 0.9.5 to 0.9.6 (#59) (@app/dependabot)
- Add jacobi preconditioner (#60) (@jdhughes-dev)


## [1.0.0] - 2026-03-29

### Changes

- ruff formatting (#7) (@jdhughes-dev)
- remove use of local versions of python packages and executables (#8) (@jdhughes-dev)
- Add pyproject.toml (#9) (@jdhughes-dev)
- Std line endings (#10) (@jdhughes-dev)
- Refs/heads/feat mhtests (#11) (@jtwhite79)
- add support for disu grids (#12) (@jdhughes-dev)
- Feat dewater (#13) (@jtwhite79)
- add get-modflow bit to readme (#14) (@kmarkovich)
- fix lint issues (#16) (@jdhughes-dev)
- add pixi for ci (#17) (@jdhughes-dev)
- add support for high_cell_sat functionality (#18) (@jdhughes-dev)
- merge develop into main (#20) (@jdhughes-dev)
- Main (#21) (@jdhughes-dev)
- add pre-commit hook (#22) (@jdhughes-dev)
- v1.1.0rc (#23) (@jdhughes-dev)
- optimization and solver updates (#25) (@jdhughes-dev)
- Fix logger so that it can be called multiple times in a loop (#26) (@jdhughes-dev)
- Add custom dvclose convergence criteria callback for scipy solvers (#27) (@jdhughes-dev)
- feat(solve_adjoint): add rclose custom convergence check (#28) (@jdhughes-dev)
- Add option to skip adjoint solve for time steps without performance measures (#29) (@jdhughes-dev)
- feat(util): add workspace context manager (#30) (@jdhughes-dev)
- doc: add initial readthedocs files and GHActions workflow (#31) (@jdhughes-dev)
- doc: add rendered notebooks to readthedocs (#32) (@jdhughes-dev)
- docs: allow trigger_rtd with push or workflow_dispatch (#34) (@jdhughes-dev)
- Change GitHub token environment variable to RTDS (#35) (@jdhughes-dev)
- doc: fix paths for uploaded assets (#36) (@jdhughes-dev)
- rtd: fix issue with readthedocs push branch identification (#38) (@jdhughes-dev)
- ci: add dependabot and update release.yml (#39) (@jdhughes-dev)
- Bump dawidd6/action-download-artifact from 14 to 19 (#44) (@app/dependabot)
- Bump actions/upload-artifact from 4 to 7 (#43) (@app/dependabot)
- Bump actions/setup-python from 5 to 6 (#42) (@app/dependabot)
- Add rtds-action to project dependencies (#45) (@jdhughes-dev)
- Bump prefix-dev/setup-pixi from 0.9.3 to 0.9.4 (#40) (@app/dependabot)
- Bump actions/checkout from 4 to 6 (#41) (@app/dependabot)
- refactor: major refactor (#37) (@jdhughes-dev)
- doc: add rtd usage section (#47) (@jdhughes-dev)
- ci: update release markdown and add checklist to draft release PR (#48) (@jdhughes-dev)
- doc: update README.md for pypi and add citation (#49) (@jdhughes-dev)
- fix: change master -> main in release workflow and docs (#50) (@jdhughes-dev)


