Using mf6adj
============

mf6adj computes adjoint-based parameter sensitivities for MODFLOW 6 models.
The typical workflow is:

1. Locate the MODFLOW 6 binary and shared library.
2. Run the MODFLOW 6 model once with flopy to produce the baseline solution.
3. Write a performance-measure (``.adj``) file.
4. Instantiate :class:`~mf6adj.Mf6Adj`, solve the forward model, then solve
   the adjoint.
5. Read the sensitivity results.

Locating MODFLOW 6
------------------

:func:`~mf6adj.get_conda_mf6_paths` finds the MODFLOW 6 executable and the
``libmf6`` shared library inside the active conda environment:

.. code-block:: python

   import mf6adj

   mf6_bin, lib_name = mf6adj.get_conda_mf6_paths()

``mf6_bin`` is used to run the baseline simulation with flopy;
``lib_name`` is passed to :class:`~mf6adj.Mf6Adj`.

Run the baseline simulation before calling mf6adj:

.. code-block:: python

   import flopy

   flopy.run_model(exe_name=mf6_bin, namefile=None, model_ws="path/to/model")

Defining performance measures
------------------------------

A *performance measure* (PM) is the quantity whose sensitivity you want —
for example, the head at an observation well or the stream-aquifer exchange
at a reach.  PMs are written to a plain-text ``.adj`` file.

Each block follows this structure::

   begin performance_measure <name>
       <kper> <kstp> <loc> <type> <form> <weight> <obsval>
       ...
   end performance_measure

**Fields**

- ``<kper>`` — stress period (1-based)
- ``<kstp>`` — time step within the stress period (1-based)
- ``<loc>`` — cell location.  For structured grids use ``<layer> <row> <col>``
  (1-based); for unstructured grids use a single node number (1-based)
- ``<type>`` — quantity: ``head``, or a MODFLOW 6 package name such as
  ``sfr_1``, ``wel_1``, ``ghb_1``, etc.
- ``<form>`` — ``direct`` (sensitivity of the simulated value) or ``residual``
  (sensitivity of the squared misfit ``(simulated − obsval)²``)
- ``<weight>`` — scalar multiplier
- ``<obsval>`` — observed value used with ``residual`` form; use ``-1.0e+30``
  as a placeholder when using ``direct`` form

Multiple lines can appear in one block to form a composite PM that spans
multiple times or locations.  Multiple ``begin … end`` blocks can appear in
the same file; the adjoint is solved independently for each.

**Head at a single cell, one stress period:**

.. code-block:: python

   with open("model.adj", "w") as f:
       f.write("begin performance_measure obs_well\n")
       f.write("1 1 3 5 2 head direct 1.0 -1.0e+30\n")
       f.write("end performance_measure\n")

**Head at a single cell across all stress periods:**

.. code-block:: python

   with open("model.adj", "w") as f:
       f.write("begin performance_measure obs_well_all_times\n")
       for kper in range(nper):
           f.write(f"{kper + 1} 1 3 5 2 head direct 1.0 -1.0e+30\n")
       f.write("end performance_measure\n")

**Residual (squared misfit) form using observed head values:**

.. code-block:: python

   with open("model.adj", "w") as f:
       f.write("begin performance_measure head_misfit\n")
       for kper, obs in enumerate(observed_heads):
           f.write(f"{kper + 1} 1 3 5 2 head residual 1.0 {obs}\n")
       f.write("end performance_measure\n")

**Stream-aquifer exchange for SFR reaches:**

.. code-block:: python

   with open("model.adj", "w") as f:
       f.write("begin performance_measure sfr_reach_1\n")
       for kper in range(nper):
           f.write(f"{kper + 1} 1 2 4 1 sfr_1 direct 1.0 -1.0e+30\n")
       f.write("end performance_measure\n")

**Composite PM combining head and stream-aquifer exchange entries:**

.. code-block:: python

   with open("model.adj", "w") as f:
       f.write("begin performance_measure combo\n")
       for kper in range(nper):
           f.write(f"{kper + 1} 1 3 5 2 head direct 1.0 -1.0e+30\n")
       for kper in range(nper):
           f.write(f"{kper + 1} 1 2 4 1 sfr_1 direct 1.0 -1.0e+30\n")
       f.write("end performance_measure\n")

Choosing the form
------------------

``direct`` sums the measure over the entries it is given, and its sensitivity
is the total derivative: what a parameter does to the measure over the whole
run.  A measure with entries at one time step gives the derivative at that
time, and reproduces a finite-difference derivative exactly.

``residual`` is the same, on the squared difference from the observed value.

``instantaneous`` holds the earlier time steps fixed rather than letting them
feed back, so what it reports is a property of the model **and of the time
discretization together**.  The same model run with more time steps reports a
smaller sensitivity: on a three-period model measuring leakage to a stream,
5 time steps a period gave a value 1.4 percent below the derivative taken by
re-running the flow model, and 20 gave one 5.8 percent below it.

.. note::

   Use ``direct`` for a **streamflow capture fraction**, one measure per time
   the fraction is wanted at.  Capture at a time cannot exceed 1, and only the
   total derivative at a single time step carries that meaning.  A ``direct``
   measure whose entries span several time steps sums them, so it returns as
   many times the fraction as it holds times, and ``instantaneous`` returns a
   number that moves when the time stepping changes.

Writing a performance measure
------------------------------

The blocks above can be written with
:func:`~mf6adj.write_performance_measures` rather than by hand.  It takes the
measures as a dict of columns, a numpy recarray, or a
:class:`pandas.DataFrame`, so cell identifiers can come straight from flopy:

.. code-block:: python

   from mf6adj import write_performance_measures

   nstp = sim.tdis.nstp.array
   ghb = gwf.get_package("ghb-1").stress_period_data.get_data()[0]

   write_performance_measures(
       "model.adj",
       {
           "swgw": {
               "cellid": ghb["cellid"],       # zero-based, as flopy gives it
               "kper": nper - 1,              # one time, so one derivative
               "kstp": nstp[nper - 1] - 1,    # the last step of that period
               "pm_type": "ghb-1",
               "pm_form": "direct",
               "weight": 1.0,
               "obsval": -1.0e30,
           },
       },
       options={"hdf5_name": "out.h5"},
   )

Indices are zero-based, as they are in flopy; the file itself is one-based and
the conversion is made on the way out.  ``times`` is crossed with the cells,
the time varying slowest.  A period holding several time steps is given as a
``(kper, kstp)`` pair, which :func:`~mf6adj.all_times` builds from the time
steps per period::

   "times": all_times(sim.tdis.nstp.array)

``kper`` and ``kstp`` may be given instead of ``times``, in which case they are
one value per entry.  ``weight`` and ``obsval`` are a single value, one value
per cell, or one value per entry.

A measure with one entry per active cell of a large model makes a sizeable
file, and ``format="hdf5"`` writes the same measure as compressed columns,
roughly nine times smaller.  Ascii is the default, since it is the format
MODFLOW 6 input is written in and is readable in a text editor.

:func:`~mf6adj.read_performance_measures` reads either format, returning the
measures and the options block in the form the writer accepts, so a file can
be converted or edited and written back:

.. code-block:: python

   from mf6adj import read_performance_measures, write_performance_measures

   measures, options = read_performance_measures("model.adj")
   write_performance_measures("model.h5", measures, options=options, format="hdf5")

Streamflow capture
-------------------

Capture is the derivative of the leakage to a stream with respect to a well
rate, so it is the ``wel6_q`` result of a measure holding every stream cell.
It is wanted at a time, and a measure sums the entries it is given, so each
time wanted is its own measure:

.. code-block:: python

   from mf6adj import write_performance_measures

   nstp = sim.tdis.nstp.array
   drn = gwf.get_package("drn-1").stress_period_data.get_data()[0]["cellid"]
   ghb = gwf.get_package("ghb-1").stress_period_data.get_data()[0]["cellid"]

   measures = {}
   for kper, n in enumerate(nstp):
       measures[f"capture_{kper + 1:03d}"] = {
           "cellid": list(drn) + list(ghb),
           "kper": kper,                              # this period
           "kstp": n - 1,                             # its own last step
           "pm_type": ["drn-1"] * len(drn) + ["ghb-1"] * len(ghb),
           "pm_form": "direct",
           "weight": 1.0,
           "obsval": -1.0e30,
       }

   write_performance_measures("model.adj", measures)

Both packages belong to one measure, because the leakage is the sum over all of
the cells the stream occupies.  ``kstp`` is taken from ``nstp`` for that period
rather than written as a constant: where the periods hold different numbers of
time steps a constant lands part way through some of them, which is accepted
and is not the time meant.

The adjoint is solved once for each measure, so this costs one backward solve
per period.

Solving the forward model and adjoint
--------------------------------------

Pass the ``.adj`` filename and ``lib_name`` to :class:`~mf6adj.Mf6Adj`, then
call :meth:`~mf6adj.Mf6Adj.solve_forward_model` and
:meth:`~mf6adj.Mf6Adj.solve_adjoint`:

.. code-block:: python

   adj = mf6adj.Mf6Adj(
       "model.adj",
       lib_name,
       logging_level="INFO",
       working_directory="path/to/model",
   )

   adj.solve_forward_model()
   sensitivity_dfs = adj.solve_adjoint()
   adj.finalize()

:meth:`~mf6adj.Mf6Adj.solve_forward_model` runs MODFLOW 6 and writes the
solution components needed by the adjoint solver to an HDF5 file
(``forward.hd5`` by default; override with ``hdf5_name``).

:meth:`~mf6adj.Mf6Adj.solve_adjoint` solves the adjoint equations for each
PM and returns a ``dict`` mapping PM name to a :class:`pandas.DataFrame`
summary.  Adjoint solution arrays for each PM are written to
``adjoint_solution_<pm_name>.hd5`` in the working directory.

:meth:`~mf6adj.Mf6Adj.finalize` closes the MODFLOW 6 API and all file
handles; always call it when finished.

For larger models, use ``bicgstab`` with a preconditioner:

.. code-block:: python

   sensitivity_dfs = adj.solve_adjoint(
       linear_solver="bicgstab",
       linear_solver_kwargs={"maxiter": 500, "atol": 1e-5},
       use_precon=True,
   )

Reading the sensitivity results
---------------------------------

The returned ``sensitivity_dfs`` dictionary maps each PM name to a summary
:class:`pandas.DataFrame`.  Full per-cell sensitivity arrays are in the HDF5
output files:

.. code-block:: python

   import h5py

   with h5py.File("path/to/model/adjoint_solution_obs_well.hd5", "r") as hdf:
       # top-level keys are stress period / time step identifiers
       # plus a "composite" group summed over all entries
       composite = hdf["composite"]
       for key in composite.keys():
           print(key, composite[key][()])

Complete minimal example
-------------------------

.. code-block:: python

   import mf6adj

   # set the path to the MODFLOW 6 executable and shared library if they are 
   # installed in the active conda environment
   mf6_bin, lib_name = mf6adj.get_conda_mf6_paths()

   # write the performance-measure file
   with open("mymodel/mymodel.adj", "w") as f:
       f.write("begin performance_measure head_obs\n")
       f.write("1 1 1 5 5 head direct 1.0 -1.0e+30\n")
       f.write("end performance_measure\n")

   # solve forward and adjoint
   adj = mf6adj.Mf6Adj(
       "mymodel.adj",
       lib_name,
       working_directory="mymodel",
   )
   adj.solve_forward_model()
   sensitivity_dfs = adj.solve_adjoint()
   adj.finalize()

   print(sensitivity_dfs["head_obs"])
