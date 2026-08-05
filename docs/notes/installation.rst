.. jinja:: project_info

   Installation
   ************

   {{ project }} is a plain Python package -- install it into a conda/virtualenv
   environment:

   .. code-block:: bash

     pip install -e .
     # or: poetry install --with dev --extras docs

   This pulls in every Python dependency the pipeline needs, including the
   ``feature/denovo_fdr`` branches of ``oktoberfest``, ``spectrum_fundamentals``, and
   ``spectrum_io`` (see ``pyproject.toml``).

   Percolator
   **********

   Provided externally -- not a {{ project }} dependency. Point ``percolator_exe:
   /path/to/percolator`` at a local install in your config YAML (default: bare
   ``percolator``, resolved via ``PATH``), or set ``percolator_module:
   percolator/3.7.1`` for sites that provide it as an environment module instead
   (loaded via ``module load`` before ``percolator_exe`` runs -- requires the invoking
   shell to have Lmod's init sourced).

   A scheduler (optional)
   ***********************

   {{ project }} runs fine with no scheduler at all -- it's a single sequential
   process. For SLURM, see :doc:`quickstart` and
   ``tutorials/run_pipeline.slurm``: one ``sbatch`` job that activates the
   environment and runs ``grovems run``.
