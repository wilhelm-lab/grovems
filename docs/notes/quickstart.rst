.. jinja:: project_info

   Quickstart
   **********

   Copy the default config and point it at your inputs:

   .. code-block:: bash

     cp assets/default_config.yaml my_config.yaml
     # edit database_search_path / denovo_search_path / rawdata_path

     grovems run --config my_config.yaml

   Any config value can be overridden per-run instead of editing the file, with
   ``--set key=value`` (repeatable):

   .. code-block:: bash

     grovems run --config my_config.yaml \
       --set run_iforest=false \
       --set outdir=/path/to/output

   Standalone IForest run
   ***********************

   If you already have a ``psa_dataframes/merged_SCAN`` from an earlier
   ``run_psa: true`` run, you can run just the IForest stage against it:

   .. code-block:: bash

     grovems run --config my_config.yaml \
       --set run_rescoring=false \
       --set run_psa=false \
       --set run_iforest=true \
       --set psa_merged_scan_dir=/path/to/psa_dataframes/merged_SCAN

   On a SLURM cluster
   *******************

   See ``tutorials/run_pipeline.slurm`` -- a single ``sbatch`` job that activates the
   environment and runs ``grovems run``. Size the job's ``#SBATCH --cpus-per-task``/
   ``--mem``/``--time`` to your data volume, and pass matching values for
   ``num_threads``/``psa_max_workers``/``percolator_threads`` in the config.

   One real file, locally
   ************************

   ``./test.sh`` builds a minimal single-raw-file fixture from the production
   proteometools dataset (symlinked, nothing copied) and runs the pipeline against it
   with IForest disabled -- see ``tests/integration_tests/README.md``.
