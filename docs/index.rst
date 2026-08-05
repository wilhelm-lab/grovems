.. jinja:: project_info

   .. {{ project }} documentation master file.
      You can adapt this file completely to your liking, but it should at least
      contain the root `toctree` directive.

   {{ project }} - Main Document
   =============================

   |docs-badge| - |build-badge| - |pypi-badge|

   .. |docs-badge| image:: https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat
       :target: https://{{ project }}.readthedocs.io/en/latest/?badge=latest
       :alt: Documentation Status
   .. |build-badge| image:: https://github.com/{{ project }}/actions/workflows/build.yaml/badge.svg
       :target: https://github.com/{{ project }}/actions/workflows/build.yaml
       :alt: Build Status
   .. |pypi-badge| image:: https://github.com/{{ project }}/actions/workflows/pypi.yaml/badge.svg
       :target: https://github.com/{{ project }}/actions/workflows/pypi.yaml
       :alt: PyPI Status

   Oktoberfest/Percolator rescoring + PSA + SUOD IForest scoring for denovo_fdr.

   .. image:: assets/full_pipeline.png
      :alt: Pipeline overview: FragPipe/Casanovo search results feed PSMs into Oktoberfest feature generation, then PSA similarity grading, then isolation-forest rescoring

   .. automodule:: {{ project }}
       :members:

   .. include:: notes/installation.rst
   .. include:: notes/quickstart.rst

   .. toctree::
      :glob:
      :maxdepth: 2
      :caption: How To

      notes/installation
      notes/quickstart
      notes/citation

   .. toctree::
      :maxdepth: 2
      :caption: Package Reference

      main

   Indices and tables
   ==================

   * :ref:`genindex`
   * :ref:`modindex`
   * :ref:`search`
