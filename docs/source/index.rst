.. Nephological Semantics documentation master file, created by
   sphinx-quickstart on Fri Aug 20 14:45:21 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Nephological Semantics
======================

Welcome! This is the current home website of the Nephological Semantics Project,
developed in the QLVL research group at KU Leuven.
You can learn more about the project :doc:`here <about>`.

One of the main products of our project is Nephosem, a Python package with functions
to create type- and token-level distributional models, both with bag-of-words and
dependency information. You can find here :doc:`the documentation <nephosem>`.

.. image:: https://zenodo.org/badge/233318567.svg

This package has been used in lexical semantics and lectometry studies within
the Nephological Semantics projects; the derived publications are listed :ref:`here <publications>`.

Semasiological workflow
-----------------------

The semasiological workflow looks at the internal structure of individual words based on the
contexts of their occurrences. For each word, it creates multiple token-level models -vector representations
of each of its instances- combining different parameter settings (i.e. ways of defining context).
Then it selects representative models and visualizes them in an `interactive tool <https://qlvl.github.io/NephoVis/>`_.
A more or less technical explanation of the procedure is explained `here <cloudspotting.marianamontes.me/workflow.html>`_.

The Nephosem package is at the core of this workflow, but is then expanded with other tools:

* The `semasioFlow <https://github.com/montesmariana/semasioFlow>`_ Python package,
which organizes and compacts Nephosem functions in a way specific to the semasiological workflow;
* The `semcloud <https://github.com/montesmariana/semcloud>`_ R package, which takes the output
of semasioFlow and prepares the data for visualization, running dimensionality reduction and clustering
and generating annotated concordances[#ann]_.
* The `nephoVis <https://github.com/QLVL/NephoVis>`_ interactive visualization tool (see link above)
for exhaustive, qualitative exploration of the models.
* The `Level 3 ShinyApp <https://github.com/montesmariana/Level3>`_ for deeper exploration of individual models.


Lectometric workflow
--------------------

Coming soon!

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   usage
   Documentation <nephosem>
   Repository <https://github.com/QLVL/nephosem/>
   about



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. rubric:: Footnotes

.. [#ann] These are not semantic annotations but model-related: context words captured by a given model are
highlighted and weighting values may be included as superscript.