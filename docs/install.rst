.. _installation:

Installation
============

We recommend installing ``ARTIST`` in a separate virtual Python environment:

.. code-block:: console

   $ python3 -m venv ./artist-venv
   $ source ./artist-venv/bin/activate
   $ pip install --upgrade pip

You can install the latest stable version of ``ARTIST`` directly from PyPI using:

.. code-block:: console

    $ pip install artist-csp

To install ``ARTIST`` with extra dependencies for running the tutorials, you will need to use:

.. code-block:: console

    $ pip install "artist-csp[tutorials]"

or for the extra dependencies required for the examples:

.. code-block:: console

    $ pip install "artist-csp[examples]"

Alternatively, you can install the latest development version directly from the GitHub repository:

.. code-block:: console

    $ pip install git+https://github.com/ARTIST-Association/ARTIST.git

If you want to get the source code and modify it, you can clone the repository with ``git`` and install ``ARTIST``
with ``pip`` in editable mode:

.. code-block:: console

    $ git clone https://github.com/ARTIST-Association/ARTIST
    $ pip install -e .

If you wish to install the developer dependencies, run:

.. code-block:: console

   $ pip install -e ."[dev]"

You can check whether your installation was successful by importing ``ARTIST`` in ``Python``:

.. code-block:: python

   import artist
