.. _installation:

Installation
============

For best results, we recommend installing ``ARTIST`` in a separate virtual environment:

.. code-block:: console

   $ python3 -m venv ./ARTIST
   $ source ./ARTIST/bin/activate
   $ pip install --upgrade pip

You can install the latest stable version of ``ARTIST`` directly from PyPI using:

.. code-block:: console

    $ pip install artist

To install ``ARTIST`` with extra dependencies to run the tutorials you will need to use:

.. code-block:: console

    $ pip install "artist[tutorials]"

or for the extra dependencies required for the examples:

.. code-block:: console

    $ pip install "artist[examples]"

Alternatively, you can install the latest development version directly from the official GitHub repository:

.. code-block:: console

    $ pip install https://github.com/ARTIST-Association/ARTIST

If you want to get the source code and modify it, you can clone the source code using ``git`` and install ``ARTIST``
with ``pip``:

.. code-block:: console

    $ git clone https://github.com/ARTIST-Association/ARTIST
    $ pip install -e .

Alternatively, if you wish to install the developer dependencies:

.. code-block:: console

   $ pip install -e ."[dev]"

You can check whether your installation was successful by importing ``ARTIST`` in ``Python``:

.. code-block:: python

   import artist
