.. _installation:

Installation
============

For best results, we recommend installing ``ARTIST`` in a separate virtual environment:

.. code-block:: console

   $ python3 -m venv ./ARTIST
   $ source ./ARTIST/bin/activate
   $ pip install --upgrade pip

The latest stable release can easily be installed from `PyPI`_ using ``pip``:

.. code-block:: console

    $ pip install artist

If you need the latest updates, you can also install ``ARTIST`` directly from the `Github main branch`_ at you own risk:

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


.. Links
.. _PyPI: [Include Link Here]
.. _Github main branch: https://github.com/ARTIST-Association/ARTIST
.. _OpenMPI: https://www.open-mpi.org/
