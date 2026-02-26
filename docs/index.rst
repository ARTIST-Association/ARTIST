.. artist documentation master file, created by
   sphinx-quickstart on Tue Feb 27 14:09:36 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to ``ARTIST``
=====================
``ARTIST`` stands for **AI-Enhanced Differentiable Ray Tracer for Irradiation Prediction in Solar Tower Digital Twins**.
The ``ARTIST`` package provides an implementation of a fully differentiable ray tracer using the PyTorch_
machine-learning framework in ``Python``. Leveraging automatic differentiation and GPU computation, ``ARTIST`` enables
gradient-based optimization within a differentiable solar tower power plant model using smooth parametric descriptions
of heliostats. While the underlying framework is designed to support the optimization of arbitrary plant components,
including towers and receivers, the current implementation focuses on data-driven heliostat surface reconstruction and
alignment.

.. figure:: ./images/juelich.png
   :width: 100 %
   :align: center

   The concentrating solar power (CSP) plant in Jülich, Germany.

|:sunny:| Our key contributions include:

- **Efficient heliostat calibration:** ``ARTIST`` combines a differentiable geometric model of heliostat kinematics with
  parallelized computation to enable efficient heliostat reconstruction from receiver flux measurements. This results in
  a flexible and robust calibration approach.

- **Accurate surface reconstruction and flux density prediction:** Leveraging learning Non-Uniform Rational B-Splines (NURBS),
  ``ARTIST`` reconstructs heliostat surfaces accurately using calibration images commonly available in solar thermal
  power plants. Thus, we can achieve sub-millimeter accuracy in mirror reconstruction from focal spot images,
  contributing to improved operational safety and efficiency. The reconstructed surfaces can be used for predicting
  unique heliostat flux densities with state-of-the-art accuracy. Check out this paper_ for more details:

     `M. Pargmann, J. Ebert, M. Götz et al. Automatic heliostat learning for in situ concentrating solar power plant
     metrology with differentiable ray tracing. Nat Commun 15, 6997 (2024).`

- **Immediate deployment**: ``ARTIST`` can be deployed from the beginning of a solar thermal power plant's operation,
  enabling in situ calibration and subsequent improvements in energy efficiency and cost reduction.

- **Optimized flux density:** ``ARTIST`` enables flux density optimization across an entire heliostat field by adjusting
  heliostat motor positions to obtain an optimal flux distribution on the receiver.

Quick Install
=============
To install the latest stable release from PyPI_, run the following in your terminal:

.. code-block:: console

    $ pip install artist-csp

You can check whether your installation was successful by importing ``ARTIST`` in ``Python``:

.. code-block:: python

   import artist

You can find more detailed installation instructions in :ref:`installation`.

Check out :ref:`usage` to find out more about to how to use ``ARTIST``.

.. toctree::
   :maxdepth: 1
   :caption: Contents

   install
   artist_under_the_hood
   heliostats
   nurbs_tutorial
   usage

.. Links
.. _PyTorch: https://pytorch.org/
.. _paper: https://doi.org/10.1038/s41467-024-51019-z
.. _PyPI: https://pypi.org/project/artist-csp/

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
