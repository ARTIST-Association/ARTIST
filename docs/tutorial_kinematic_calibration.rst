.. _tutorial_kinematic_calibration:

``ARTIST`` Tutorial: Kinematic Calibration
==========================================

.. note::

    You can find the corresponding ``Python`` script for this tutorial here:
    https://github.com/ARTIST-Association/ARTIST/blob/main/tutorials/04_kinematic_calibration_tutorial.py

This tutorial explains how kinematic calibration is performed in ``ARTIST``, specifically we will look at:

- Why do we need to calibrate the kinematic?
- How to load calibration data.
- How to set up the ``KinematicCalibrator`` responsible for the kinematic calibration.

Before starting this scenario make sure you already know how to :ref:`load a scenario<tutorial_heliostat_raytracing>`,
run ``ARTIST`` in a :ref:`distributed environment for raytracing<tutorial_distributed_raytracing>`, and understand the
structure of a :ref:`scenario<scenario>`. If you are not using your own scenario, we recommend using one of the
"test_scenario_paint_multiple_heliostat_groups_ideal.h5" or "test_scenario_paint_multiple_heliostat_groups_deflectometry.h5"
scenarios provided in the "scenarios" folder.

Kinematic Calibration Basics
----------------------------
In the real world most components of the kinematic have mechanical errors. This means if we tell an actuator to orient
a heliostat along a specific angle, the heliostat might not end up pointing exactly at the specified aim point.
In ``ARTIST`` we create a digital twin of a solar tower power plant. In the computer simulation, the heliostat will, per default,
point exactly where we tell it to. To keep the predictions made with ``ARTIST`` as accurate as possible we need to
consider the mechanical errors and offsets of the real-world kinematic. In the kinematic calibration process the kinematic module
learns all offset or deviation parameters of the real-world kinematic, to mimic its behavior.

Loading the Calibration Data
----------------------------
Calibrating the kinematic in ``ARTIST`` requires calibration data. In this tutorial we consider calibration data from
the ``PAINT`` database: https://paint-database.org/. Multiple heliostats can be calibrated simultaneously and each
heliostat can be calibrated with multiple calibration data points at once.

Calibration data consists of calibration properties and a flux image. We load this data with a mapping, analog to the
appraoch shown in the tutorial on :ref:`surface reconstruction<tutorial_surface_reconstruction>`. Specifically, we
create a ``heliostat_data_mapping`` list of tuples, where each tuple contains the heliostat's name and the paths to its
calibration data, which include both a ``.json`` file with calibration properties and a ``.png`` flux image:


.. code-block::

    # Please use the following style: list[tuple[str, list[pathlib.Path], list[pathlib.Path]]]
    heliostat_data_mapping: list[tuple[str, list[pathlib.Path], list[pathlib.Path]]] = [
    (
        "heliostat_name_1",
        [
            pathlib.Path(
                "please/insert/the/path/to/the/paint/data/here/calibration-properties.json"
            ),
            # ....
        ],
        [
            pathlib.Path("please/insert/the/path/to/the/paint/data/here/flux.png"),
            # ....
        ],
    ),
    (
        "heliostat_name_2",
        [
            pathlib.Path(
                "please/insert/the/path/to/the/paint/data/here/calibration-properties.json"
            ),
            # ....
        ],
        [
            pathlib.Path("please/insert/the/path/to/the/paint/data/here/flux.png"),
            # ....
        ],
    ),
    # ...
    ]

This data is then saved into a data dictionary which will be later used in the optimization:

.. code-block::

    # Create dict for the data source name and the heliostat_data_mapping.
    data: dict[str, str | list[tuple[str, list[pathlib.Path], list[pathlib.Path]]]] = {
        config_dictionary.data_source: config_dictionary.paint,
        config_dictionary.heliostat_data_mapping: heliostat_data_mapping,
    }

If you are not using your own data, you can use the sample data provided in the "data", for example for the heliostats
AA31, AA39, and AC43.

Next, you can load the scenario and set up the distributed environment as in previous tutorials.

Configuring Scheduler and Optimizer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

As in the :ref:`surface reconstruction<tutorial_surface_reconstruction>` tutorial, the kinematic optimizer also uses the
``torch.optim.Adam`` optimizer. Therefore we again need to define the parameters used for the learning rate scheduler
and the optimization configuration:

.. code-block::

    scheduler = (
        config_dictionary.exponential
    )  # exponential, cyclic or reduce_on_plateau
    scheduler_parameters = {
        config_dictionary.gamma: 0.9,
        config_dictionary.min: 1e-6,
        config_dictionary.max: 1e-3,
        config_dictionary.step_size_up: 500,
        config_dictionary.reduce_factor: 0.3,
        config_dictionary.patience: 10,
        config_dictionary.threshold: 1e-3,
        config_dictionary.cooldown: 10,
    }

    # Set optimization parameters.
    optimization_configuration = {
        config_dictionary.initial_learning_rate: 0.0005,
        config_dictionary.tolerance: 0.0005,
        config_dictionary.max_epoch: 1000,
        config_dictionary.num_log: 100,
        config_dictionary.early_stopping_delta: 1e-4,
        config_dictionary.early_stopping_patience: 10,
        config_dictionary.scheduler: scheduler,
        config_dictionary.scheduler_parameters: scheduler_parameters,
    }

Now we are ready to set up the kinematic calibration.

Setting up the ``KinematicCalibrator``
--------------------------------------

Before we can create a ``KinematicCalibrator`` object we need to decide which method we want to use to perform calibration.
Currently there are two methods to calibrate the kinematic. Either we use geometric considerations and the
motor positions from the calibration data or we optimize using flux density distributions and the differentiable
ray tracer. Choosing the optimization method depends on the available calibration data.
Both methods need information about:

- The centers of the measured flux density distributions,
- The incident ray directions during the measurements,

In this tutorial we use the raytracing method, since our experiments show this is slightly more robust:

.. code-block::

     kinematic_calibration_method = config_dictionary.kinematic_calibration_raytracing

Now we can create a ``KinematicCalibrator`` object responsible for the kinematic calibration:

.. code-block::

    kinematic_calibrator = KinematicCalibrator(
        ddp_setup=ddp_setup,
        scenario=scenario,
        data=data,
        optimization_configuration=optimization_configuration,
        calibration_method=kinematic_calibration_method,
    )


Performing Calibration
----------------------
The set up is now complete and the kinematic calibration can begin. The kinematic calibration is an optimization process.
Before starting the calibration we need to define the loss, in this tutorial we use the ``FocalSpotLoss`` since we are
working with raytracing, however for the motor positions variant a ``VectorLoss`` would be required:

.. code-block::

    loss_definition = FocalSpotLoss(scenario=scenario)

Now we can simply perform the calibration with the ``calibrate()`` method:

.. code-block::

     _ = kinematic_calibrator.calibrate(loss_definition=loss_definition, device=device)

The ``calibrate()`` method returns the loss per heliostat as a flattened tensor, which may be useful for logging or
analysis.


What Happens in Calibration?
----------------------------

To understand calibration, lets look at a small example based on this tutorial. We were to consider a scenario with
three heliostats: ``AA31``, ``AA39``, and ``AA43``. When we perform raytracing using these three heliostats, we get the
following flux images on the target:

.. list-table:: Heliostat bitmaps before calibration
   :widths: 33 33 33
   :header-rows: 0

   * - .. figure:: ./images/heliostat_AA31_before_calibration.png
          :scale: 32%

     - .. figure:: ./images/heliostat_AA39_before_calibration.png
          :scale: 32%

     - .. figure:: ./images/heliostat_AC43_before_calibration.png
          :scale: 32%

If we look closely, we can see that all of these focal spots are off-center, i.e. the centroid or center of mass of each
focal spot is not in the middle of the target.

However, after calibration, if we again perform raytracing we get the following images:

.. list-table:: Heliostat bitmaps after calibration
   :widths: 33 33 33
   :header-rows: 0

   * - .. figure:: ./images/heliostat_AA31_after_calibration.png
          :scale: 32%

     - .. figure:: ./images/heliostat_AA39_after_calibration.png
          :scale: 32%

     - .. figure:: ./images/heliostat_AC43_after_calibration.png
          :scale: 32%

Whilst the changes are small - the focal spots are now clearly centered in the target. Therefore, we can now consider our
heliostats to be calibrated - and that is all there is to kinematic calibration in ``ARTIST``!

.. note::

    The images generated in this tutorial are for illustrative purposes, often with reduced resolution and without
    hyperparameter optimization. Therefore, they should not be taken as a measure of the quality of ``ARTIST``. Please
    see our publications for further information.
