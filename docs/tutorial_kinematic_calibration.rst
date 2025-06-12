.. _tutorial_kinematic_calibration:

``ARTIST`` Tutorial: Kinematic Calibration
==========================================

.. note::

    You can find the corresponding ``Python`` script for this tutorial here:
    https://github.com/ARTIST-Association/ARTIST/blob/main/tutorials/02_kinematic_optimization_tutorial.py

This tutorial provides a brief introduction to ``ARTIST`` showcasing how the kinematic calibration is performed.
The tutorial will run through some basic concepts necessary to understanding ``ARTIST`` including:

- Why do we need to calibrate the kinematic.
- How to load calibration data.
- How to set up the ``KinematicOptimizer`` responsible for the kinematic calibration.

It is best if you already know about the following processes in ``ARTIST``

- How to load a scenario.
- Aligning heliostats before ray tracing.
- Performing heliostat ray tracing to generate a flux density image on the receiver.

If you need help with this look into our other tutorials such as the tutorial on :ref:`heliostat raytracing <tutorial_heliostat_raytracing>`.

Kinematic Calibration Basics
----------------------------
In the real world most components of the kinematic have mechanical errors. This means if we tell an actuator to orient
a heliostat along a specific angle, the heliostat might not end up pointing exactly at the specified aim point.
In ``ARTIST`` we create a digital twin of a solar tower power plant. In the computer simulation, the heliostat will, per default,
point exactly where we tell it to. To keep the predictions made with ``ARTIST`` as accurate as possible we need to
consider the mechanical errors and offsets of the real-world kinematic. In the kinematic calibration process the kinematic module
learns all offset or deviation parameters of the real-world kinematic, to mimic its behavior.
Calibrating the kinematic in ``ARTIST`` requires calibration data. The flux density distributions used as input to the calibration
can be gained by pointing single heliostats at calibration targets.

Loading the Calibration Data
----------------------------
Calibration data can be accessed from the ``PAINT`` database: https://paint-database.org/.
Multiple heliostats can be calibrated at once and each heliostat can be calibrated with multiple calibration data points at once.
To load the calibration data into the ``ARTIST`` environment we use a heliostat to calibration file mapping and the ``paint_loader``.

The mapping from heliostat to calibration files should follow this pattern:

.. code-block::

    # Please follow the following style: list[tuple[str, list[pathlib.Path]]]
    heliostat_calibration_mapping = [
        (
            "name1",
            [
                pathlib.Path(
                    "please/insert/the/path/to/the/paint/data/here/calibration-properties.json"
                ),
                # pathlib.Path(
                #     "please/insert/the/path/to/the/paint/data/here/calibration-properties.json"
                # ),
                # ....
            ],
        ),
        (
            "name2",
            [
                pathlib.Path(
                    "please/insert/the/path/to/the/paint/data/here/calibration-properties.json"
                ),
            ],
        ),
        # ...
    ]

In this mapping the first heliostat would have two calibration files and the second heliostat would have one.
You can specify as many as you want for each helisotat. The data is loaded with the ``paint_loader`` like this:

.. code-block::

    # Load the calibration data.
    (
        focal_spots_calibration,
        incident_ray_directions_calibration,
        motor_positions_calibration,
        heliostats_mask_calibration,
        target_area_mask_calibration,
    ) = paint_loader.extract_paint_calibration_data(
        heliostat_calibration_mapping=[
            (heliostat_name, paths)
            for heliostat_name, paths in heliostat_calibration_mapping
            if heliostat_name in heliostat_group.names
        ],
        heliostat_names=heliostat_group.names,
        target_area_names=scenario.target_areas.names,
        power_plant_position=scenario.power_plant_position,
        device=device,
    )

Now we have all the calibration data we need to calibrate the kinematic of the provided scenario.

For this kinematic type there are altogether 28 optimizable parameters.
18 parameters are kinematic deviation parameters, and then there are 5 actuator parameters for each actuator.
You can select all of them with the following code:

.. code-block::

    # Select the kinematic parameters to be optimized and calibrated.
    optimizable_parameters = [
        heliostat_group.kinematic_deviation_parameters.requires_grad_(),
        heliostat_group.actuator_parameters.requires_grad_(),
    ]

Setting up the ``KinematicOptimizer``
-------------------------------------
The kinematic optimizer object is responsible for the kinematic calibration. We define the kinematic optimizer by
creating an ``KinematicOptimizer`` object as shown below:

.. code-block::

    # Create the kinematic optimizer.
    kinematic_optimizer = KinematicOptimizer(
        scenario=scenario,
        calibration_group=calibration_group,
        optimizer=optimizer,
    )

This object defines the following kinematic optimizer properties:

- The ``scenario`` provides all of the environment variables.
- The ``calibration_group`` contains all (replicated) heliostats.
- The ``optimizer`` is a ``torch.optim.Optimizer`` like ``torch.optim.Adam`` that contains the optimizable parameters.

Optimizing the parameters
-------------------------
The set up is now complete and the kinematic calibration can begin. The kinematic calibration is an optimization process.
We start the optimization process by calling:

.. code-block::

    # Calibrate the kinematic.
    kinematic_optimizer.optimize(
        focal_spots_calibration=focal_spots_calibration,
        incident_ray_directions=incident_ray_directions_calibration,
        active_heliostats_mask=heliostats_mask_calibration,
        target_area_mask_calibration=target_area_mask_calibration,
        motor_positions_calibration=motor_positions_calibration,
        tolerance=tolerance,
        max_epoch=max_epoch,
        num_log=max_epoch,
        device=device,
    )


Currently there are two methods to calibrate the kinematic. Either we use geometric considerations and the
motor positions from the calibration data or we optimize using flux density distributions and the differentiable
ray tracer. Choosing the optimization method depends on the available calibration data.
Both methods need information about:

- The centers of the measured flux density distributions,
- The incident ray directions during the measurements,

The calibration via the ``motor_positions`` additionally needs information about the motor positions
that were measured during the calibration data acquisition. The ``motor_positions`` is an optional
parameter in the ``optimize()`` function above. Since we included them here, the calibration happens via the motor positions.

Optimization methods
--------------------
Here is the workflow of the kinematic calibration with motor positions:

- We start with default values for all optimizable parameters.
- We calculate the preferred reflection direction of our heliostat through knowledge about the
  center of the calibration flux density distribution.
- In the optimization loop we calculate the current orientation of the heliostat from the motor positions,
  then we calculate the actual reflection direction of the heliostat. The loss is defined by the
  difference between the actual reflection direction and the preferred reflection direction from the calibration data.
- The optimizer updates the optimizable parameters until it is accurate enough or the maximum number of epochs is reached.

Here is the workflow of the kinematic calibration with the differentiable ray tracer.

- We start with default values for all optimizable parameters.
- In the optimization loop we align the heliostat by providing the incident ray direction of the calibration data.
  Then we create the heliostat ray tracer by specifying the used calibration target instead of the receiver. We trace the rays
  and create a bitmap of the flux density distribution. From this distribution we calculate the center. The loss is defined as the
  difference between the actual center from the ray traced distribution and the center of the calibration data.
- The optimizer updates the optimizable parameters until it is accurate enough or the maximum number of epochs is reached.
