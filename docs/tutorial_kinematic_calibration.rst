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
To load the calibration data into the ``ARTIST`` environment we use the ``paint_loader`` and specify
the ``calibration_properties_path`` by providing a valid path and the ``power_plant_position``, so that coordinates
can be transformed into the local ENU coordinate system used in ``ARTIST``.

.. code-block::

    # Load the calibration data.
    (
        calibration_target_names,
        center_calibration_images,
        sun_positions,
        all_calibration_motor_positions,
    ) = paint_loader.extract_paint_calibration_data(
        calibration_properties_paths=calibration_properties_paths,
        power_plant_position=scenario.power_plant_position,
        device=device,
    )

Now we have all the calibration data we need to calibrate the kinematic of the provided scenario.

Calibration Scenario and Optimizable Parameters
-----------------------------------------------
Currently, only one heliostat can be calibrated at a time. If your scenario contains multiple heliostats, you can
create a calibration scenario with a single heliostat like this:

.. code-block::

    # Create a calibration scenario from the original scenario.
    # It contains a single heliostat, chosen by its index.
    calibration_scenario = scenario.create_calibration_scenario(
        heliostat_index=0, device=device
    )

This heliostat uses a rigid body kinematic. For this kinematic type there are altogether 28 optimizable parameters.
18 parameters are kinematic deviation parameters, and then there are 5 actuator parameters for each actuator.
You can select all of them with the following code:

.. code-block::

    # Select the kinematic parameters to be optimized and calibrated.
    optimizable_parameters = [
        calibration_scenario.heliostat_field.all_kinematic_deviation_parameters.requires_grad_(),
        calibration_scenario.heliostat_field.all_actuator_parameters.requires_grad_(),
    ]

Setting up the ``KinematicOptimizer``
-------------------------------------
The kinematic optimizer object is responsible for the kinematic calibration. We define the kinematic optimizer by
creating an ``KinematicOptimizer`` object as shown below:

.. code-block::

    # Create the kinematic optimizer.
    kinematic_optimizer = KinematicOptimizer(
        scenario=calibration_scenario,
        optimizer=optimizer,
        scheduler=scheduler,
    )

This object defines the following kinematic optimizer properties:

- The ``scenario`` provides all of the environment variables.
- The ``optimizer`` is a ``torch.optim.Optimizer`` like ``torch.optim.Adam`` that contains the optimizable parameters.
- The ``scheduler`` is a ``torch.optim.lr_scheduler`` like ``torch.optim.lr_scheduler.ReduceLROnPlateau``.

Optimizing the parameters
-------------------------
The set up is now complete and the kinematic calibration can begin. The kinematic calibration is an optimization process.
We start the optimization process by calling:

.. code-block::

    # Calibrate the kinematic.
    kinematic_optimizer.optimize(
        tolerance=tolerance,
        max_epoch=max_epoch,
        center_calibration_images=center_calibration_images,
        incident_ray_directions=incident_ray_directions,
        calibration_target_names=calibration_target_names,
        motor_positions=all_calibration_motor_positions,
        num_log=max_epoch,
        device=device,
    )

Currently there are two methods to calibrate the kinematic. Either we use geometric considerations and the
motor positions from the calibration data or we optimize using flux density distributions and the differentiable
ray tracer. The kinematic calibration via the motor position is generally faster. However, choosing the optimization
method depends on the available calibration data. Both methods need information about:

- The center of the measured flux density distribution,
- The incident ray direction during the measurement,

The faster calibration via the ``motor_positions`` additionally needs information about the motor positions
that were measured during the data acquisition. The ``motor_positions`` is an optional parameter in the ``optimize()``
function above. Since we included them here, the calibration happens via the motor positions.

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
