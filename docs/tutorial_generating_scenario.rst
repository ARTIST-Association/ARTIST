.. _tutorial_generating_scenario:

``ARTIST`` Tutorial: Generating a Scenario HDF5 File
====================================================

.. note::

    You can find the corresponding ``Python`` scripts for this tutorial here:
    https://github.com/ARTIST-Association/ARTIST/blob/main/tutorials/00_generate_scenario_from_stral.py
    https://github.com/ARTIST-Association/ARTIST/blob/main/tutorials/00_generate_scenario_from_paint.py

In this tutorial, we will guide you through the process of generating simple ``ARTIST`` scenario HDF5 files. Before
starting the tutorial, make sure you have read the information regarding the structure of an ``ARTIST`` scenario file
:ref:`that you can find here <scenario>`!

As mentioned in the :ref:`information on a scenario <scenario>`, an ``ARTIST`` scenario consists of five main elements:

- One power plant location
- At least one (but possibly more) target areas.
- At least one (but possibly more) light sources.
- At least one (but usually more) heliostats.
- A prototype which is used if the individual heliostats do not have individual parameters.

In this tutorial, we will develop simple ``ARTIST`` scenarios that contain:

- A power plant location
- One or more target areas.
- One ``Sun`` as a light source.
- One or more heliostats.
- A corresponding prototype to define the properties of the heliostat.

Before we start defining the scenario, we need to determine where it will be saved. We define this by setting the
``scenario_path`` variable. If this location does not exist, the scenario generation will automatically fail.

.. code-block::

    # Specify the path to your scenario file.
    scenario_path = pathlib.Path("please/insert/the/path/to/the/scenario/here/name")

After this step, the process differs depending on which data source you are using. Data from different data sources
may have to be handled differently to create scenarios in ``ARTIST``. Depending on the structure of the input data,
different functions will need to be called. This tutorial shows how to convert ``STRAL`` and ``PAINT`` data to create
usable scenarios for ``ARTIST``.

We will first look at using ``PAINT`` data to generate a scenario. If you are only interested in ``STRAL`` please
jump to :ref:`stral`.

.. _paint:

Generating a Scenario With ``PAINT`` Data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For generating a scenario from ``PAINT`` specify the following files (If you want to set up a scenario with multiple
heliostats, simply add more files for each heliostat):

- One ``tower-measurement.json`` file
- One or more ``heliostat-properties.json`` file
- One or more ``deflectometry.h5`` file

.. code-block::

    # Specify the path to your tower-measurements.json file.
    tower_file = pathlib.Path(
        "please/insert/the/path/to/the/tower/measurements/here/tower-measurements.json"
    )

    # Specify the following data for each heliostat that you want to include in the scenario:
    # A tuple of: (heliostat-name, heliostat-properties.json, deflectometry.h5)
    heliostat_files_list = [
        (
            "name1",
            pathlib.Path(
                "please/insert/the/path/to/the/heliostat/properties/here/heliostat_properties.json"
            ),
            pathlib.Path(
                "please/insert/the/path/to/the/deflectometry/data/here/deflectometry.h5"
            ),
        ),
        (
            "name2",
            pathlib.Path(
                "please/insert/the/path/to/the/heliostat/properties/here/heliostat_properties.json"
            ),
            pathlib.Path(
                "please/insert/the/path/to/the/deflectometry/data/here/deflectometry.h5"
            ),
        ),
        # ... Include as many as you want, but at least one!
    ]

Now we will get into building the scenario.

.. _plant_and_target:

Power Plant and Target Areas
----------------------------
The location of the power plant as well as information on the target areas (i.e., the receiver or calibration targets) is
loaded simultaneously in paint form the ``tower-measurement.json`` file.

We can load this information using functions from the ``paint_scenario_parser``. In this case, the function below will return
an instance of the the ``PowerPlantConfig`` class as well as an instance of the ``TowerAreaListConfig`` containing a
list of viable target areas.

.. code-block::

    # Include the power plant and target area configuration.
    power_plant_config, target_area_list_config = (
        paint_scenario_parser.extract_paint_tower_measurements(
            tower_measurements_path=tower_file, device=device
        )
    )

The ``PowerPlantConfig`` contains the following attributes:

- The ``power_plant_position`` indicating the power plants location.

The ``TargetAreaListConfig`` contains a list of multiple ``TargetAreaConfig`` objects, which each define the
following attributes:

- A ``target_area_key`` used to identify the target area when loading the ``ARTIST`` scenario.
  This one is a receiver.
- The ``geometry`` currently modelled – in this case a planar target area.
- The ``center`` which defines the position of the target areas's middle. Note that because this is a position
  tensor, the final element of the tensor in the 4D representation is a 1 – for more information see
  :ref:`our note on coordinates <artist_under_hood>`.
- A ``normal_vector`` defining the normal vector to the plane of the target area. Note that because this is a direction
  tensor, the final element of the tensor in the 4D representation is a 0 – for more information see
  :ref:`our note on coordinates <artist_under_hood>`.
- The ``plane_e`` which defines the target area plane in the east direction.
- The ``plane_u`` which defines the target area plane in the up direction.

.. _light_source:

Light Source
------------
The light source is the object responsible for providing light that is then reflected by the heliostats. Typically, this
light source is a ``Sun``, however in certain situations it may be beneficial to model multiple artificial light
sources. Light source information are not included in any files, you have to define them by yourself.
We define the light source by creating a ``LightSourceConfig`` object as shown below:

.. code-block::

    # Include the light source configuration.
    light_source1_config = LightSourceConfig(
        light_source_key="sun_1",
        light_source_type=config_dictionary.sun_key,
        number_of_rays=10,
        distribution_type=config_dictionary.light_source_distribution_is_normal,
        mean=0.0,
        covariance=4.3681e-06,
    )

This configuration defines the following light source properties:

- The ``light_source_key`` used to identify the light source when loading the ``ARTIST`` scenario.
- The ``light_source_type`` which defines what type of light source is used. In this case, it is a ``Sun``.
- The ``number_of_rays`` which defines how many rays are sampled from the light source for ray tracing.
- The ``distribution_type`` which models what distribution is used to model the light source. In this case, we use a
  normal distribution.
- The ``mean`` and the ``covariance`` which are the parameters of the previously defined normal distribution used to
  model the light source.

Since our scenario only contains one light source but ``ARTIST`` scenarios are designed to load multiple light sources,
we have to wrap our light source in a list and create a ``LightSourceListConfig`` object:

.. code-block::

    # Create a list of light source configs - in this case only one.
    light_source_list = [light_source1_config]

    # Include the configuration for the list of light sources.
    light_source_list_config = LightSourceListConfig(light_source_list=light_source_list)


Prototypes and Heliostats
-------------------------
``ARTIST`` always requires prototypes and heliostats - see :ref:`our tutorial here <scenario>` for more information.

The prototypes and list of heliostats can be easily extracted using the ``paint_scenario_parser``. Here it important to define one
target area from the list of possible target areas as the default aim point. In this case we use the receiver for this,
as shown below:

.. code-block::

    target_area = [
        target_area
        for target_area in target_area_list_config.target_area_list
        if target_area.target_area_key == config_dictionary.target_area_receiver
    ]

Now, before we load the heliostats we need to do some configuration. ``ARTIST`` internally models all surfaces with
:ref:`NURBS <nurbs>`, which are learnt when loading the data. Therefore, we have to set certain parameters, such as the
number of control points, the fit tolerance, the number of epochs to train for, etc. We also need to configure an optimizer
for the training process and a learning rate scheduler. This is shown below:

.. code-block::

    number_of_nurbs_control_points = torch.tensor([20, 20], device=device)
    nurbs_fit_method = config_dictionary.fit_nurbs_from_normals
    nurbs_deflectometry_step_size = 100
    nurbs_fit_tolerance = 1e-10
    nurbs_fit_max_epoch = 400

    # Please leave the optimizable parameters empty, they will automatically be added for the surface fit.
    nurbs_fit_optimizer = torch.optim.Adam([torch.empty(1, requires_grad=True)], lr=1e-3)
    nurbs_fit_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        nurbs_fit_optimizer,
        mode="min",
        factor=0.2,
        patience=50,
        threshold=1e-7,
        threshold_mode="abs",
    )

Then, with a single function we can load the heliostat list configuration, learn the surfaces, and generate the
prototype configuration.

.. code-block::

    heliostat_list_config, prototype_config = (
        paint_scenario_parser.extract_paint_heliostats_fitted_surface(
            paths=heliostat_files_list,
            power_plant_position=power_plant_config.power_plant_position,
            number_of_nurbs_control_points=number_of_nurbs_control_points,
            deflectometry_step_size=nurbs_deflectometry_step_size,
            nurbs_fit_method=nurbs_fit_method,
            nurbs_fit_tolerance=nurbs_fit_tolerance,
            nurbs_fit_max_epoch=nurbs_fit_max_epoch,
            nurbs_fit_optimizer=nurbs_fit_optimizer,
            nurbs_fit_scheduler=nurbs_fit_scheduler,
            device=device,
        )
    )

The ``heliostat_list_config`` is a list of ``HeliostatConfig`` objects which includes the following information:

- The ``name`` used to identify the heliostat.
- The numerical ``id`` of the heliostat.
- The ``position`` of the heliostat.
- The configuration for the ``surface`` of the heliostat (see :py:class:`artist.scenario.configuration_classes.SurfaceConfig`).
- The configuration for the ``kinematic`` of the heliostat (see :py:class:`artist.scenario.configuration_classes.KinematicConfig`).
- A list of configurations for the ``actuators`` required by the heliostat (see :py:class:`artist.scenario.configuration_classes.ActuatorConfig`).

The ``prototype_config`` is a ``PrototypeConfig`` object, containing information on:

- The ``surface_prototype`` used in the scenario, for heliostats without individual surface configurations (see :py:class:`artist.scenario.configuration_classes.SurfacePrototypeConfig`).
- The ``kinematic_prototype`` used in the scenario, for heliostats without individual kinematic configurations (see :py:class:`artist.scenario.configuration_classes.KinematicPrototypeConfig`).
- A list of ``actuators_prototype`` used in the scenario, for heliostats without individual actuator configurations (see :py:class:`artist.scenario.configuration_classes.ActuatorPrototypeConfig`).

Different Surface Options
~~~~~~~~~~~~~~~~~~~~~~~~~

``ARTIST`` does not require deflectometry data to generate a scenario. It is also possible to generate a
scenario with an *ideal* surface. The true surface can then either be learnt via raytracing
(see :ref:`the NURBS surface reconstructor<tutorial_surface_reconstruction>`), or if not information on the true surface
is available an ideal surface can also be applied. To generate heliostats with ideal surface you call the function:

.. code-block::

    heliostat_list_config, prototype_config = (
            paint_scenario_parser.extract_paint_heliostats_ideal_surface(
                paths=heliostat_files_list,
                power_plant_position=power_plant_config.power_plant_position,
                device=device,
            )
        )

It is also not necessary to define and optimizer in this setting.

It is also possible to generate scenarios containing both fitted and ideal surfaces with the function ``extract_paint_heliostats_mixed_surface()``.
In this case, the type of surface created depends on the mapping derived above. More specifically, if you provide a path
to a deflectometry file, then the surface will be fitted, if not, then an ideal surface will be generated.
For example, for the following mapping:

.. code-block::

    heliostat_files_list = [
        (
            "heliostat_1",
            pathlib.Path(
                "please/insert/the/path/to/the/heliostat/properties/here/heliostat_properties.json"
            ),
            pathlib.Path(
                "please/insert/the/path/to/the/deflectometry/data/here/deflectometry.h5"
            ),
        ),
        (
            "heliostat_2",
            pathlib.Path(
                "please/insert/the/path/to/the/heliostat/properties/here/heliostat_properties.json"
            ),
        ),
    ]

Calling the function:

.. code-block::

    heliostat_list_config, prototype_config = (
        paint_scenario_parser.extract_paint_heliostats_mixed_surface(
            paths=heliostat_files_list,
            power_plant_position=power_plant_config.power_plant_position,
            number_of_nurbs_control_points=number_of_nurbs_control_points,
            deflectometry_step_size=nurbs_deflectometry_step_size,
            nurbs_fit_method=nurbs_fit_method,
            nurbs_fit_tolerance=nurbs_fit_tolerance,
            nurbs_fit_max_epoch=nurbs_fit_max_epoch,
            nurbs_fit_optimizer=nurbs_fit_optimizer,
            nurbs_fit_scheduler=nurbs_fit_scheduler,
            device=device,
        )
    )

will generate a scenario where "heliostat_1" has a fitted surface and "heliostat_2" has an ideal surface.

**NOTE:** In this situation, the prototype will always be an ideal surface.


.. _create_hdf5:

Creating the HDF5 File
----------------------

Now we have all the required information to generate the HDF5 and finish the scenario. We can generate this scenario by
running the ``main`` function shown below:

.. code-block::

    if __name__ == "__main__":
        """Generate the scenario given the defined parameters."""
        scenario_generator = ScenarioGenerator(
            file_path=scenario_path,
            power_plant_config=power_plant_config,
            target_area_list_config=target_area_list_config,
            light_source_list_config=light_source_list_config,
            prototype_config=prototype_config,
            heliostat_list_config=heliostats_list_config,
        )
        scenario_generator.generate_scenario()

This ``main`` function initially defines the ``ScenarioGenerator`` object based on the previously defined ``scenario_path``
and our configurations for the receiver(s), light source(s), prototype, and heliostat(s).

If you go to the location you defined at the very start you should now see a HDF5 file there -- and that is all there is
to generating a scenario in ``ARTIST``!

.. _stral:

Generating a Scenario with ``STRAL`` Data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To generate a scenario from ``STRAL``, you only need a single ``.binp`` file.

.. code-block::

    # Specify the path to your stral_data.binp file.
    stral_file_path = pathlib.Path(
        "please/insert/the/path/to/the/stral/data/here/stral_data.binp"
    )

Many of the steps required to generate the scenario are very similar to before, but there are some changes.

Power Plant
-----------
``STRAL`` data contains no information about the power plant position, so you have to enter the
coordinates manually, as shown below:

.. code-block::

    # Include the power plant configuration.
    power_plant_config = PowerPlantConfig(
      power_plant_position=torch.tensor([0.0, 0.0, 0.0], device=device)
    )

Information on the ``PowerPlantConfig`` class is provided above (see :ref:`plant_and_target`).

Target Areas
------------
We also need to manually define the ``TargetAreaConfig`` when using ``STRAL``:

.. code-block::

    # STRAL
    # Include a single tower area (receiver)
    receiver_config = TargetAreaConfig(
        target_area_key="receiver",
        geometry=config_dictionary.target_area_type_planar,
        center=torch.tensor([0.0, -50.0, 0.0, 1.0], device=device),
        normal_vector=torch.tensor([0.0, 1.0, 0.0, 0.0], device=device),
        plane_e=8.629666667,
        plane_u=7.0,
    )

Information on the ``TargetAreaConfig`` class is provided above (see :ref:`plant_and_target`).

Since our scenario only contains one target area (a receiver) but ``ARTIST`` scenarios are designed to load multiple
target areas, when using ``STRAL`` we have to manually wrap our target area in a list and create a
``TargetAreaListConfig`` object:

.. code-block::

    # Create list of target area configs - in this case only one.
    target_area_config_list = [receiver_config]

    # Include the tower area configurations.
    target_area_list_config = TargetAreaListConfig(target_area_config_list)

Light Source
------------
Generating a light source when using ``STRAL`` data is identical to ``PAINT`` data, please see: :ref:`light_source`.

Prototypes
----------
In ``STRAL`` prototypes need to be defined manually. A prototype always contains a surface prototype, a kinematic
prototype, and an actuator prototype.

We start with the surface prototype. We first need to extract information regarding the facet translation vectors, the
canting, and the surface points and normals from ``STRAL`` with the following code:

.. code-block::

    (
        facet_translation_vectors,
        canting,
        surface_points_with_facets_list,
        surface_normals_with_facets_list,
    ) = stral_scenario_parser.extract_stral_deflectometry_data(
        stral_file_path=stral_file_path, device=device
    )

Before we can generate a NURBS surface based on the surface normals and points from ``STRAL`` we need to define the surface
generator and the optimizer and scheduler to fit the surface:

.. code-block::

    surface_generator = SurfaceGenerator(device=device)

    # Please leave the optimizable parameters empty, they will automatically be added for the surface fit.
    nurbs_fit_optimizer = torch.optim.Adam([torch.empty(1, requires_grad=True)], lr=1e-3)
    nurbs_fit_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        nurbs_fit_optimizer,
        mode="min",
        factor=0.2,
        patience=50,
        threshold=1e-7,
        threshold_mode="abs",
    )

Finally, we can use the configuration to generate a fitted surface:

.. code-block::

    surface_config = surface_generator.generate_fitted_surface_config(
        heliostat_name="heliostat_1",
        facet_translation_vectors=facet_translation_vectors,
        canting=canting,
        surface_points_with_facets_list=surface_points_with_facets_list,
        surface_normals_with_facets_list=surface_normals_with_facets_list,
        optimizer=nurbs_fit_optimizer,
        scheduler=nurbs_fit_scheduler,
        device=device,
    )

Alternatively, we can also generate an ideal surface that is not fitted based on defelectometry data. To generate this
surface you don't need to define an optimizer or scheduler, but can simply call:

.. code-block::

     surface_config = surface_generator.generate_ideal_surface_config(
        facet_translation_vectors=facet_translation_vectors,
        canting=canting,
        device=device,
    )

To generate the surface configuration, we simply define a surface configuration prototype based on the list of facets
contained in the `SurfaceConfig` object created above:

.. code-block::

    surface_prototype_config = SurfacePrototypeConfig(facet_list=surface_config.facet_list)

The next prototype object we consider is the kinematic prototype. The kinematic modeled in ``ARTIST`` assumes that
all heliostats are initially pointing in the south direction; however, depending on the CSP considered, the heliostats may
initially be orientated in a different direction. For our scenario, we want the heliostats to initially be orientated upwards,
i.e., they point directly at the sky. A further element of a kinematic configuration is ``KinematicDeviations`` which are small
disturbance parameters to represent offsets caused by the two-joint kinematic modeled in ``ARTIST``. However, in this tutorial
we ignore these deviations. Therefore, we can now create the kinematic prototype by generating a ``KinematicPrototypeConfig`` object:

.. code-block::

    kinematic_prototype_config = KinematicPrototypeConfig(
        type=config_dictionary.rigid_body_key,
        initial_orientation=torch.tensor([0.0, 0.0, 1.0, 0.0], device=device),
    )

This object defines:

- The ``type`` applied in the scenario; in this case, we are using a rigid body kinematic.
- The ``initial_orientation`` which is the direction we defined above.
- If we have ``KinematicDeviations``, we would also include them in this definition.

With the kinematic prototype defined, the final prototype we require is the actuator prototype. For the rigid body
kinematic applied in this scenario, we require exactly two actuators. These actuators require min and max motor positions
which are not included in the ``STRAL`` data, therefore we have to define them manually. Here we use the min amd max motor
positions that are relevant for Jülich

.. code-block::

    min_max_motor_positions_actuator_1 = [0.0, 60000.0]
    min_max_motor_positions_actuator_2 = [0.0, 80000.0]

We can now define these actuators with ``ActuatorConfig`` objects as shown below:

.. code-block::

    # Include an ideal actuator.
    actuator1_prototype = ActuatorConfig(
        key="actuator_1",
        type=config_dictionary.ideal_actuator_key,
        clockwise_axis_movement=False,
        min_max_motor_positions=min_max_motor_positions_actuator_1,
    )

    # Include an ideal actuator.
    actuator2_prototype = ActuatorConfig(
        key="actuator_2",
        type=config_dictionary.ideal_actuator_key,
        clockwise_axis_movement=True,
        min_max_motor_positions=min_max_motor_positions_actuator_2,
    )

These configurations define:

- The ``key`` used when loading the actuator from an ``ARTIST`` scenario.
- The ``type`` which in this case is an ideal actuator for both actuators.
- The ``clockwise_axis_movement`` parameter which defines if the actuator operates per default in a clockwise or
  counter-clockwise direction.

If we were considering different types of actuators, e.g., a linear actuator, we would also have to define specific
actuator parameters – however we will stick to a simple configuration for this tutorial. To complete the actuator
prototype, we need to wrap both actuators in a list and generate an ``ActuatorPrototypeConfig`` object:

.. code-block::

    # Create a list of actuators.
    actuator_prototype_list = [actuator1_prototype, actuator2_prototype]

    # Include the actuator prototype config.
    actuator_prototype_config = ActuatorPrototypeConfig(
        actuator_list=actuator_prototype_list
    )

Now that all the aspects of our prototype are defined, we can create the final ``PrototypeConfig`` object, which simply
combines all the above configurations into one object, as shown below:

.. code-block::

    # Include the final prototype config.
    prototype_config = PrototypeConfig(
        surface_prototype=surface_prototype_config,
        kinematic_prototype=kinematic_prototype_config,
        actuator_prototype=actuator_prototype_config,
    )

Heliostat from ``STRAL``
------------------------
Having defined the prototype we can now define our heliostat by creating a ``HeliostatConfig`` object as shown below:

.. code-block::

    # Include the configuration for a heliostat.
    heliostat1 = HeliostatConfig(
        name="heliostat_1",
        id=1,
        position=torch.tensor([0.0, 5.0, 0.0, 1.0], device=device),
    )

This heliostat configuration requires:

- A ``name`` used to identify the heliostat when loading the ``ARTIST`` scenario.
- The ``id``, a unique identifier that can be used to quickly identify the heliostat within the scenario.
- The ``position`` which defines the position of the heliostat in the field. Note the one in the fourth
  dimension according to the previously discussed :ref:'coordinate convention <coordinates>'.

In this setting, the heliostat does not have any individual surface, kinematic, or actuator parameters, and will
automatically use the parameters defined in the prototype. However, since ``ARTIST`` is designed to load multiple
heliostats, we do need to wrap our heliostat configuration in a list and create a ``HeliostatListConfig`` object as shown below:

.. code-block::

    heliostat_list = [heliostat1]

    # Create the configuration for all heliostats.
    heliostats_list_config = HeliostatListConfig(heliostat_list=heliostat_list)

If we wanted heliostats with individual measurements, we would have to define the individual surface, kinematic, and
actuator configurations for each heliostat.

Creating the HDF5 File
----------------------
Creating the HDF5 based on ``STRAL`` data is the same process as when using ``PAINT`` data (see :ref:`create_hdf5`).

.. warning::

    When generating a scenario, the logger reports what version of the scenario generator is currently running. Changes
    in versions may result in a scenario that is incompatible with the current ``ARTIST`` version.
