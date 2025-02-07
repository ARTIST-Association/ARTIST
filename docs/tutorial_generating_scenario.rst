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
Data from different data sources may have to be handled differently to create scenarios in ``ARTIST``. Depending on the
structure of the input data, different functions will need to be called. This tutorial shows how to convert ``STRAL`` and
``PAINT`` data to create usable scenarios for ``ARTIST``.

Before we start defining the scenario, we need to determine where it will be saved. We define this by setting the
``scenario_path`` variable. If this location does not exist, the scenario generation will automatically fail.

.. code-block::

    # The following parameter is the name of the scenario.
    scenario_path = "please/insert/your/path/here/name"

Additionally we have to specify where our ``STRAL`` or ``PAINT`` input files are saved.
While ``STRAL`` saves everything in one file, ``PAINT`` data is split into several different files.
The code below shows, that creating a scenario from ``STRAL`` requires a ``.binp`` file.

.. code-block::

    # STRAL
    # Specify the path to your stral_data.binp file.
    stral_file_path = pathlib.Path(
        "please/insert/the/path/to/the/strahl/data/here/stral_data.binp"
    )

For generating a scenario from ``PAINT`` specify the following files (If you want to set up a scenario with multiple
heliostats, simply add more files for each heliostat):

- One ``tower-measurement.json`` file
- One or more ``heliostat-properties.json`` file
- One or more ``deflectometry.h5`` file

.. code-block::

    # PAINT
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

Now we can get started defining each of these elements and then generating the scenario!
Differences between the handling of ``STRAL`` and ``PAINT`` files will be highlighted.

Power Plant
-----------
This is where the power plant location is saved in WGS84 latitude, longitude, altitude coordinates.
Since ``STRAL`` files do not contain information about the power plant position, we have to enter the
coordinates manually, as shown below:

.. code-block::

    # STRAL
    # Include the power plant configuration.
    power_plant_config = PowerPlantConfig(
      power_plant_position=torch.tensor([0.0, 0.0, 0.0], device=device)
    )

``PAINT`` stores information about the power plant coordinates in ``tower-measurement.json`` files. In
general, when using ``PAINT`` as data source, you can use functions from the ``paint_loader`` to retrieve the data.
We can define the location of the power plant with the ``PowerPlantConfig`` class as shown below, note
that the ``tower-measurement.json`` files also contain information about each target area on the solar
tower(s). The ``paint_loader`` retrieves this inforamtion together with the ``PowerPlantConfig``.

.. code-block::

    # PAINT
    # Include the power plant configuration.
    power_plant_config, target_area_list_config = (
        paint_loader.extract_paint_tower_measurements(
            tower_measurements_path=tower_file, device=device
        )
    )

Both configurations define the following properties:

- The ``power_plant_position`` indicating the power plants location.

Target Areas
------------
The target areas are located on the solar tower, it is where the reflected light from the heliostats is concentrated.
We can define a target area with the ``TargetAreaConfig`` class as shown below. Since we already retrieved this config
for the ``PAINT`` data above, the following only shows manually creating ``TaTargetAreaConfig`` for the use in
``STRAL``-scenario-generators:

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

This configuration defines the following properties:

- A ``target_area_key`` used to identify the target area when loading the ``ARTIST`` scenario.
  This one is a receiver.
- The ``geometry`` currently modelled – in this case a planar target area.
- The ``center`` which defines the position of the target areas's middle. Note that because this is a position
  tensor, the final element of the tensor in the 4D representation is a 1 – for more information see
  :ref:`our note on coordinates <coordinates>`.
- A ``normal_vector`` defining the normal vector to the plane of the target area. Note that because this is a direction
  tensor, the final element of the tensor in the 4D representation is a 0 – for more information see
  :ref:`our note on coordinates <coordinates>`.
- The ``plane_e`` which defines the target area plane in the east direction.
- The ``plane_u`` which defines the target area plane in the up direction.

Since our scenario only contains one target area (a receiver) but ``ARTIST`` scenarios are designed to load multiple
target areas, we have to wrap our target area in a list and create a ``TargetAreaListConfig`` object:

.. code-block::

    # Create list of target area configs - in this case only one.
    target_area_config_list = [receiver_config]

    # Include the tower area configurations.
    target_area_list_config = TargetAreaListConfig(target_area_config_list)

As mentioned above this is all done automatically, when using ``PAINT`` data and the ``paint_loader```.

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
        number_of_rays=200,
        distribution_type=config_dictionary.light_source_distribution_is_normal,
        mean=0.0,
        covariance=4.3681e-06,
    )

This configuration defines the following light source properties:

- The ``light_source_key`` used to identify the light source when loading the ``ARTIST`` scenario.
- The ``light_source_type`` which defines what type of light source is used. In this case, it is a ``Sun``.
- The ``number_of_rays`` which defines how many rays are sampled from the light source for raytracing.
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


Prototypes from ``STRAL``
-------------------------
You can skip this part if you are only interested in the ``PAINT`` data source.
The next step in defining our scenario is to define our *prototype*. We define the prototype before defining the
heliostat, since in this tutorial we load the heliostat based on the prototype parameters. A prototype always contains
a *surface* prototype, a *kinematic* prototype, and an *actuator* prototype.

We start with the *surface* prototype. In this case, we generate the surface based on a STRAL scenario using a
``SurfaceConverter`` as defined below:

.. code-block::

    # STRAL
    # Generate surface configuration from STRAL data.
    surface_converter = SurfaceConverter(
        max_epoch=400,
    )

This converter can be initialized with default values but we reduce ``max_epoch`` by setting:

- ``max_epoch`` which specifies the maximum number of epochs for the NURBS facet learning.

A surface consists of multiple facets. Since we are using data from STRAL to recreate the surface for our prototype, we
can create this list of facets by calling the ``generate_surface_config_from_stral()`` function as shown below:

.. code-block::

    # STRAL
    facet_prototype_list = surface_converter.generate_surface_config_from_stral(
        stral_file_path=stral_file_path, device=device
    )

This function loads data from STRAL and then uses this data to learn a Non-Rational Uniform B-Spline (NURBS) surface
for each of the facets. Therefore, this function requires:

- The ``stral_file_path`` specifying where the STRAL binary data is saved.

The output of this function is a list of ``FacetConfig`` objects, which define the parameters that enable ``ARTIST`` to
recreate the learned NURBS facet surfaces when the scenario is loaded.

Now that the facet list has been created automatically by learning NURBS from STRAL data, we need to generate a
``SurfacePrototypeConfig`` object to save the surface:

.. code-block::

    # STRAL
    # Generate the surface prototype configuration.
    surface_prototype_config = SurfacePrototypeConfig(facet_list=facet_prototype_list)

The next prototype object we consider is the *kinematic* prototype. The kinematic modeled in ``ARTIST`` assumes that
all heliostats are initially pointing in the south direction; however, depending on the CSP considered, the heliostats may
initially be orientated in a different direction.For our scenario, we want the heliostats to initially be orientated upwards,
i.e., they point directly at the sky. A further element of a kinematic configuration is ``KinematicDeviations`` which are small
disturbance parameters to represent offsets caused by the two-joint kinematic modeled in ``ARTIST``. However, in this tutorial
we ignore these deviations. Therefore, we can now create the kinematic prototype by generating a ``KinematicPrototypeConfig`` object:

.. code-block::

    # STRAL
    # Include the kinematic prototype configuration.
    kinematic_prototype_config = KinematicPrototypeConfig(
        type=config_dictionary.rigid_body_key,
        initial_orientation=torch.tensor([0.0, 0.0, 1.0, 0.0], device=device),
    )

This object defines:

- The ``type`` applied in the scenario; in this case, we are using a *rigid body kinematic*.
- The ``initial_orientation`` which is the direction we defined above.
- If we have ``KinematicDeviations``, we would also include them in this definition.

With the kinematic prototype defined, the final prototype we require is the *actuator* prototype. For the rigid body
kinematic applied in this scenario, we require **exactly two** actuators. We can define these actuators via
``ActuatorConfig`` objects as shown below:

.. code-block::

    # STRAL
    # Include an ideal actuator.
    actuator1_prototype = ActuatorConfig(
        key="actuator_1",
        type=config_dictionary.ideal_actuator_key,
        clockwise_axis_movement=False,
    )

    # Include a second ideal actuator.
    actuator2_prototype = ActuatorConfig(
        key="actuator_2",
        type=config_dictionary.ideal_actuator_key,
        clockwise_axis_movement=True,
    )

These configurations define:

- The ``key`` used when loading the actuator from an ``ARTIST`` scenario.
- The ``type`` which in this case is an ideal actuator for both actuators.
- The ``clockwise_axis_movement`` parameter which defines if the actuator operates per default in a clockwise or
  counter-clockwise direction.

If we were considering different types of actuators, e.g., a *linear actuator*, we would also have to define specific
actuator parameters – however we will stick to a simple configuration for this tutorial. To complete the actuator
prototype, we need to wrap both actuators in a list and generate an ``ActuatorPrototypeConfig`` object:

.. code-block::

    # STRAL
    # Create a list of actuators.
    actuator_prototype_list = [actuator1_prototype, actuator2_prototype]

    # Include the actuator prototype config.
    actuator_prototype_config = ActuatorPrototypeConfig(
        actuator_list=actuator_prototype_list
    )

Now that all the aspects of our prototype are defined, we can create the final ``PrototypeConfig`` object, which simply
combines all the above configurations into one object, as shown below:

.. code-block::

    # STRAL
    # Include the final prototype config.
    prototype_config = PrototypeConfig(
        surface_prototype=surface_prototype_config,
        kinematic_prototype=kinematic_prototype_config,
        actuator_prototype=actuator_prototype_config,
    )

Heliostat from ``STRAL``
------------------------
You can skip even further if you are only interested in the ``PAINT`` data source.
Having defined the prototype we can now define our heliostat by creating a ``HeliostatConfig`` object as shown below:

.. code-block::

    # STRAL
    # Include the configuration for a heliostat.
    heliostat1 = HeliostatConfig(
        name="heliostat_1",
        id=1,
        position=torch.tensor([0.0, 5.0, 0.0, 1.0], device=device),
        aim_point=torch.tensor([0.0, -50.0, 0.0, 1.0], device=device),
    )

This heliostat configuration requires:

- A ``name`` used to identify the heliostat when loading the ``ARTIST`` scenario.
- The ``id``, a unique identifier that can be used to quickly identify the heliostat within the scenario.
- The ``position`` which defines the position of the heliostat in the field. Note the one in the fourth
  dimension according to the previously discussed :ref:'coordinate convention <coordinates>'.
- The ``aim_point`` which defines the desired aim point of the heliostat – in this case the center of
  the receiver target area. Note the one in the fourth dimension according to the previously discussed
  :ref:'coordinate convention <coordinates>'.

Since the heliostat does not have any individual surface, kinematic, or actuator parameters, we do not need to include
them here. However, since ``ARTIST`` is designed to load multiple heliostats, we do need to wrap our heliostat
configuration in a list and create a ``HeliostatListConfig`` object as shown below:

.. code-block::

    # STRAL
    # Create a list of all the heliostats - in this case, only one.
    heliostat_list = [heliostat1]

    # Create the configuration for all heliostats.
    heliostats_list_config = HeliostatListConfig(heliostat_list=heliostat_list)


Prototypes and Heliostats from ``PAINT``
----------------------------------------
This sections shows how to load prototype configurations and heliostat configurations from ``PAINT``
by using the ``paint_loader`` again. Before we can use the ``paint_loader`` to extract everything we need
to choose one target area that is the default aimpoint of our heliostats. Usually it will make sense to
choose a receiver for this, as shown below:

.. code-block::

    # PAINT
    target_area = [
        target_area
        for target_area in target_area_list_config.target_area_list
        if target_area.target_area_key == config_dictionary.target_area_reveicer
    ]

    heliostat_list_config, prototype_config = paint_loader.extract_paint_heliostats(
        heliostat_and_deflectometry_paths=heliostat_files_list,
        power_plant_position=power_plant_config.power_plant_position,
        aim_point=target_area[0].center,
        device=device,
    )


Generate Scenario
-----------------
We have now defined all aspects of our simple scenario. The only step remaining is to generate the scenario. We can
generate this scenario by running the ``main`` function shown below:

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

Running the ``main`` function should produce the following output for ``STRAL`` conversions:

.. code-block::

    [2025-01-21 11:36:15,234][artist.util.surface_converter][INFO] - Beginning extraction of data from ```STRAL``` file.
    [2025-01-21 11:36:15,234][artist.util.surface_converter][INFO] - Reading STRAL file located at: /.../ARTIST/tutorials/data/test_stral_data.binp
    [2025-01-21 11:36:35,280][artist.util.surface_converter][INFO] - Loading ``STRAL`` data complete.
    [2025-01-21 11:36:35,280][artist.util.surface_converter][INFO] - Beginning generation of the surface configuration based on data.
    [2025-01-21 11:36:35,281][artist.util.surface_converter][INFO] - Converting to NURBS surface.
    [2025-01-21 11:36:35,281][artist.util.surface_converter][INFO] - Converting facet 1 of 4.
    [2025-01-21 11:36:37,484][artist.util.surface_converter][INFO] - Epoch: 0, Loss: 0.0022271068301051855, LR: 0.001.
    [2025-01-21 11:37:26,242][artist.util.surface_converter][INFO] - Epoch: 100, Loss: 0.0002696856390684843, LR: 0.001.
    [2025-01-21 11:38:15,108][artist.util.surface_converter][INFO] - Epoch: 200, Loss: 5.375401087803766e-05, LR: 0.0002.
    [2025-01-21 11:38:41,483][artist.util.surface_converter][INFO] - Converting facet 2 of 4.
    [2025-01-21 11:38:42,048][artist.util.surface_converter][INFO] - Epoch: 0, Loss: 0.0023851273581385612, LR: 0.001.
    [2025-01-21 11:39:30,980][artist.util.surface_converter][INFO] - Epoch: 100, Loss: 0.00029010826256126165, LR: 0.001.
    [2025-01-21 11:40:19,777][artist.util.surface_converter][INFO] - Epoch: 200, Loss: 0.0002631085517350584, LR: 0.001.
    [2025-01-21 11:41:08,512][artist.util.surface_converter][INFO] - Epoch: 300, Loss: 5.31846126250457e-05, LR: 0.0002.
    [2025-01-21 11:41:27,034][artist.util.surface_converter][INFO] - Converting facet 3 of 4.
    [2025-01-21 11:41:27,602][artist.util.surface_converter][INFO] - Epoch: 0, Loss: 0.002238568849861622, LR: 0.001.
    [2025-01-21 11:42:16,646][artist.util.surface_converter][INFO] - Epoch: 100, Loss: 0.00027722641243599355, LR: 0.001.
    [2025-01-21 11:43:05,519][artist.util.surface_converter][INFO] - Epoch: 200, Loss: 0.00028296327218413353, LR: 0.001.
    [2025-01-21 11:43:54,312][artist.util.surface_converter][INFO] - Epoch: 300, Loss: 0.0002574330137576908, LR: 0.001.
    [2025-01-21 11:44:43,152][artist.util.surface_converter][INFO] - Epoch: 400, Loss: 5.116819738759659e-05, LR: 0.0002.
    [2025-01-21 11:44:43,152][artist.util.surface_converter][INFO] - Converting facet 4 of 4.
    [2025-01-21 11:44:43,726][artist.util.surface_converter][INFO] - Epoch: 0, Loss: 0.0021815903019160032, LR: 0.001.
    [2025-01-21 11:45:32,926][artist.util.surface_converter][INFO] - Epoch: 100, Loss: 0.0002895369252655655, LR: 0.001.
    [2025-01-21 11:46:21,622][artist.util.surface_converter][INFO] - Epoch: 200, Loss: 0.00023776448506396264, LR: 0.001.
    [2025-01-21 11:47:10,265][artist.util.surface_converter][INFO] - Epoch: 300, Loss: 4.86823009850923e-05, LR: 0.0002.
    [2025-01-21 11:47:44,279][artist.util.surface_converter][INFO] - Surface configuration based on data complete!
    [2025-01-21 11:47:44,280][artist.util.scenario_generator][INFO] - Generating a scenario saved to: [Your-File-Location-and-Name].
    [2025-01-21 11:47:44,281][artist.util.scenario_generator][INFO] - Using scenario generator version 1.0.
    [2025-01-21 11:47:44,281][artist.util.scenario_generator][INFO] - Including parameters for the power plant.
    [2025-01-21 11:47:44,282][artist.util.scenario_generator][INFO] - Including parameters for the target areas.
    [2025-01-21 11:47:44,283][artist.util.scenario_generator][INFO] - Including parameters for the light sources.
    [2025-01-21 11:47:44,284][artist.util.scenario_generator][INFO] - Including parameters for the prototype.
    [2025-01-21 11:47:44,290][artist.util.scenario_generator][INFO] - Including parameters for the heliostats.

We see that the STRAL data is used to convert the surface to NURBS and following this conversion the scenario generator
includes all defined parameters for the target areas, light sources, prototypes and heliostats and saves the resulting HDF5
file.

If you go to the location you defined at the very start you should now see a HDF5 file there -- and that is all there is
to generating a scenario in ``ARTIST``!

.. warning::

    The logger also reports what version of the scenario generator is currently running. Changes in versions may result
    in a scenario that is incompatible with the current ``ARTIST`` version.
