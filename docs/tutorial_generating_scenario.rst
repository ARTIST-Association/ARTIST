.. _tutorial_generating_scenario:

``ARTIST`` Tutorial: Generating a Scenario HDF5 File
====================================================

.. note::

    You can find the corresponding ``Python`` scripts for this tutorial here:

    - https://github.com/ARTIST-Association/ARTIST/blob/main/tutorials/00_generate_scenario_from_stral_tutorial.py
    - https://github.com/ARTIST-Association/ARTIST/blob/main/tutorials/00_generate_scenario_from_paint_tutorial.py

In this tutorial, we will walk through the process of generating simple ``ARTIST`` HDF5 scenario files. Before
starting, please make sure you have read the :ref:`scenario documentation <scenario>` describing the structure of an
``ARTIST`` scenario file.

As outlined in the :ref:`scenario overview <scenario>`, an ``ARTIST`` scenario consists of five main elements:

- One power plant location,
- at least one (but possibly more) target areas,
- at least one (but possibly more) light sources,
- at least one (but usually more) heliostats, and
- a prototype used whenever a heliostat does not define individual parameters.

In this tutorial, we generate minimal example scenarios that include:

- A power plant location,
- one or more target areas,
- a single ``Sun`` light source,
- one or more heliostats, and
- a corresponding prototype defining the heliostat properties.

Defining the Scenario Path
^^^^^^^^^^^^^^^^^^^^^^^^^^

Before defining the scenario content, we must specify where the generated HDF5 file will be stored. This is done by
setting the ``scenario_path`` variable. If the specified directory does not exist, the scenario generation will fail.

.. code-block:: python

    # Specify the path to your scenario file.
    scenario_path = pathlib.Path("please/insert/the/path/to/the/scenario/here/name")

Choosing A Data Source
^^^^^^^^^^^^^^^^^^^^^^

The subsequent steps depend on the data source used to construct the scenario. Data with different input formats from
different sources requires different preprocessing steps before it can be converted into an ``ARTIST`` scenario.
This tutorial shows how to convert STRAL and PAINT data to create usable scenarios for ``ARTIST``. The solar tower ray
tracing laboratory `STRAL <https://elib.dlr.de/78440/>`_ is a ray-tracing software, and
`PAINT <https://paint-database.org/>`_ is the first FAIR database for operational data of concentrating solar power
plants.

We will first cover the workflow for PAINT data. If you are only interested in STRAL, you can jump directly to
:ref:`stral`.

.. _paint:

Generating a Scenario With PAINT Data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To generate an ``ARTIST`` scenario from ``PAINT`` data, you must provide the following files:

- One ``tower-measurement.json`` file
- One or more ``heliostat-properties.json`` files
- One or more ``deflectometry.h5`` files

If you want to include multiple heliostats in your scenario, simply add one set of heliostat-specific files, i.e.,
properties and deflectometry, per heliostat.

.. code-block:: python

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
        # ... Include as many heliostats as you want, but at least one!
    ]

With the required input files defined, we can now proceed to building the scenario.

.. _plant_and_target:

Power Plant and Target Areas
----------------------------
The power plant location and the associated target areas (i.e., the receiver or calibration targets) are loaded
simultaneously from the ``tower-measurement.json`` file. We can extract this information using functions from the
``paint_scenario_parser`` module. The function shown below will return

- an instance of ``PowerPlantConfig`` and
- an instance of ``TargetAreaPlanarListConfig``, containing a list of viable planar target areas.
- an instance of ``TargetAreaCylindricalListConfig``, containing a list of viable cylindrical target areas.

.. code-block:: python

    # Include the power plant and target area configurations.
    (
        power_plant_config,
        target_area_list_planar_config,
        target_area_list_cylindrical_config,
    ) = paint_scenario_parser.extract_paint_tower_measurements(
        tower_measurements_path=tower_file, device=device
    )

The ``PowerPlantConfig`` object provides the power plant's geographic location via the ``power_plant_position``
attribute. The ``TargetAreaPlanarListConfig`` contains a list of multiple ``TargetAreaPlanarConfig`` objects.
Each defines the following attributes:

``target_area_key``
  An identifier used to reference the target area when loading the ``ARTIST`` scenario – in this
  case, a receiver.
``center``
  The target area's middle position. Since this is a position tensor, its final element in the 4D
  representation is a 1 – for more information, see :ref:`our docs page on coordinates <artist_under_hood>`.
``normal``
  The target area plane's normal vector. Since this is a direction tensor, its final element in the
  4D representation is a 0 – for more information, see :ref:`our docs page on coordinates <artist_under_hood>`.
``plane_e``
  The direction vector defining the target area plane's east direction.
``plane_u``
  The direction vector defining the target area plane's up direction.

The ``TargetAreaCylindricalListConfig`` contains a list of multiple ``TargetAreaCylindricalConfig`` objects.
Each defines the following attributes:

``target_area_key``
  An identifier used to reference the target area when loading the ``ARTIST`` scenario – in this
  case, a receiver.
``radius``
  The cylinder radius.
``height``
  The cylinder height.
``axis``
  The cylinder axis. Since this is a direction tensor, its final element in the
  4D representation is a 0 – for more information, see :ref:`our docs page on coordinates <artist_under_hood>`.
``normal``
  The target area plane's normal vector. Since this is a direction tensor, its final element in the
  4D representation is a 0 – for more information, see :ref:`our docs page on coordinates <artist_under_hood>`.
``opening_angle``
  The cylinder opening angle. Cylindrical target areas can either be full cylinders or cylinder sectors.
  For a full cylinder the opening angle is 2 pi.

.. _light_source:

Light Source
------------
The light source provides the radiation that is reflected by the heliostats. In most scenarios, this light source
represents the sun. However, in certain applications, such as calibration setups, it may be useful to model multiple
artificial light sources. Light source information is not read from external files and must be defined manually.
We define the light source by creating a ``LightSourceConfig`` object as shown below:

.. code-block:: python

    # Include the light source configuration.
    light_source_config = LightSourceConfig(
        light_source_key="sun",
        light_source_type=config_dictionary.sun_key,
        number_of_rays=10,
        distribution_type=config_dictionary.light_source_distribution_is_normal,
        mean=0.0,
        covariance=4.3681e-06,
    )

This configuration specifies the following light source properties:

``light_source_key``
  Used to identify the light source when loading the ``ARTIST`` scenario.
``light_source_type``
  The type of light source used – in this case, a ``Sun``.
``number_of_rays``
  The number of rays to be sampled from the light source for ray tracing.
``distribution_type``
  The type of distribution used to model the light source – in this case, a normal distribution.
``mean``
  The mean parameter of the selected normal distribution.
``covariance``
  The covariance parameter of the selected normal distribution.

Although this example uses only a single light source, ``ARTIST`` scenarios are designed to support multiple sources.
Therefore, the light source configuration must be wrapped in a list and passed to a ``LightSourceListConfig`` object:

.. code-block:: python

    # Create a list of light source configs - in this case only one.
    light_source_list = [light_source_config]

    # Include the configuration for the list of light sources.
    light_source_list_config = LightSourceListConfig(light_source_list=light_source_list)


Prototypes and Heliostats
-------------------------
Every ``ARTIST`` scenario requires both prototypes and heliostats (see :ref:`our tutorial here <scenario>` for more
information).

The prototypes and list of heliostats can be easily extracted using the ``paint_scenario_parser``. Before doing so, we
must define a default aim point by selecting one target area from the previously loaded list – typically the
receiver:

.. code-block:: python

    target_area = [
        target_area
        for target_area in target_area_list_config.target_area_list
        if target_area.target_area_key == config_dictionary.target_area_receiver
    ]

Before loading the heliostats, we need to do some configuration. ``ARTIST`` internally models all heliostat surfaces
using :ref:`NURBS <nurbs>`, which are learned when loading the data. Therefore, we need to specify parameters
controlling the fitting process, such as:

- the number of NURBS control points,
- the fitting method
- the tolerance and number of epochs to train, and
- an optimizer and learning rate scheduler for the training process.

This is shown below:

.. code-block:: python

    number_of_nurbs_control_points = torch.tensor([20, 20], device=device)
    nurbs_fit_method = config_dictionary.fit_nurbs_from_normals
    nurbs_deflectometry_step_size = 100
    nurbs_fit_tolerance = 1e-10
    nurbs_fit_max_epoch = 400

    # Leave the optimizable parameters empty, they will automatically be added for the surface fit.
    nurbs_fit_optimizer = torch.optim.Adam([torch.empty(1, requires_grad=True)], lr=1e-3)
    nurbs_fit_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        nurbs_fit_optimizer,
        mode="min",
        factor=0.2,
        patience=50,
        threshold=1e-7,
        threshold_mode="abs",
    )

With the configuration defined, a single function call :

- loads the heliostat list configuration,
- learn the NURBS surfaces, and
- generates the prototype configuration.

.. code-block:: python

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

``heliostat_list_config``
    A list of ``HeliostatConfig`` objects, where each object contains:

    - The ``name`` used to identify the heliostat
    - The numerical ``id`` of the heliostat
    - The heliostat ``position``
    - The ``surface`` configuration of the heliostat (see :py:class:`artist.scenario.configuration_classes.SurfaceConfig`).
    - The ``kinematics`` configuration of the heliostat (see :py:class:`artist.scenario.configuration_classes.KinematicsConfig`).
    - A list of configurations for the ``actuators`` required by the heliostat (see :py:class:`artist.scenario.configuration_classes.ActuatorConfig`).

``prototype_config``
    A ``PrototypeConfig`` object defines fallback configurations used when heliostats do not provide individual
    parameters. It contains:

    - The ``surface_prototype`` (see :py:class:`artist.scenario.configuration_classes.SurfacePrototypeConfig`)
    - The ``kinematics_prototype`` (see :py:class:`artist.scenario.configuration_classes.KinematicsPrototypeConfig`)
    - A list of ``actuators_prototype`` (see :py:class:`artist.scenario.configuration_classes.ActuatorPrototypeConfig`)

Different Surface Options
~~~~~~~~~~~~~~~~~~~~~~~~~

``ARTIST`` does not require deflectometry data to generate a scenario. Instead, scenarios can also be created with
*ideal* heliostat surfaces. The true surface can later be learned via ray tracing (see
:ref:`the NURBS surface reconstructor<tutorial_surface_reconstruction>`). If no information about the true surface
is available, the ideal surface can simply be used as is. To generate heliostats with ideal surfaces, call:

.. code-block:: python

    heliostat_list_config, prototype_config = (
            paint_scenario_parser.extract_paint_heliostats_ideal_surface(
                paths=heliostat_files_list,
                power_plant_position=power_plant_config.power_plant_position,
                device=device,
            )
        )

In this case, no optimizer or NURBS fitting parameters need to be defined.

It is also possible to generate mixed-surface scenarios containing both fitted and ideal surfaces using the function
``extract_paint_heliostats_mixed_surface()``. In this case, the surface type is determined automatically from the
provided input mapping. If you provide a path to a deflectometry file, the surface will be fitted; if not, an ideal
surface will be generated.
For example, for the following mapping:

.. code-block:: python

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

.. code-block:: python

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

will generate a scenario in which ``heliostat_1`` has a fitted surface and ``heliostat_2`` has an ideal surface.

**NOTE:** In mixed-surface scenarios, the prototype surface will always be an ideal surface.


.. _create_hdf5:

Creating the HDF5 File
----------------------

At this point, we have all the information needed to generate the HDF5 file and complete the scenario. We can create the
scenario by running the ``main`` function shown below:

.. code-block:: python

    if __name__ == "__main__":
        # Generate the scenario given the defined parameters.
        scenario_generator = ScenarioGenerator(
            file_path=scenario_path,
            power_plant_config=power_plant_config,
            target_area_list_planar_config=target_area_list_planar_config,
            target_area_list_cylindrical_config=target_area_list_cylindrical_config,
            light_source_list_config=light_source_list_config,
            prototype_config=prototype_config,
            heliostat_list_config=heliostats_list_config,
        )
        scenario_generator.generate_scenario()

Based on the previously defined ``scenario_path`` and our configurations for the receiver(s), light source(s),
prototype, and heliostat(s), a ``ScenarioGenerator`` object is instantiated. This object is then used to generate the
actual HDF5 file.

After running this script, a new HDF5 file will appear at the location you specified at the very beginning – and that is
all it takes to generate a scenario in ``ARTIST``!

.. _stral:

Generating a Scenario with STRAL Data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To generate a scenario from STRAL, you only need a single ``.binp`` file.

.. code-block:: python

    # Specify the path to your stral_data.binp file.
    stral_file_path = pathlib.Path(
        "please/insert/the/path/to/the/stral/data/here/stral_data.binp"
    )

Many of the steps for generating a scenario are very similar to those for PAINT data, but a few differences specific to
STRAL need to be taken into account.

Power Plant
-----------
STRAL data does not include the power plant location, so you must enter the coordinates manually:

.. code-block:: python

    # Include the power plant configuration.
    power_plant_config = PowerPlantConfig(
      power_plant_position=torch.tensor([0.0, 0.0, 0.0], device=device)
    )

More details on the ``PowerPlantConfig`` class are provided above (see :ref:`plant_and_target`).

Target Areas
------------
When using STRAL data, we also need to manually define the ``TargetAreaPlanarListConfig``
and the ``TargetAreaCylindricalListConfig``:

.. code-block:: python

    # STRAL
    # Include a single planar tower target area.
    target_area_list_planar_config = TargetAreaPlanarConfig(
        target_area_key="planar",
        center=torch.tensor([0.0, -50.0, 0.0, 1.0], device=device),
        normal_vector=torch.tensor([0.0, 1.0, 0.0, 0.0], device=device),
        plane_e=8.629666667,
        plane_u=7.0,
    )
    target_area_planar_list_config = TargetAreaPlanarListConfig(
        [target_area_list_planar_config]
    )

    # Include a single cylindrical tower target area.
    target_area_list_cylindrical_config = TargetAreaCylindricalConfig(
        target_area_key="cylinder",
        radius=4.14,
        center=torch.tensor([0.0, 0.0, 0.0, 1.0], device=device),
        height=6.0,
        axis=torch.tensor([0.0, 0.0, 1.0, 0.0], device=device),
        normal=torch.tensor([0.0, 1.0, 0.0, 0.0], device=device),
        opening_angle=60,
    )
    target_area_cylindrical_list_config = TargetAreaCylindricalListConfig(
        [target_area_list_cylindrical_config]
    )

More details on the ``TargetAreaPlanarConfig`` class are provided above (see :ref:`plant_and_target`).

Light Source
------------
Generating a light source using STRAL data is identical to ``PAINT`` data, please see :ref:`light_source`.

Prototypes
----------
With STRAL data, prototypes must be defined manually. A prototype always consists of a surface prototype, a kinematics
prototype, and an actuator prototype.

We start with the surface prototype. First, we need to extract information regarding the facet translation vectors, the
canting, and the surface points and normals from STRAL:

.. code-block:: python

    (
        facet_translation_vectors,
        canting,
        surface_points_with_facets_list,
        surface_normals_with_facets_list,
    ) = stral_scenario_parser.extract_stral_deflectometry_data(
        stral_file_path=stral_file_path, device=device
    )

Before we can generate a NURBS surface based on the surface normals and points from STRAL, we need to define the surface
generator and the optimizer and scheduler to fit the surface:

.. code-block:: python

    surface_generator = SurfaceGenerator(device=device)

    # Leave the optimizable parameters empty, they will automatically be added for the surface fit.
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

.. code-block:: python

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

Alternatively, we can generate an ideal surface that is not fitted based on deflectometry data. To generate such a
surface, we do not need to define an optimizer or scheduler but can simply call:

.. code-block:: python

     surface_config = surface_generator.generate_ideal_surface_config(
        facet_translation_vectors=facet_translation_vectors,
        canting=canting,
        device=device,
    )

To create the surface configuration, we define a surface configuration prototype based on the list of facets
contained in the `SurfaceConfig` from above:

.. code-block:: python

    surface_prototype_config = SurfacePrototypeConfig(facet_list=surface_config.facet_list)

Next, we consider the kinematics prototype. The kinematics in ``ARTIST`` assume that all heliostats initially
point in the south direction; however, depending on the CSP considered, the heliostats may be orientated differently. In
our scenario, we orient the heliostats upwards, i.e., they point directly at the sky. A further element of a kinematics
configuration is ``KinematicsDeviations`` which are small disturbance parameters representing offsets caused by the
two-joint kinematics modeled in ``ARTIST``. In this tutorial, we ignore these deviations. Therefore, we can
create the kinematics prototype by generating a ``KinematicsPrototypeConfig`` object as:

.. code-block:: python

    kinematics_prototype_config = KinematicsPrototypeConfig(
        type=config_dictionary.rigid_body_key,
        initial_orientation=torch.tensor([0.0, 0.0, 1.0, 0.0], device=device),
    )

This object defines:

``type``
  The type used in the scenario – in this case, rigid-body kinematics.
``initial_orientation``
  The initial heliostat orientation which is the direction we defined above.
``KinematicsDeviations``
  The offsets (ignored here).

With the kinematics prototype defined, the final prototype required is the actuator prototype. For the rigid-body
kinematics, we need exactly two actuators. Since STRAL data does not include motor position limits, we have to define
them manually. Here, we use the minimum and maximum motor positions for the Jülich plant:

.. code-block:: python

    min_max_motor_positions_actuator_1 = [0.0, 60000.0]
    min_max_motor_positions_actuator_2 = [0.0, 80000.0]

We can now define the actuators using ``ActuatorConfig`` objects as shown below:

.. code-block:: python

    # Include two ideal actuators.
    actuator1_prototype = ActuatorConfig(
        key="actuator_1",
        type=config_dictionary.ideal_actuator_key,
        clockwise_axis_movement=False,
        min_max_motor_positions=min_max_motor_positions_actuator_1,
    )
   actuator2_prototype = ActuatorConfig(
        key="actuator_2",
        type=config_dictionary.ideal_actuator_key,
        clockwise_axis_movement=True,
        min_max_motor_positions=min_max_motor_positions_actuator_2,
    )

These configurations define:

``key``
  The key used when loading the actuator from an ``ARTIST`` scenario.
``type``
  The actuator type – in this case, an ideal actuator for both actuators.
``clockwise_axis_movement``
  Defines whether the actuator operates in a clockwise or counter-clockwise direction.

For different types of actuators, e.g., a linear actuator, we would also have to define specific actuator parameters.
However, we will stick to a simple configuration for this tutorial. To complete the actuator prototype, we wrap both
actuators in a list and generate an ``ActuatorPrototypeConfig`` object:

.. code-block:: python

    # Create a list of actuators.
    actuator_prototype_list = [actuator1_prototype, actuator2_prototype]

    # Include the actuator prototype config.
    actuator_prototype_config = ActuatorPrototypeConfig(
        actuator_list=actuator_prototype_list
    )

With all prototypes defined, we can combine them into the final ``PrototypeConfig`` object as shown below:

.. code-block:: python

    # Include the final prototype config.
    prototype_config = PrototypeConfig(
        surface_prototype=surface_prototype_config,
        kinematics_prototype=kinematics_prototype_config,
        actuator_prototype=actuator_prototype_config,
    )

Heliostat from ``STRAL``
------------------------
Having defined the prototype, we can now define our heliostat by creating a ``HeliostatConfig`` object:

.. code-block:: python

    # Include the configuration for a heliostat.
    heliostat1 = HeliostatConfig(
        name="heliostat_1",
        id=1,
        position=torch.tensor([0.0, 5.0, 0.0, 1.0], device=device),
    )

This heliostat configuration specifies:

``name``
  A name used to identify the heliostat when loading the ``ARTIST`` scenario.
``id``
  A unique identifier that can be used to quickly identify the heliostat within the scenario.
``position``
  The heliostat's position in the field. Note the one in the fourth dimension according to the previously discussed
  :ref:`coordinate convention <coordinates>`.

Since no individual surface, kinematics, or actuator parameters are provided, this heliostat will automatically use the
prototype configurations. Since ``ARTIST`` is designed to support multiple heliostats, we need to wrap our heliostat
configuration in a list and create a ``HeliostatListConfig`` object:

.. code-block:: python

    heliostat_list = [heliostat1]

    # Create the configuration for all heliostats.
    heliostats_list_config = HeliostatListConfig(heliostat_list=heliostat_list)

If individual measurements are available, it is also possible to define custom surface, kinematics, and actuator
configurations for each heliostat.

Creating the HDF5 File
----------------------
Creating the HDF5 based on STRAL data follows the same procedure as with PAINT data (see :ref:`create_hdf5`).

.. warning::

    When generating a scenario, the logger reports the version of the scenario generator being used. Scenario files
    generated with a different version may be incompatible with the current ``ARTIST`` version.
