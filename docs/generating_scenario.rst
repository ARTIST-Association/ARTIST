.. _generating_scenario:

``ARTIST`` Tutorial: Generating a Scenario HDF5 File
====================================================

.. note::
    You can find the corresponding ``Python`` script for this tutorial here:
    https://github.com/ARTIST-Association/ARTIST/blob/main/tutorials/01_generate_scenario_heliostat_raytracing.py

In this tutorial we will guide you through the process of generating a simple ``ARTIST`` scenario HDF5 file. Before
starting the tutorial make sure you have read the information regarding the structure of an ``ARTIST`` scenario file
:ref:`that you can find here <scenario>`!

Before we start defining the scenario we need to determine where it will be saved. We define this by setting the
``file_path`` variable. If this location does not exist the scenario generation will automatically fail.

.. code-block::

    # The following parameter is the name of the scenario.
    file_path = "please/insert/your/path/here/name"

As mentioned in the :ref:`information on a scenario <scenario>`, an ``ARTIST`` scenario consists of four main elements:

- At least one (but possibly more) receivers.
- At least one (but possibly more) light sources.
- At least one (but usually more) heliostats.
- A prototype which is used if the individual heliostats do not have individual parameters.

In this tutorial we will develop a very simple ``ARTIST`` scenario that contains:
- One planar receiver.
- One ``Sun`` as a light source.
- One heliostat.
- Uses the prototype to define the properties of the heliostat.

Now we can get started defining each of these elements and then generating the scenario!

Receiver
--------
The receiver is where the reflected light from the heliostats is concentrated. We can define a receiver with the
``ReceiverConfig`` class as shown below:

.. code-block::

    # Include the receiver configuration.
    receiver1_config = ReceiverConfig(
        receiver_key="receiver1",
        receiver_type=config_dictionary.receiver_type_planar,
        position_center=torch.tensor([0.0, -50.0, 0.0, 1.0]),
        normal_vector=torch.tensor([0.0, 1.0, 0.0, 0.0]),
        plane_e=8.629666667,
        plane_u=7.0,
        resolution_e=256,
        resolution_u=256,
    )

This configuration defines the following properties:

- A ``receiver_key`` used to identify the receiver when loading the ``ARTIST`` scenario.
- The ``receiver_type`` currently modelled -- in this case a planar receiver.
- The ``position_center`` which defines the position of the middle of the receiver. Note that because this is a position
  tensor, the final element of the tensor in the 4D representation is a 1 - for more information see
  :ref:`our note on coordinates <coordinates>`.
- A ``normal_vector`` defining the normal vector to the plane of the receiver. Note that because this is a direction
  tensor, the final element of the tensor in the 4D representation is a 0 - for more information see
  :ref:`our note on coordinates <coordinates>`.
- The ``plane_e`` which defines the receiver plane in the east direction.
- The ``plane_u`` which defines the receiver plane in the up direction.
- The resolution parameters, ``resolution_e`` and ``resolution_u`` defining the resolution of the receiver in the east
  and up directions.

Since our scenario only contains one receiver but ``ARTIST`` scenario are designed to load multiple receivers, we have
to wrap our receiver in a list and create a ``ReceiverListConfig`` object.

.. code-block::

    # Create list of receiver configs - in this case only one.
    receiver_list = [receiver1_config]

    # Include the configuration for the list of receivers.
    receiver_list_config = ReceiverListConfig(receiver_list=receiver_list)


Light Source
------------
The light source is the object responsible for providing light that is then reflected by the heliostats. Typically this
light source is a ``Sun``, however in certain situations it may be beneficial to model multiple artificial light
sources. We define the light source by creating a ``LightSourceConfig`` object as shown below:

.. code-block::

    # Include the light source configuration.
    light_source1_config = LightSourceConfig(
        light_source_key="sun1",
        light_source_type=config_dictionary.sun_key,
        number_of_rays=200,
        distribution_type=config_dictionary.light_source_distribution_is_normal,
        mean=0.0,
        covariance=4.3681e-06,
    )

This configuration defines the following light source properties:

- The ``light_source_key`` used to identify the light source when loading the ``ARTIST`` scenario.
- The ``light_source_type`` which defines what type of light source is used. In this case it is a ``Sun``.
- The ``number_of_rays`` which defines how many rays are sampled from the light source for raytracing.
- The ``distribution_type`` which models what distribution is used to model the light source. In this case we use a
  normal distribution.
- The ``mean`` and the ``covariance``, which are the parameters of the previously defined normal distribution used to
  model the light source.

Since our scenario only contains one light source but ``ARTIST`` scenario are designed to load multiple light sources,
we have to wrap our light source in a list and create a ``LightSourceListConfig`` object.

.. code-block::

    # Create a list of light source configs - in this case only one.
    light_source_list = [light_source1_config]

    # Include the configuration for the list of light sources.
    light_source_list_config = LightSourceListConfig(light_source_list=light_source_list)


Prototype
---------
The next step in defining our scenario is to define our *prototype*. We define the prototype before defining the
heliostat, since in this tutorial we load the heliostat based on the prototype parameters. A prototype always contains
a *surface* prototype, a *kinematic* prototype, and a *actuator* prototype.

We start with the *surface* prototype. In this case we generate the surface based on a STRAL scenario using a
``StralToSurfaceConverter``, as defined below:

.. code-block::

    # Generate surface configuration from STRAL data.
    stral_converter = StralToSurfaceConverter(
        stral_file_path=f"{ARTIST_ROOT}/measurement_data/stral_test_data",
        surface_header_name="=5f2I2f",
        facet_header_name="=i9fI",
        points_on_facet_struct_name="=7f",
        step_size=100,
    )

This converter requires:

- A ``stral_file_path`` which is the path to the file containing the STRAL data.
- A ``surface_header_name`` which is required to define the ``Struct`` used to load surface information from the STRAL
  file.
- A ``facet_header_name`` used to define the ``Struct`` to load facet information from the STRAL file.
- A ``points_on_facet_strct_name`` used to define the ``Struct`` to load the points from each facet defined in the STRAL
  file.
- A ``step_size``, which is used to reduce the number of points considered from the STRAl file. Per default, STRAL files
  contain an extremely large number of points which increases compute without improving accuracy. Therefore we only
  select one in 100 points (which still results in approximately 800 points per facet) to reduce this compute.

A surface consists of multiple facets. Since we are using data from STRAL to recreate the surface for our prototype we
can create this list of facets by calling the ``generate_surface_config_from_stral()`` function, as shown below:

.. code-block::

    facet_prototype_list = stral_converter.generate_surface_config_from_stral(
        number_eval_points_e=200,
        number_eval_points_n=200,
        conversion_method=config_dictionary.convert_nurbs_from_normals,
        number_control_points_e=20,
        number_control_points_n=20,
        degree_e=3,
        degree_n=3,
        tolerance=3e-5,
        max_epoch=10000,
        initial_learning_rate=1e-3,
    )

This function loads data from STRAL and then uses this data to learn a Non-Rational Uniform B-Spline (NURBS) surface
for each of the facets. Therefore, this function requires:

- The ``number_of_eval_points_e`` and ``number_of_eval_points_n``. This defines how many evaluation points will be used
  when generating discrete points based on the NURBS surface after loading the ``ARTIST`` scenario.
- The ``conversion_method`` used to learn the NURBS surface. In this case we are learning the surface based on the
  surface normals from the STRAL data.
- The ``number_control_points_e`` and ``number_control_points_n`` which defines the number of control points in the east
  and north direction. These control points are the parameters that are optimised when learning the NURBS surface. As a
  result a larger number of control points allows for finer adjustments but also increases training time.
- The ``degree_e`` and ``degree_n``, which defines the degree of the splines used to model the NURBS in the east and
  north direction.
- The ``tolerance`` which is a threshold for training. Once the NURBS loss is under this threshold the training will
  automatically stop.
- The ``max_epoch`` parameter, which defines the maximum number of epochs used for training. In this case it is 10000,
  however due to the ``tolerance`` parameter the training should stop much earlier.
- The ``initial_learning_rate`` used for learning the NURBS surface. In this case it is 0.001. The training makes use of
  a learning rate scheduler which dynamically adjusts the learning rate during the training process.

The output of this function is a list of ``FacetConfig`` objects, which define the parameters that enable ``ARTIST`` to
recreate the learned NURBS facet surfaces when the scenario is loaded.

Now the facet list has been created automatically by learning NURBS from STRAL data, we need to generate a
``SurfacePrototypeConfig`` object to save the surface.

.. code-block::

    # Generate the surface prototype configuration
    surface_prototype_config = SurfacePrototypeConfig(facets_list=facet_prototype_list)

The next prototype object we consider is the *kinematic* prototype. The first aspect of the kinematic prototype are the
``KinematicOffsets``. The kinematic modelled in ``ARTIST`` assumes that all heliostats are initially pointing in the
south direction, however depending on the CSP considered, the heliostats may initially be orientated in a different
direction.

For our scenario, we want the heliostats to initially be orientated upwards, i.e. they point directly at the sky.
Therefore we need to include a rotation of 90 degrees along the east axis to adjust the initial orientation. We include
this by defining a ``KinematicOffset`` object as shown below:

.. code-block::

    # Include the initial orientation offsets for the kinematic.
    kinematic_prototype_offsets = KinematicOffsets(
        kinematic_initial_orientation_offset_e=torch.tensor(math.pi / 2)
    )

This configuration defines:

- A ``kinematic_initial_orientation_offset_e`` which is an initial orientation offset along the east axis.
- It is also possible to set initial orientation offsets in the north and up direction, however we do not require these
  offsets for our scenario.

A further element of a kinematic configuration are ``KinematicDeviations`` which are small disturbance parameters to
represent offsets caused by the three-joint kinematic modelled in ``ARTIST``. However, in this tutorial we ignore these
deviations. Therefore, we can now create the kinematic prototype by generating a ``KinematicPrototypeConfig`` object.

.. code-block::

    # Include the kinematic prototype configuration.
    kinematic_prototype_config = KinematicPrototypeConfig(
        kinematic_type=config_dictionary.rigid_body_key,
        kinematic_initial_orientation_offsets=kinematic_prototype_offsets,
    )

This object defines:

- The ``kinematic_type`` applied in the scenario, in this case we are using a *rigid body kinematic*.
- The ``kinematic_initial_orientation_offsets`` which are the offsets we defined above.
- If we have ``KinematicDeviations`` we would also include them in this definition.

With the kinematic prototype defined, the final prototype we require is the *actuator* prototype. For the rigid body
kinematic applied in this scenario we require **exactly two** actuators. We can define these actuators via `
``ActuatorConfig`` objects as shown below:

.. code-block::

    # Include an ideal actuator.
    actuator1_prototype = ActuatorConfig(
        actuator_key="actuator1",
        actuator_type=config_dictionary.ideal_actuator_key,
        actuator_clockwise=False,
    )
    # Include a second ideal actuator.
    actuator2_prototype = ActuatorConfig(
        actuator_key="actuator2",
        actuator_type=config_dictionary.ideal_actuator_key,
        actuator_clockwise=True,
    )

These configurations define:

- The ``actuator_key`` used to when loading the actuator from an ``ARTIST`` scenario.
- The ``actuator_type``, which in this case is an ideal actuator for both actuators.
- The ``actuator_clockwise`` parameter, which defines if the actuator operates per default in a clockwise or
  anti-clockwise direction.

If we were considering different types of actuators, e.g. a *linear actuator* we would also have to define specific
actuator parameters -- however we will stick to a simple configuration for this tutorial. To complete the actuator
prototype we need to wrap both actuators in a list and generate an ``ActuatorPrototypeConfig`` object.

.. code-block::

    # Create a list of actuators.
    actuator_prototype_list = [actuator1_prototype, actuator2_prototype]

    # Include the actuator prototype config.
    actuator_prototype_config = ActuatorPrototypeConfig(
        actuator_list=actuator_prototype_list
    )

Now that all the aspects of our prototype are defined we can create the final ``PrototypeConfig`` object, which simply
combines all the above configurations into one object, as shown below:

.. code-block::

    # Include the final prototype config.
    prototype_config = PrototypeConfig(
        surface_prototype=surface_prototype_config,
        kinematic_prototype=kinematic_prototype_config,
        actuator_prototype=actuator_prototype_config,
    )

Heliostat
---------
Having defined the prototype we can now define our heliostat by creating a ``HeliostatConfig`` object as shown below:

.. code-block::

    # Include the configuration for a heliostat.
    heliostat1 = HeliostatConfig(
        heliostat_key="heliostat1",
        heliostat_id=1,
        heliostat_position=torch.tensor([0.0, 5.0, 0.0, 1.0]),
        heliostat_aim_point=torch.tensor([0.0, -50.0, 0.0, 1.0]),
    )

This heliostat configuration requires:

- A ``heliostat_key`` used to identify the heliostat when loading the ``ARTIST`` scenario.
- The ``heliostat_id``, a unique identifier that can be used to quickly identify the heliostat within the scenario.
- The ``heliostat_position`` which defines the position of the heliostat in the field. Note the one in the fourth
  dimension according to the previously discussed :ref:'coordinate convention <coordinates>'.
- The ``heliostat_aim_point`` which defines the desired aim point of the heliostat -- in this case the center of
  the receiver. Note the one in the fourth dimension according to the previously discussed
  :ref:'coordinate convention <coordinates>'.

Since the heliostat does not have any individual surface, kinematic, or actuator parameters we do not need to include
them here. However, since ``ARTIST`` is designed to load multiple heliostats we do need to wrap our heliostat
configuration in a list and create a ``HeliostatListConfig`` object as shown below:

.. code-block::

    # Create a list of all the heliostats - in this case, only one.
    heliostat_list = [heliostat1]

    # Create the configuration for all heliostats.
    heliostats_list_config = HeliostatListConfig(heliostat_list=heliostat_list)


Generate Scenario
-----------------
We have now defined all aspects of our simple scenario. The only step remaining is to generate the scenario. We can
generate this scenario by running the ``main`` function shown below:

.. code-block::

    if __name__ == "__main__":
        """Generate the scenario given the defined parameters."""

        # Create a scenario object.
        scenario_object = ScenarioGenerator(
            file_path=file_path,
            receiver_list_config=receiver_list_config,
            light_source_list_config=light_source_list_config,
            prototype_config=prototype_config,
            heliostat_list_config=heliostats_list_config,
        )

        # Generate the scenario.
        scenario_object.generate_scenario()

This ``main`` function initially defines the ``ScenarioGenerator`` object based on the previously defined ``file_path``
and our configurations for the receiver(s), light source(s), prototype, and heliostat(s).

Running the ``main`` function should produce the following output:

.. code-block::

    [2024-05-23 18:27:08,487][STRAL-to-surface-converter][INFO] - Beginning generation of the surface configuration based on STRAL data.
    [2024-05-23 18:27:08,488][STRAL-to-surface-converter][INFO] - Reading STRAL file located at: /Users/kphipps/Work/Gits/ARTIST/measurement_data/stral_test_data
    [2024-05-23 18:27:10,518][STRAL-to-surface-converter][INFO] - Loading STRAL data complete
    [2024-05-23 18:27:10,518][STRAL-to-surface-converter][INFO] - Converting to NURBS surface
    [2024-05-23 18:27:10,518][STRAL-to-surface-converter][INFO] - Converting facet 1 of 4.
    [2024-05-23 18:27:10,899][STRAL-to-surface-converter][INFO] - Epoch: 0, Loss: 0.0022271068301051855, LR: 0.001
    [2024-05-23 18:27:14,639][STRAL-to-surface-converter][INFO] - Epoch: 100, Loss: 0.00028488607495091856, LR: 0.001
    [2024-05-23 18:27:18,371][STRAL-to-surface-converter][INFO] - Epoch: 200, Loss: 0.0002691124682314694, LR: 0.001
    [2024-05-23 18:27:22,079][STRAL-to-surface-converter][INFO] - Epoch: 300, Loss: 0.00024914421373978257, LR: 0.001
    [2024-05-23 18:27:25,773][STRAL-to-surface-converter][INFO] - Epoch: 400, Loss: 5.134618186275475e-05, LR: 0.0002
    [2024-05-23 18:27:27,010][STRAL-to-surface-converter][INFO] - Converting facet 2 of 4.
    [2024-05-23 18:27:27,052][STRAL-to-surface-converter][INFO] - Epoch: 0, Loss: 0.0023851273581385612, LR: 0.001
    [2024-05-23 18:27:30,793][STRAL-to-surface-converter][INFO] - Epoch: 100, Loss: 0.0002649309462867677, LR: 0.001
    [2024-05-23 18:27:34,495][STRAL-to-surface-converter][INFO] - Epoch: 200, Loss: 0.0002669502573553473, LR: 0.001
    [2024-05-23 18:27:38,181][STRAL-to-surface-converter][INFO] - Epoch: 300, Loss: 5.571055589825846e-05, LR: 0.0002
    [2024-05-23 18:27:41,945][STRAL-to-surface-converter][INFO] - Epoch: 400, Loss: 5.3180556278675795e-05, LR: 0.0002
    [2024-05-23 18:27:42,311][STRAL-to-surface-converter][INFO] - Converting facet 3 of 4.
    [2024-05-23 18:27:42,353][STRAL-to-surface-converter][INFO] - Epoch: 0, Loss: 0.0022385690826922655, LR: 0.001
    [2024-05-23 18:27:46,091][STRAL-to-surface-converter][INFO] - Epoch: 100, Loss: 0.000276801671134308, LR: 0.001
    [2024-05-23 18:27:49,819][STRAL-to-surface-converter][INFO] - Epoch: 200, Loss: 0.0001415298174833879, LR: 0.0002
    [2024-05-23 18:27:53,640][STRAL-to-surface-converter][INFO] - Epoch: 300, Loss: 5.236068318481557e-05, LR: 0.0002
    [2024-05-23 18:27:54,627][STRAL-to-surface-converter][INFO] - Converting facet 4 of 4.
    [2024-05-23 18:27:54,669][STRAL-to-surface-converter][INFO] - Epoch: 0, Loss: 0.0021815903019160032, LR: 0.001
    [2024-05-23 18:27:58,391][STRAL-to-surface-converter][INFO] - Epoch: 100, Loss: 0.000285717542283237, LR: 0.001
    [2024-05-23 18:28:02,108][STRAL-to-surface-converter][INFO] - Epoch: 200, Loss: 0.00024928184575401247, LR: 0.001
    [2024-05-23 18:28:05,795][STRAL-to-surface-converter][INFO] - Epoch: 300, Loss: 0.0002589945506770164, LR: 0.001
    [2024-05-23 18:28:09,491][STRAL-to-surface-converter][INFO] - Epoch: 400, Loss: 4.5302869693841785e-05, LR: 4e-05
    [2024-05-23 18:28:09,565][STRAL-to-surface-converter][INFO] - Surface configuration based on STRAL data complete!
    [2024-05-23 18:28:09,565][scenario-generator][INFO] - Generating a scenario saved to: [Your-File-Location-and-Name]
    [2024-05-23 18:28:09,567][scenario-generator][INFO] - Using scenario generator version 1.0
    [2024-05-23 18:28:09,567][scenario-generator][INFO] - Including parameters for the receivers
    [2024-05-23 18:28:09,568][scenario-generator][INFO] - Including parameters for the light sources
    [2024-05-23 18:28:09,569][scenario-generator][INFO] - Including parameters for the prototype
    [2024-05-23 18:28:09,570][scenario-generator][INFO] - Including parameters for the heliostats

We see that the STRAL data is used to convert the surface to NURBS and following this conversion the scenario generator
includes all defined parameters for the receivers, light sources, prototypes and heliostats and saves the resulting HDF5
file.

If you go to the location you defined at the very start you should now see a HDF5 file there -- and that is all there is
to generating a scenario in ``ARTIST``!

.. warning::

    The logger also reports what version of the scenario generator is currently running. Changes in versions may result
    in a scenario that is incompatible with the current ``ARTIST`` version.
