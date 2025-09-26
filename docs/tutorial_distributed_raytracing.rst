.. _tutorial_distributed_raytracing:

``ARTIST`` Tutorial: Distributed Ray Tracing
============================================

.. note::

    You can find the corresponding ``Python`` script for this tutorial here:
    https://github.com/ARTIST-Association/ARTIST/blob/main/tutorials/02_heliostat_raytracing_distributed_tutorial.py

This tutorial provides a brief introduction to ``ARTIST`` showcasing how the distributed environment is set up by performing distributed ray tracing.

It is best if you already know about the following processes in ``ARTIST``

- How to load a scenario.
- Aligning heliostats.
- Performing heliostat ray tracing to generate a flux density image on a target area.

If you need help with this look into our tutorial on :ref:`heliostat raytracing <tutorial_heliostat_raytracing>`.

Initial Setup
-------------

``ARTIST`` is designed for parallel computation. To enable this (even when considering different types of heliostats
with different kinematic and actuator configurations) we require ``HeliostatGroups``. Detailed information on heliostat
groups and how ``ARTIST`` is designed can be found in this description of :ref:`what is happening under the hood<artist_under_hood>`
in ``ARTIST``.

Therefore, before we do anything we need to make sure we know how many heliostat groups are present. This can be achieved
by calling the ``get_number_of_heliostat_groups_from_hdf5()`` function in the ``Scenario`` class:

.. code-block::

    number_of_heliostat_groups = Scenario.get_number_of_heliostat_groups_from_hdf5(
        scenario_path=scenario_path
    )

In the distributed ray tracing the heliostat-tracing process can be distributed and parallelized using Distributed Data
Parallel. For the distributed ray tracing using DDP, not only are the heliostat groups computed in parallel, but the
data samples per group can also be computed in parallel. We will see exactly how this works later in the tutorial.

The Distributed Environment
---------------------------

Before we start running raytracing, we need to set up the distributed environment. Based on the available devices, the
environment is initialized with the appropriate communication backend. For computation on GPUs the ``nccl`` backend
optimized for NVIDIA GPUs is chosen. For computation on CPUs ``gloo`` is used as backend. If the program is run without
the intention of being distributed, the world size will be set to 1, accordingly the only rank is 0.

All of this setup is handled automatically via:

.. code-block::

    with setup_distributed_environment(
        number_of_heliostat_groups=number_of_heliostat_groups,
        device=device,
    ) as ddp_setup:

**Note:** The rest of the tutorial occurs within this ``with`` block. This ensures that the distributed environment is
running during execution and will be automatically cleaned up afterwards. The dictionary ``ddp_setup`` contains all
distributed environment parameters.


Mapping between active heliostats, target areas and incident ray directions
---------------------------------------------------------------------------

``ARTIST`` offers the flexibility, to activate and deactivate certain heliostats in the scenario, to have some heliostats
aim at one target area, while others aim elsewhere and also to have different incident ray directions for different heliostats
in the same alignment and raytracing process. Differing incident ray directions for different heliostats may not make much
sense in the usual operation of the power plant, but this is very useful for calibration tasks.

To map each heliostat with its designated target area and incident ray direction you can use the following mapping structure:

.. code-block::

    # heliostat_target_light_source_mapping = [
        ("heliostat_1", "target_name_2", incident_ray_direction_tensor_1),
        ("heliostat_2", "target_name_2", incident_ray_direction_tensor_2),
        (...)
    ]

However, in this tutorial we want to consider all heliostats and therefore set our mapping to ``None``:

.. code-block::

    heliostat_target_light_source_mapping = None

In this case it is later still possible to set a specific default target area index and a default incident ray direction, however
if these are not provided then all heliostats are assigned to the first target area found in the scenario with a incident
ray direction of "north", i.e., the light source position is directly in the south.


Distributed Raytracing
----------------------

Now we are almost ready to start the distributed raytracing, however we need to first set the resolution of the generated
bitmap, and also create a tensor to store the final result:

.. code-block::

    bitmap_resolution = torch.tensor([256, 256])

    combined_bitmaps_per_target = torch.zeros(
        (
            scenario.target_areas.number_of_target_areas,
            bitmap_resolution[0],
            bitmap_resolution[1],
        ),
        device=device,
    )

Now the heliostat groups come in to play. We need to consider each heliostat group separately - in a distributed setting
these groups can be computed in parallel, otherwise they will be processed sequentially. Therefore, the entire distributed
raytracing process takes place within a ``for`` loop:

.. code-block::

    for heliostat_group_index in ddp_setup[config_dictionary.groups_to_ranks_mapping][
        ddp_setup[config_dictionary.rank]
    ]:
        heliostat_group = scenario.heliostat_field.heliostat_groups[
            heliostat_group_index
        ]

Within this loop, the first step is to determine which heliostats are being considered ("activated") and which target
areas are being used -- this is achieved using the ``heliostat_target_light_source_mapping`` that we defined earlier:

.. code-block::

    (
        active_heliostats_mask,
        target_area_mask,
        incident_ray_directions,
    ) = scenario.index_mapping(
        heliostat_group=heliostat_group,
        string_mapping=heliostat_target_light_source_mapping,
        device=device,
    )

We can then activate the heliostats as in the :ref:`previous tutorial on single heliostat raytracing<tutorial_heliostat_raytracing>`:

.. code-block::

    # For each index 0 indicates a deactivated heliostat and 1 an activated one.
    # An integer greater than 1 indicates that the heliostat in this index is regarded multiple times.
    heliostat_group.activate_heliostats(
        active_heliostats_mask=active_heliostats_mask, device=device
    )

and also align the surfaces for all activated heliostats with the incident ray direction:

.. code-block::

    heliostat_group.align_surfaces_with_incident_ray_directions(
        aim_points=scenario.target_areas.centers[target_area_mask],
        incident_ray_directions=incident_ray_directions,
        active_heliostats_mask=active_heliostats_mask,
        device=device,
    )

Now we are ready to create a distributed ``HeliostatRayTracer``. In this case it is important to provide the ``world_size``,
the ``rank``, the ``batch_size``, and a ``random_seed``:

.. code-block::

    ray_tracer = HeliostatRayTracer(
        scenario=scenario,
        heliostat_group=heliostat_group,
        world_size=ddp_setup[config_dictionary.heliostat_group_world_size],
        rank=ddp_setup[config_dictionary.heliostat_group_rank],
        batch_size=heliostat_group.number_of_active_heliostats,
        random_seed=ddp_setup[config_dictionary.heliostat_group_rank],
        bitmap_resolution=bitmap_resolution,
    )

In this tutorial the ``batch_size`` is equal to the number of active heliostats. The ``batch_size`` determines how many heliostats
are parallelized within this group's raytracing process. If the number of active heliostats is high and your GPUs do not have enough
memory capacity, you can reduce the ``batch_size`` to prevent ``CUDA out of memory`` errors during runtime. However, this also means
slightly longer runtimes, as the batches within each group are then also computed sequentially.

Now we are ready to perform raytracing! This is still performed on a per-heliostat basis with the function ``trace_rays()``:

.. code-block::

    bitmaps_per_heliostat = ray_tracer.trace_rays(
        incident_ray_directions=incident_ray_directions,
        active_heliostats_mask=active_heliostats_mask,
        target_area_mask=target_area_mask,
        device=device,
    )

Consider an example scenario, with two heliostat groups that have two heliostats each:
 - ``Group 0``: ``AA28``, ``AC43``
 - ``Group 1``: ``AA31``, ``AA39``

The ``world_size`` is three, this means there is ``rank 0``, ``rank 1`` and ``rank 2``. The ranks are distributed among the groups in a
round-robin fashion, therefore ``Group 0`` is computed on ``rank 0`` and ``rank 2`` while ``Group 1`` is computed on ``rank 1``. Since
``Group 0`` has 2 ranks available, this group can perform nested parallelization. Heliostat 0 of ``Group 0``, named ``AA28`` is handled
by ``rank 0`` and heliostat 1 of ``Group 0`` named ``AC43`` is handled by ``rank 2``. ``Group 1`` has two heliostats but only one rank
assigned, meaning there is no nested parallelization possible.
The ray tracer method ``trace_rays()`` produces bitmaps per heliostat.

.. list-table:: Bitmaps per heliostats
   :widths: 33 33 33
   :header-rows: 0

   * - .. figure:: ./images/bitmap_of_heliostat_AA28_in_group_0_on_rank_0.png
          :scale: 30%

          Rank 0
     - .. figure:: ./images/bitmap_of_heliostat_AA31_in_group_1_on_rank_1.png
          :scale: 30%

          Rank 1
     - .. figure:: ./images/bitmap_of_heliostat_AA28_in_group_0_on_rank_2.png
          :scale: 30%

          Rank 2


   * - .. figure:: ./images/bitmap_of_heliostat_AC43_in_group_0_on_rank_0.png
          :scale: 30%

          Rank 0
     - .. figure:: ./images/bitmap_of_heliostat_AA39_in_group_1_on_rank_1.png
          :scale: 30%

          Rank 1
     - .. figure:: ./images/bitmap_of_heliostat_AC43_in_group_0_on_rank_2.png
          :scale: 30%

          Rank 2

However, now there may be multiple heliostats in the scenario all focusing on the same target. In this case, we need to
determine the resulting flux image for that target, i.e., the combined result of all heliostats focusing on this target.
This can be achieved with the ``get_bitmaps_per_target()`` function:

.. code-block::

    bitmaps_per_target = ray_tracer.get_bitmaps_per_target(
        bitmaps_per_heliostat=bitmaps_per_heliostat,
        target_area_mask=target_area_mask,
        device=device,
    )

Since there may also be multiple heliostats in one group, we need to make sure the results from all heliostats are considered in
this bitmap:

.. code-block::

    combined_bitmaps_per_target = combined_bitmaps_per_target + bitmaps_per_target

All heliostats in this example are aimed at the same target area, called the ``multi_focus_tower``, this is the first target area in this scenario.
This means all bitmaps in the ``combined_bitmaps_per_target`` tensor are empty, except the ones in index 0 (only those will be plotted from now on).

.. list-table:: Bitmaps per target area (on the ``multi_focus_tower``)
   :widths: 33 33 33
   :header-rows: 0

   * - .. figure:: ./images/combined_bitmap_on_multi_focus_tower_from_group_0_on_rank_0.png
          :scale: 30%

          Rank 0
     - .. figure:: ./images/combined_bitmap_on_multi_focus_tower_from_group_1_on_rank_1.png
          :scale: 30%

          Rank 1
     - .. figure:: ./images/combined_bitmap_on_multi_focus_tower_from_group_0_on_rank_2.png
          :scale: 30%

          Rank 2

Notice how only the bitmap on ``rank 1`` is actually a combined bitmap of two individual fluxes. This is because both of those fluxes,
from heliostats ``AA31`` and ``AA39`` were actually computed on the same rank and since the ranks have not been synchronized yet, each
rank only has the information it computed on its own.
Neither the ray tracing results within each group, nor the combined results from each group have been synchronized. Therefore, to obtain
the final bitmap per target we need to perform an ``all_reduce``. One final ``all_reduce`` is sufficient, but for the purpose of this
tutorial it is interesting to look at intermediate results and the nested ``all_reduce``.

.. code-block::

    if ddp_setup[config_dictionary.is_nested]:
        torch.distributed.all_reduce(
            combined_bitmaps_per_target,
            op=torch.distributed.ReduceOp.SUM,
            group=ddp_setup[config_dictionary.process_subgroup],
        )

.. list-table:: Bitmaps per target area (on the ``multi_focus_tower``) after nested reduce
   :widths: 33 33 33
   :header-rows: 0

   * - .. figure:: ./images/reduced_bitmap_on_multi_focus_tower_on_rank_0.png
          :scale: 30%

          Rank 0
     - .. figure:: ./images/reduced_bitmap_on_multi_focus_tower_on_rank_1.png
          :scale: 30%

          Rank 1
     - .. figure:: ./images/reduced_bitmap_on_multi_focus_tower_on_rank_2.png
          :scale: 30%

          Rank 2

This ``all_reduce`` is performed per process subgroup, meaning it only reduces the results of heliostats within the respective
group and can be skipped because the global ``all_reduce`` would handle it as well.
The final bitmap on each target is reduced by:

.. code-block::

    if ddp_setup[config_dictionary.is_distributed]:
        torch.distributed.all_reduce(
            combined_bitmaps_per_target, op=torch.distributed.ReduceOp.SUM
        )

.. list-table:: Bitmaps per target area (on the ``multi_focus_tower``) after final reduce
   :widths: 33 33 33
   :header-rows: 0

   * - .. figure:: ./images/final_reduced_bitmap_on_multi_focus_tower_on_rank_0.png
          :scale: 30%

          Rank 0
     - .. figure:: ./images/final_reduced_bitmap_on_multi_focus_tower_on_rank_1.png
          :scale: 30%

          Rank 1
     - .. figure:: ./images/final_reduced_bitmap_on_multi_focus_tower_on_rank_2.png
          :scale: 30%

          Rank 2

Now all ranks are synchronized and we have the final image shared across them. With that we have completed fully
distributed raytracing in ``ARTIST``!

.. note::

    The images generated in this tutorial are for illustrative purposes, often with reduced resolution and without
    hyperparameter optimization. Therefore, they should not be taken as a measure of the quality of ``ARTIST``. Please
    see our publications for further information.
