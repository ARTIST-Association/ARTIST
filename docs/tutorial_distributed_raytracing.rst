.. _tutorial_distributed_raytracing:

``ARTIST`` Tutorial: Distributed Raytracing
===========================================

.. note::

    You can find the corresponding ``Python`` script for this tutorial here:
    https://github.com/ARTIST-Association/ARTIST/blob/main/tutorials/01_heliostat_raytracing_distributed_tutorial.py

This tutorial provides a brief introduction to ``ARTIST`` showcasing how the distributed environment is set up by performing distributed raytracing.

It is best if you already know about the following processes in ``ARTIST``

- How to load a scenario.
- Activating the kinematic in a heliostat to align this heliostat for raytracing.
- Performing heliostat raytracing to generate a flux density image on the receiver.

If you need help with this look into our other tutorials such as the tutorial on :ref:`heliostat raytracing <tutorial_heliostat_raytracing>`.

The Distributed Environment
---------------------------
The Heliostat-Tracing process can be parallelized using Distributed Data Parallel.
Based on the available devices, the environemnt is initialized with the appropriate communication backend.
For computation on GPUs the ``nccl`` backend optimized for NVIDIA GPUs is chosen. For computation on CPUs ``gloo`` is used as backend.
If the program is run without the intention of being distributed, the world size will be set to 1, accordingly the only rank is 0.
All of this is handled by running the following code:

.. code-block::

    # The distributed environment is setup and destroyed using a Generator object.
    environment_generator = utils.setup_distributed_environment(device=device)

    device, is_distributed, rank, world_size = next(environment_generator)

This completly sets up the distributed environment. To use it during the raytracing process, we initialize the
``HeliostatRayTracer`` slightly different than before:

.. code-block::

    # Create raytracer
    raytracer = HeliostatRayTracer(
        scenario=scenario, world_size=world_size, rank=rank, batch_size=1, random_seed=rank
    )


We can specify the ``world_size`` and the ``rank`` because both were set up earlier.
The ``HeliostatRayTracer`` handles all the parallelization for you. The ray tracing process is distributed over the defined number
ranks. Each rank handles a portion of the overall rays.

**Example**
Let's say our ``world_size`` is four. We will now have four individual ``ranks`` that perform Heliostat Raytracing in parallel.
Since we are using Distributed Data Parallel, each ``rank`` is assigned an exact copy of the single heliostat in our scenario.
The data, in our case the rays, are split up and each ``rank`` handles a portion of them. Each ray is assigned to exactly one
``rank``, no ray is duplicated. If we were to plot the results of all four raytracing results of the seperate ``ranks``, we get these
Flux Density Distributions:

+------------------------+------------------------+------------------------+------------------------+
| .. image:: ./images/distributed_flux_rank_0.png | .. image:: ./images/distributed_flux_rank_1.png |
|    :scale: 25%                                  |    :scale: 25%                                  |
|                                                 |                                                 |
+------------------------+------------------------+------------------------+------------------------+
| .. image:: ./images/distributed_flux_rank_2.png | .. image:: ./images/distributed_flux_rank_3.png |
|    :scale: 25%                                  |    :scale: 25%                                  |
|                                                 |                                                 |
+------------------------+------------------------+------------------------+------------------------+

The only step left is to add up all of those bitmaps to receive the total Flux Density Distribution from the considered heliostat:

.. code-block::

    if is_distributed:
        final_bitmap = torch.distributed.all_reduce(
            final_bitmap, op=torch.distributed.ReduceOp.SUM
        )

The total Flux Density Distribution now looks like this:

.. figure:: ./images/distributed_final_flux.png
   :width: 80 %
   :align: center

Cleaning up the Distributed Environment
---------------------------------------
When trying to initialize another distributed task in the same program by creating another process group,
it is important to make sure that the two groups dont get mixed up. This is why we should explicitly
destroy the process group used for the raytracing after we are done using it.
This is also handled by the ``environment_generator`` we set up in the beginning of this tutorial.
Simply execute the following code and you are done:

.. code-block::

    # Make sure the code after the yield statement in the environment Generator
    # is called, to clean up the distributed process group.
    try:
        next(environment_generator)
    except StopIteration:
        pass


Further Information
-------------------
Currently the heliostat-raytracing parallelization with DDP parallelizes over the ``number_of_rays``
which is set in the ``lightsource``. During the initialization of the ``HeliostatRayTracer``, a ``DistortionsDataset``
is set up. This dataset is later handed to a sampler and a data loader that distribute individual parts of
the dataset among the distributed ranks. The ``DistortionsDataset`` samples ray distortions according to the
parameters in the ``lightsource``. In the end the dataset contains a tuple of ray distortions in the east and up direction.
If we inspect one element of the dataset tuple for example ``distortions_e`` (and everything is the same for ``distortions_u```),
we see that it is a multi-dimensional tensor of shape (number of rays, number of facets, number of surface points per facet).
This means for each surface point on each facet we sample 5 different ray distortions. As defined in the ``DistortionsDataset``,
the length of the dataset always equals to ``number_of_rays``. The dataset is split by the sampler and loader along this dimension.
If ``number_of_rays`` is only one, the dataset cannot be split, all rays go to rank zero, even if you parallelize with four ranks.
rank one to three will be idle. If the ``number_of_rays`` is greater or equal the world size, all ranks will receive data.
