.. _tutorial_distributed_raytracing:

``ARTIST`` Tutorial: Distributed Raytracing
==========================================

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

    is_distributed, rank, world_size = next(environment_generator)

To set the device on each rank, run this code:

.. code-block::

    if device.type == "cuda":
        torch.cuda.set_device(rank % torch.cuda.device_count())

This completly sets up the distributed environment. To use it during the raytracing process, we initialize the
``HeliostatRayTracer`` slightly different than before:

.. code-block::

    # Create raytracer
    raytracer = HeliostatRayTracer(
        scenario=scenario,
        world_size=world_size,
        rank=rank,
        batch_size=100,
    )

We can specify the ``world_size`` and the ``rank`` because both were set up earlier.
The ``HeliostatRayTracer`` handles all the parallelization for you. The ray tracing process is distributed over the defined number
ranks. Each rank handles a portion of the overall rays. In the end, after the raytracing we get one flux distribution from each rank.
The only step left is to add up all of those bitmaps to receive the complete flux density distribution from the considered heliostat:

.. code-block::

    if is_distributed:
        final_bitmap = torch.distributed.all_reduce(
            final_bitmap, op=torch.distributed.ReduceOp.SUM
        )

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
