---
title: '`artist`: A Python Package for AI-Enhanced Differentiable Raytracing in Solar Tower Power Plants'
tags:
  - Python
  - Concentrating Solar Power
  - Solar Tower Power Plants
  - Differentiable Raytracing
authors:
  - name: Marlene Busch
    orcid: 0009-0008-5730-7528
    affiliation: 1
  - name: Kaleb Phipps
    orcid: 0000-0002-9197-1739
    affiliation: 2, 3
  - name: Daniel Maldonado Quinto
    orcid: 0000-0003-2929-8667
    affiliation: 1
  - name: Marie Weiel
    orcid: 0000-0001-9648-4385
    affiliation: 2, 3
  - name: Robert Pitz-Paal
    orcid: 0000-0002-3542-3391
    affiliation: 1, 4
  - name: Markus Götz
    orcid: 0000-0002-2233-1041
    affiliation: 2, 3
  - name: Max Pargmann
    orcid: 0000-0002-4705-6285
    affiliation: 1
affiliations:
 - name: German Aerospace Center (DLR), Institute of Solar Research, Germany
   index: 1
 - name: Karlsruhe Institute of Technology (KIT), Scientific Computing Center (SCC), Germany
   index: 2
 - name: Helmholtz AI, Karlsruhe, Germany
   index: 3
 - name: RWTH Aachen University, Chair of Solar Technology
   index: 4
date: 15 October 2025
bibliography: paper.bib
---

# Summary

`artist` is a software package for the simulation and optimization of concentrating solar power (CSP) plant operation. Solar tower power plants use an array of mirrors (heliostats), to reflect and concentrate sunlight onto a small area called the receiver. The thermal power, represented through a flux distributions in the digital twin, can be used directly as high-temperature heat in industrial processes or it can be converted into carbon-neutral electricity. This Python package, `artist`, implements a fully differentiable digital twin for solar tower power plants, providing tools for data-driven power plant component modeling and aim point optimization. The differentiable ray tracer in `artist` simulates light transport in the three-dimensional scene, including environmental conditions, allowing ray tracing to be integrated in the gradient-based optimizations. All physical hardware is inherently subject to small manufacturing tolerances and imperfections which tend to increase as components age. In solar tower power plants this leads to individually deformed fluxes for each heliostat. To improve efficiency in a solar tower power plant, each heliostats deformed and misaligned flux needs to aim at an individual point on the receiver for an optimal combined flux density distribution. Small surface deformations [@Ulmer:2011] and misalignments in each heliostat due to inaccurate kinematic components [@Sattler:2020] are main contributors for uncertainties which accumulate over the total heliostat field and cannot be neglected in the simulation. The digital twin `artist` provides functionality to reconstruct the real-world heliostat surfaces and the heliostat kinematic. The simulation includes algorithms for alignment and raytracing to predict flux densities on the receiver. Based on this the heliostat aim points can be optimized. The main functionality of `artist` is shown in \autoref{fig:flowchart}.

# Statement of Need

In solar tower power plants, digital twins with precise simulation and reliable predictive capabilities are essential to realize fully autonomous power plant operation and a consequential reduction in costs [@Huang:2021]. While solar tower power plants may vary in their individual architectural details, their digital twins consistently rely on ray tracing. Conventional ray tracers for CSP [@Ahlbrink:2012; @SolTrace; @Tonatiuh:2018] achieve good results in simulating power plant behavior. However, they can only use ray tracing to make predictions based on the supplied data and the resulting model. From a machine learning perspective, these ray tracers are confined to forward computations. Therefore, they often require large amounts of data to function accurately. `artist` addresses this limitation with its differentiable implementation of the ray tracer and all connecting modules. Single differentiable digital twin tasks for CSP simulations have also been addressed in related works, using inverse deep learning for heliostat surface modeling [@Lewen:2025] or generative neural networks for flux predictions [@Kuhl:2024]. In contrast to these black-box AI algorithms, `artist` maintains interpretability due to its foundation in physical models. In CSP technology, transparency in system behavior is critical for building trust in automated decision-making and software generated operation suggestions. The methodology in `artist` remains consistent across all optimization tasks, integrating modeling, prediction and optimization into one combined tool.
`artist` is designed for researchers, power plant operators, developers within the CSP community or anyone else interested in the field. `artist` includes data loaders compatible with various data sources, including the open-access CSP database PAINT [@Phipps:2025]. Therefore, anyone including people who do not have direct access to an operational power plant can contribute to research progress in CSP technologies with `artist`. We aim to strengthen community engagement and collaboration for further research advancements by developing `artist` as an easily accessible open-source software.

![Features of `artist`, the AI-enhanced differentiable Ray Tracer for Irradiation Prediction in Solar Tower Digital Twins. To create digital twins of solar tower power plants in `artist`, users are asked to provide HDF5-files containing data about the physical layout of the power plant. The HDF5 scenarios can be generated by `artist` from various data sources. `artist` unpacks these files to initiate the reconstruction prediction and optimization. The optimized power plant parameters, i.e., optimized motor positions for each heliostat, can be used directly as control input to a power plant control software. To efficiently represent heliostat surfaces in the digital twin, `artist` contains a fully differentiable, parallelized NURBS implementation for three-dimensional surfaces. \label{fig:flowchart}](flowchart.png)

# State of the Field

Commercial digital twin software solutions for solar tower power plants such as the raytracer STRAL [@Ahlbrink:2012] or TieSOL [@Mitchell:2025] are non-differentiable and therefore lack the ability to reconstruct heliostat field models or to optimize power plant parameters. Instead they assume idealized hardware or provide tools for exact modeling which rely on infeasible measurements. These already commercially proven software solutions are typically proprietary and closed-source, limiting the possibilities for public contributions. Considering open-source, commercially validated, non-differentiable ray tracers such as SolTrace [@SolTrace], the modifications needed to propagate gradients through the entire simulation are so extensive and fundamental that the development of a new tool is justified.
In literature many proof-of-concept studies exist, addressing single tasks relevant for solar tower power plant digital twin simulation and optimization. These include differentiable ray tracing [@Pargmann:2024], surface modeling [@Lewen:2025] and kinematic calibration [@Sattler:2024]. While each of these approaches demonstrates its capabilities, they mostly remain isolated solutions. As each single task solution employs an individual methodology specialized for the specific requirements, a simple unification is challenging. `artist` addresses this gap by using one coherent optimization strategy to redefine the single task solutions, adding to them and creating an integrated and practicably usable software product fundamentally different from the task specific solutions but solving the same problems. As no comparable open-source alternative exists, we provide the basis for future contributions.

# Software Design

`artist` is optimized in its design for computational efficiency by balancing memory consumption, execution time and simulation accuracy. To preserve its real-time capabilities and scalability in runtime when considering power plants with several thousand mirrors, `artist` features native GPU acceleration and supports distributed computation. On a single GPU `artist` parallelizes the computation of individual heliostats by leveraging the Structure of Arrays (SoA) format to store data contiguously in memory. Combined with the GPU’s Single-Instruction, Multiple-Thread (SMIT) processing model, the chosen data structure enables coalesced memory access, which minimizes memory overhead and maximizes GPU utilization. For distributed execution on multiple compute nodes `artist` uses a nested data-parallel approach that distributes groups of heliostats among the available compute resources. Additionally, each distributed group can be parallelized through nested sub-processes. Heliostats within a single group share mathematically identical alignment calculations, which differ from the rest, so they must be separated for parallelized computation. In `artist`, parameter learning purely relies on gradient descent optimization combined with physics-informed models and constraints. This combination stabilizes the optimization of the underdetermined system and reduces the search space. It also enables fast convergence on fewer samples which in turn reduces latency and memory usage. Users can configure key simulation and optimization parameters to adjust the trade-off between memory usage, computational time and precision to their needs. The core algorithms in `artist` are implemented using fully differentiable methods building on PyTorch's automatic differentiation system to propagate gradients. Overall, `artist` adheres to the FAIR principles for research software (FAIR4RS) [@Barker:2022] and ensures portability across multiple hardware stacks by leveraging CI/CD pipelines with automated tests on windows, linux and macos. Even though `artist` is optimized for GPU execution, the software supports both CPU and GPU execution. `artist` automatically selects the appropriate communication layer for multi-node parallel execution based on the underlying OS and hardware configuration. The modular architecture, built on abstraction and inheritance, enables its application across diverse solar tower power plant designs. Users can build on existing interfaces to incorporate specific design details of their power plants into the heliostat field models. The software package `artist` is fully documented, with a documentation accessible via Read the Docs, which includes installation instructions, software tutorials, theoretical background information on mathematical concepts and used data structures as well as the API reference.

# Research Impact Statement

The underlying concepts of `artist` are based on previous publications, which demonstrate the potential of the differentiable ray tracing approach [@Pargmann:2024] and more generally show the potential of increasing solar tower power plant efficiency in order to provide an environmentally friendly solution to meet the globally rising demand for energy [@CSPRoadMapNREL; @Carballo:2025; @Edenhofer:2011].

# AI Usage Disclosure

No generative AI tools were used in the development of this software, the writing of this manuscript, or the preparation of supporting materials.

# Acknowledgements

This work is supported by the Helmholtz Association Initiative and Networking Fund through the Helmholtz AI platform, HAICORE@KIT and the ARTIST project under grant number ZT-I-PF-5-159.

# References
