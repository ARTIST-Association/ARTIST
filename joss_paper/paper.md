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

`artist` is a Python software package for the simulation and optimization of concentrating solar power (CSP) plant
operation. Solar tower power plants use an array of mirrors, known as heliostats, to reflect and concentrate sunlight
onto a small receiver located at the top of a tower. The resulting thermal power, represented as a flux distribution on
the receiver, can either be used directly as high-temperature heat in industrial processes or converted into
carbon-neutral electricity.
`artist` implements a fully differentiable digital twin of solar tower power plants and provides tools for data-driven
power plant component modeling and heliostat aim point optimization. Its differentiable ray tracer simulates light
transport within a three-dimensional scene in due consideration of environmental conditions, while enabling the
integration of ray tracing into gradient-based optimization. The key functionality and workflow of `artist` are
illustrated in \autoref{fig:flowchart}.
In real-world operation, all physical plant components are subject to manufacturing tolerances and imperfections, which
typically increase as the system ages. As a result, each heliostat produces an individually distorted flux distribution.
To maximize plant efficiency, the real-world flux of every heliostat must be directed to a specific aim point on the
receiver so that the combined flux density distribution becomes optimal. The main sources of uncertainty are small
surface deformations [@Ulmer:2011] and misalignments in each heliostat caused by inaccuracies in the kinematic
components [@Sattler:2020]. These effects accumulate across the heliostat field and must be accounted for in accurate
simulations. To address these challenges, `artist` provides functions to reconstruct real-world heliostat surface
geometries and kinematic properties. It includes algorithms for alignment and ray tracing that enable the prediction of
flux densities on the receiver. Building on this, heliostat aim points can be optimized to maximize the plant's overall
energy yield.

# Statement of Need

Digital twins of solar tower power plants with precise simulation and reliable predictive capabilities are essential for
enabling fully autonomous operation and thereby reducing costs [@Huang:2021]. While individual plants may differ in
architectural details, their strong dependence on accurately modeling light paths and reflections makes ray tracing a
key component. Conventional CSP ray tracers [@Ahlbrink:2012; @SolTrace; @Tonatiuh:2018] achieve good results in
simulating power plant behavior but are limited to making predictions through forward simulations based on
the supplied data and underlying model. From a machine learning perspective, these ray tracers are thus restricted to
forward computations and often require large amounts of data to achieve high accuracy.
`artist` addresses this limitation through its differentiable implementation of the ray tracer and all connecting
modules. Single differentiable digital twin tasks for CSP simulations have already been explored in related works, such
as inverse deep learning approaches for heliostat surface modeling [@Lewen:2025] and generative neural networks for flux
prediction [@Kuhl:2024]. In contrast to these black-box AI methods, `artist` maintains interpretability through its
foundation in physical models. Transparency in system behavior is critical for building trust in automated decision-making
and software-generated operational recommendations. The methodology implemented in `artist` remains consistent across
all optimization tasks, integrating modeling, prediction and optimization into a unified framework.
`artist` is designed for researchers, power plant operators, developers within the CSP community, and anyone interested
in the field. It provides data loaders compatible with various data sources, including the open-access CSP database
PAINT [@Phipps:2025], enabling contributions from researchers who do not have direct access to an operational power
plant. By developing `artist` as an easily accessible open-source software package, we aim to strengthen community
engagement and foster collaboration for further research advancements.

![Features of `artist`, the AI-enhanced differentiable Ray Tracer for Irradiation prediction in Solar tower digital Twins.
To create digital twins of solar tower power plants in `artist`, users provide HDF5 files containing data about the
power plant's physical layout. These HDF5 scenarios can be generated by `artist` from various data sources.
`artist` unpacks the files to initiate the reconstruction prediction and optimization. The optimized power plant
parameters, i.e., the motor positions for each heliostat, can be used directly as input for power plant
control software. To efficiently represent heliostat surfaces in the digital twin, `artist` includes a fully
differentiable, parallelized NURBS implementation for three-dimensional surfaces. \label{fig:flowchart}](flowchart.png)

# State of the Field

Commercial digital twin software solutions for solar tower power plants, such as the raytracer STRAL [@Ahlbrink:2012] or
TieSOL [@Mitchell:2025], are non-differentiable and therefore lack the ability to reconstruct heliostat field models or
to optimize power plant parameters. Instead, they assume idealized hardware or provide tools for exact modeling which
rely on practically infeasible measurements. Moreover, these commercially proven software solutions are typically
proprietary and closed-source, limiting opportunities for public contributions. Even for open-source, commercially
validated, non-differentiable ray tracers such as SolTrace [@SolTrace], the modifications required to propagate
gradients through the entire simulation would be so extensive and fundamental that the development of a new tool is
justified.
In literature, many proof-of-concept studies address single tasks relevant for digital twin simulation and optimization
of solar tower power plants. These include differentiable ray tracing [@Pargmann:2024], surface modeling
[@Lewen:2025], kinematic calibration [@Sattler:2024], and flux prediction [@Kuhl:2024]. While each of these approaches
demonstrates promising capabilities, they mostly remain isolated solutions. Since each task relies on a methodology
specialized to its specific requirements, straightforward unification is challenging. `artist` addresses this gap
by applying one coherent optimization strategy across all tasks, redefining and extending existing approaches to form an
integrated and practicably usable software product that is fundamentally different from the task-specific solutions but
solves the same problems. As no comparable open-source alternative exists, it provides a solid foundation for future
contributions.

# Software Design

`artist` is designed for computational efficiency by balancing memory consumption, execution time, and simulation
accuracy. To preserve real-time capabilities and maintain scalable runtimes for power plants with several thousand
mirrors, `artist` features native GPU acceleration and supports distributed computation. On a single GPU, `artist`
parallelizes the computation of individual heliostats by storing data in the Structure of Arrays (SoA) format, ensuring
contiguous memory layout. Combined with the GPU’s Single-Instruction, Multiple-Thread (SMIT) processing model, this
data structure enables coalesced memory access, minimizing memory overhead while maximizing GPU utilization. For
distributed execution across multiple compute nodes, `artist` uses a nested data-parallel approach that distributes
groups of heliostats among the available compute resources. Each distributed group can additionally be parallelized
through nested sub-processes. Heliostats within a single group share mathematically identical alignment calculations
that differ from those of other groups, which requires them to be processed separately for efficient parallelization.
In `artist`, parameter learning relies purely on gradient-descent based optimization combined with physics-informed
models and constraints. This combination stabilizes the optimization of the underdetermined system and reduces the
search space. It also enables fast convergence with fewer samples, thereby reducing latency and memory usage. Users can
configure key simulation and optimization parameters to adjust the trade-off between memory usage, computational time,
and precision according to their needs. The core algorithms in `artist` are implemented using fully differentiable
methods building on PyTorch's automatic differentiation system for gradient propagation.
Overall, `artist` adheres to the FAIR principles for research software (FAIR4RS) [@Barker:2022] and ensures portability
across multiple hardware platforms by leveraging CI/CD pipelines with automated tests on Windows, Linux, and MacOS. Even
though `artist` is optimized for GPU execution, it supports both CPU and GPU operation and automatically selects the
appropriate communication layer for multi-node parallel execution based on the underlying operating system and hardware
configuration. Its modular architecture, built on abstraction and inheritance, enables application across diverse solar
tower power plant designs. Users can extend existing interfaces to incorporate plant-specific design details into
heliostat field models. The `artist` software package is fully documented via Read the Docs, including installation
instructions, tutorials, theoretical background on mathematical concepts and data structures as well as a complete API
reference.

# Research Impact Statement

The underlying concepts of `artist` build on previous publications that demonstrate the potential of the differentiable
ray tracing approach [@Pargmann:2024]. More generally, these works highlight the potential for improving solar tower
power plant efficiency as a means of providing an environmentally friendly solution to meet the globally rising demand
for energy [@CSPRoadMapNREL; @Carballo:2025; @Edenhofer:2011].

# AI Usage Disclosure

Generative AI was employed solely as an editorial and technical aid, specifically for debugging code and refining the
manuscript. Generative AI did not contribute to the underlying architectural design or any scientific findings.

# Acknowledgements

This work is supported by the Helmholtz Association Initiative and Networking Fund through the Helmholtz AI platform,
HAICORE@KIT and the ARTIST project under grant number ZT-I-PF-5-159.

# References
