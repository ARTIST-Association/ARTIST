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

`artist`is a software package for concentrating solar power (CSP) plant digital twins. Solar tower power plants use an array of mirrors (heliostats), to reflect and concentrate sunlight onto a small area called the receiver. This process generates heat energy which is either used directly in industrial processes or to produce electricity. Efficient power plant operation is complex and differentiable digital twins can play an important role in enabling data-driven optimization and control. This Python package, `artist`, implements a fully differentiable digital twin for solar tower power plants, allowing for high-performance, memory-efficient optimization and parameter learning of the plant's components. At its core, the differentiable ray tracer simulates how light interacts with the three-dimensional scene, including environmental conditions, enabling gradient-based optimization from predicted flux distributions. By including differentiable models of all power plant components - including Non-Uniform Rational B-Splines (NURBS) surface models - `artist` can be used for highly accurate surface reconstruction, kinematic reconstruction, and aim point optimization. To ensure scalability, `artist` features native GPU acceleration, data-parallel processing, support for distributed computation, and is designed for portability across multiple hardware stacks.


# Statement of Need

Concentrating solar power is a sustainable and renewable alternative to fossil fuels and nuclear energy, providing an environmentally friendly solution to meet the globally rising demand for energy [@CSPRoadMapNREL]. The absorbed thermal power in a solar tower can be converted into electricity or high-temperature heat for industrial processes. The economic performance of solar tower power plants has yet to reach its full potential, as operational costs remain high due to mechanical imperfections, real-time control requirements and dynamic weather conditions [@Carballo:2025]. Digital twins with advanced simulation techniques, as well as precise behavior analysis and prediction capabilities are essential for establishing fully autonomous power plant operation and a consequential reduction in costs [@Huang:2021]. While solar tower power plants may vary in their individual architectural details, their digital twins consistently rely on ray tracing. Conventional ray tracers [@Ahlbrink:2012], [@SolTrace], [@Tonatiuh:2018] achieve good results in simulating power plant behavior. However, they can only use ray tracing to make predictions based on supplied data and their current model. From a machine learning perspective, these ray tracers are confined to forward computations, and therefore they often require large amounts of data to function accurately. `artist` addresses this limitation with its differentiable implementation of the ray tracer and all connecting modules. The differentiability significantly improves the data requirements for CSP digital twins and also enables additional applications, including heliostat field layout optimization and solar tower design optimizations. The underlying concepts of `artist` are based on previous publications, which have demonstrated the potential of increasing solar tower power plant efficiency [@Pargmann:2024]. `artist`'s modular architecture, built on abstraction and inheritance, enables its application across diverse solar tower power plant designs. Users can incorporate specific design details and define custom power plant behavior to be used in combination with shared differentiable algorithms for alignment, ray tracing, heliostat surface reconstruction, and kinematic reconstruction already defined in `artist`. This software is designed for researchers, power plant operators, developers within the CSP community or anyone else interested in the field. `artist` includes data loaders compatible with various data sources, including the open-access CSP database PAINT [@Phipps:2025], for users who do not have direct access to an operational power plant. Overall, the accessibility of the data, the modularity of the software, and its adherence to the FAIR principles for research software [@Barker:2022] aim to strengthen community engagement and collaboration for further research advancements.

# Features

The main features of `artist` are shown in Figure \autoref{fig:flowchart}. To create digital twins of solar tower power plants in `artist`, users are asked to provide HDF5-files containing data about the physical layout of the power plant. The HDF5 scenarios can be generated by `artist` from various data sources. `artist` unpacks these files to initiate the simulation process by aligning heliostats and performing ray tracing to predict flux density distributions. This combination of alignment and ray tracing is used iteratively in the optimization tasks for reconstructing real-world mirror surfaces and the kinematic and for subsequently optimizing the heliostat aim points. The optimized parameters, can be used directly as input to a power plant control software. To efficiently handle heliostat surfaces, `artist` contains a fully differentiable, parallelized NURBS implementation.

![Features of `artist`, the AI-enhanced differentiable Ray Tracer for Irradiation Prediction in Solar Tower Digital Twins. \label{fig:flowchart}](flowchart.png)

# Acknowledgements

This work is supported by the Helmholtz Association Initiative and Networking Fund through the Helmholtz AI platform, HAICORE@KIT and the ARTIST project under grant number ZT-I-PF-5-159.

# References
