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
  - name: second author
    orcid: 0000-0000-0000-0000
    affiliation: 1
  - name: third author
    orcid: 0000-0000-0000-0000
    affiliation:
  - name: fourth author
    orcid: 0000-0000-0000-0000
    affiliation: 1
  - name: fifth author
    orcid: 0000-0000-0000-0000
    affiliation: 1
affiliations:
 - name: German Aerospace Center (DLR), Institute of Solar Research, Germany
   index: 1
 - name: Karlsruhe Institute of Technology (KIT), Scientific Computing Center (SCC), Germany
   index: 2
 - name: Helmholtz AI, Karlsruhe, Germany
   index: 3
date: 00 June 2025
bibliography: paper.bib
---

# Summary
The core component of `artist` is a differentiable ray tracer. Generally, ray tracers simulate how light interacts with objects in a 3D scene. `artist` is specialized for applications in Concentrating Solar Power (CSP) research and the optimization of Solar Tower Power Plant operation. Solar Tower Power Plants use mirrors (heliostats) to reflect and concentrate sunlight onto a small area, called the receiver, located at the top of the solar tower. Ray tracers compute flux density distributions based on environmental conditions, to ultimatly optimize the distribution of the heat or thermal power on the receiver. All within a single tool, `artist` combines its ray tracing capabilities with the functionality to create entire digital twins of CSP heliostat fields and receivers. Furthermore, this Python package offers a high-performance, memory-efficient interface for various optimizations and for parameter learning of the power plant components. `artist` can learn parameters of heliostat surface deformations from readily available data, such as measured flux density distributions, but also offers the more traditional approach to model surfaces from measured point cloud data. Additionally, artist provides calibration algorithms for the kinematic system. This accounts for heliostat alignment errors caused by actuator inaccuracies. Through learning real-world parameters, artist enhances the accuracy of its ray tracing results and predicts more accurate flux density distributions. Leveraging the machine learning framework `PyTorch`, `artist` is highly optimized for use on GPUs and offers options for parallelized and distributed computation, ensuring scalability.

# Statement of Need

Concentrating Solar Power is a sustainable and renewable alternative to fossil fuels and nuclear energy, providing a more environmentally friendly solution to meet the globally rising demand for energy. The absorbed thermal power in a solar tower can be converted into electricity, high-temperature heat for industrial processes, or carbon-neutral fuels. The economic performance of Solar Tower Power Plants has yet to reach its full potential as operational costs continue to be high due to real-time control requirements, dynamic weather conditions, and temperature constraints. Digital twins with advanced simulation techniques as well as precise behavior analyses and prediction capabilities are essential for establishing fully autonomous power plant operation and a consequential reduction in costs. While Solar Tower Power Plants may vary in their individual architectural details, their digital twins consistently rely on ray tracing. Conventional ray tracers [@Ahlbrink:20212], [@SolTrace], [@Tonatiuh:2028], achieve good results in simulating power plant behavior. However, they can only use ray tracing for making predictions based on supplied data and the current model, rather than learning parameters or optimizing the model itself. From a machine learning perspective, these raytracers are confined to forward computations only. Consequently, they often require large amounts of data to function accurately. `artist` addresses this issue with its differentiable implementation of the ray tracer and all connecting modules. The differentiability significantly improves the data requirements for CSP digital twins and also unlocks additional applications for `artist`, including aim point control and optimization as well as heliostat field layout optimization and solar tower design optimization for power plants situated in diverse environmental conditions. The underlying concepts of `artist` are based on previous publications [@Pargmann:2024], which have demonstrated their potential in increasing solar tower power plant efficiency. `artist`'s modular architecture, built on abstraction and inheritance, enables its application in diverse solar tower power plant designs. Users can incorporate specific design details and define custom behavior of their power plant while utilizing shared differentiable algorithms for alignment, ray tracing and optimization defined in `artist`. This software is designed for researchers, power plant operators, developers within the CSP community or anyone else interested in the topic. `artist` provides data loaders and converters that are compatible with various data sources, including the open-access CSP database PAINT [@Phibbs:2025], for users who do not have direct access to an operational power plant. Overall the accessibility of the data, the modularity and the adherence to the FAIR-principles for research software [@Barker:2022] aims to strengthen community engagement and collaboration for further publicly accessible research advancements.

# Features
The main features of `artist` are shown in Figure \ref{fig:flowchart}. To create digital twins of solar tower power plants in `artist`, users are asked to provide HDF5 files containing data about the physical layout of the power plant. The HDF5 scenarios can be generated by `artist` from various data sources. `artist` unpacks these files to initiate the simulation and optimization process by aligning heliostats, performing raytracing, predicting flux density distributions and performing calibration routines. To efficiently handle and learn heliostat surfaces `artist` contains a fully differentiable Non-Uniform Rational B-Spline (NURBS) implementation.
![Features of `artist` the AI-enhanced differentiable Ray Tracer for Irradiation Prediction in Solar Tower Digital Twins. \label{fig:flowchart}](artist_flowchart.png)

# Acknowledgements

# References
