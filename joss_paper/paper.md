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
date: 0 June 2025
bibliography: paper.bib
---


# Summary

Concentrating solar power (CSP) is a sustainable and renewable alternative to fossil fuels and nuclear energy, providing a more environmentally friendly solution to meet the globally rising demand for energy. Solar tower power plants use mirrors (heliostats) to reflect and concentrate sunlight onto a small area, the receiver on top of the solar tower. There the absorbed thermal power can be converted into electricity, high-temperature heat for industrial processes, or carbon-neutral fuels. Due to real-time control requirements, dynamic weather conditions, temperature constraints for the receiver and imperfectly built heliostats, full power plant automation remains challenging and operational costs rise. Concurrently these uncertainties may require artificially lowering the power plant's operating temperature, which in turn reduces overall economic performance. Efficiency can be improved through advanced simulations and precise predictions using a digital twin environment. While solar tower power plants may vary in their individual architectural details, their digital twins consistently rely on ray tracing for accurate performance analysis and optimization.

# Statement of need

Conventional ray tracers [@SolTrace], [@Tonatiuh:2028] achieve good results in simulating power plant behavior. However, they are limited in their application, as they require large amounts of hard to obtain data, such as deflectometry measurements, to achieve high accuracies [@Ahlbrink:20212]. This can be addressed by a differentiable implementation of the ray tracer and all connecting modules. Differentiability generally allows for the learning and optimization of all model parameters and significantly improves the data requirements. `artist` is in its core a differentiable ray tracer and furthermore offers the functionality to create digital twins of CSP heliostat fields and receivers. Designed for researchers, power plant operators, and developers within the CSP community, this Python package offers a high-performance, memory-efficient interface for various solar tower power plant simulation applications, including alignment and ray tracing. Additionally, `artist`'s focus on differentiability facilitates important optimization capabilities such as surface learning and heliostat, all within a single tool. Leveraging the machine learning framework `PyTorch`, `artist` is highly optimized for use on GPUs and offers options for parallelized and distributed computation, ensuring scalability. The underlying concepts implemented in `artist` are based on previous publications [@Pargmann:2024], which have demonstrated their potential in increasing solar tower power plant efficiency and working towards autonomous power plant operation. `artist`'s modular architecture, built on abstraction and inheritance, enables its application in diverse solar tower power plant designs. Users can incorporate specific design details and define custom behavior of their power plant while utilizing shared differentiable algorithms for alignment, ray tracing and optimization defined in `artist`. This modularity, the adherence to the FAIR-principles for research software [@Barker:2022] and `artist` being the first free, open-source differentiable ray tracer for CSP applications aims to strengthen community engagement and collaboration for further publicly accessible research advancements.

# Features

To create digital twins of solar tower power plants in `artist`, users are asked to provide HDF5 files containing data about the physical layout of the power plant. `artist` facilitates research on concentrating solar power without direct access to a power plant and measured data, by providing data converters to generate the standardized HDF5 files from different data sources such as the open access CSP database PAINT [@Phibbs:2025]. The HDF5 scenarios are loaded into `artist` to begin the simulation by aligning heliostats and performing raytracing. The results are predicted flux density distributions for specific aim points and sun positions. Given that physical imperfections of heliostats and other components in real-world power plants can only be economically eliminated to a certain extent, `artist` offers functionality to learn and adapt to these uncertain parameters. In reality, heliostats are defined by minor surface deformities which distort flux images. To efficiently learn these surface variations from readily available data in the form of measured flux density distributions, `artist` contains a fully differentiable implementation of Non-Uniform Rational B-Splines (NURBS). Furthermore, `artist` provides algorithms for kinematic calibration to account for misalignments caused by actuator inaccuracies. By learning and incorporating real-world parameters, `artist` enhances the accuracy of its predictions. This unlocks additional applications for `artist`, including aim point control and optimization as well as heliostat field layout optimization and solar tower design optimization for power plants situated in diverse environmental conditions.

![ARTIST: AI-enhanced differentiable Ray Tracer for Irradiation Prediction in Solar Tower Digital Twins](../logos/artist_logo.svg){width=50%}

# Acknowledgements

# References
