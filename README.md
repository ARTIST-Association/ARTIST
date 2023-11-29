![ARTIST Logo](logo.svg)

# AI-enhanced differentiable Ray Tracer for Irradiation Prediction in Solar Tower Digital Twins

[![](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![](https://img.shields.io/badge/Contact-max.pargmann%40dlr.de-orange)](mailto:max.pargmann@dlr.de)

## What ARTIST can do for you

The `ARTIST` package provides an implementation of a differentiable ray tracer using the `PyTorch` machine-learning 
framework in `Python`. Leveraging automatic differentiation and GPU computation, it facilitates the optimization of 
heliostats, towers, and camera parameters within a solar field by combining gradient-based optimization methods with 
smooth parametric descriptions of heliostats.

**Our contributions include:**

- **Neural-network driven heliostat calibration:** A two-layer hybrid model for most efficient heliostat calibration. 
  It comprises a robust geometric model for pre-alignment and a neural network disturbance model, which gradually adapts 
  its impact via regularization sweeps. On this way, high data requirements of data-centric methods are overcome while maintaining flexibility for modeling complex real-world systems. 
  Check out [this paper](https://doi.org/10.1016/j.solener.2023.111962) for more details.  

- **Surface reconstruction and flux density prediction:** Leveraging learning Non-Uniform Rational B-Splines (NURBS), 
  `ARTIST` reconstructs heliostat surfaces accurately using calibration images commonly available in solar thermal power plants. 
  Thus, we can achieve sub-millimeter accuracy in mirror reconstruction from focal spot images, contributing to improved 
  operational safety and efficiency. The reconstructed surfaces can be used for predicting unique heliostat flux densities 
  with state-of-the-art accuracy. Check out [this paper](https://doi.org/10.21203/rs.3.rs-2554998/v1) for more details.

- **Advanced data set sampling strategies:** `ARTIST` utilizes a time-independent data set sampling strategy based on Euler 
  angles to improve accuracy by minimizing the needed data for calibration. Check out [this paper](https://doi.org/10.21203/rs.3.rs-2898838/v1) for more details.

- **Immediate deployment**: `ARTIST` enables deployment at the beginning of a solar thermal plant's operation, 
  allowing for in situ calibration and subsequent improvements in energy efficiencies and cost reductions.

- **Optimized flux density:** Coming soon so stay tuned :rocket:!


## Installation
We heavily recommend to install the `ARTIST` package in a dedicated `Python3.8+` virtual environment.
1. Clone the `ARTIST` repository:
   ```bash
   git clone https://github.com/ARTIST-Association/ARTIST.git
   ```
2. Install the package from the main branch:
   ```bash
   pip install .
   ```
## Structure
```
├──artist # Parent package
│   ├───io # IO functionality
│   │   └───tests # IO functionality tests
│   ├───physics_objects # Physical objects in raytracing environment, e.g., heliostats or receiver
│   │   └───heliostats                 
│   │       ├───alignment
│   │       │   └───tests
│   │       │       └───bitmaps
│   │       └───surface
│   │           └───facets
│   ├───raytracing
│   ├───scenario
│   │   └───light_source
│   │       └───tests
│   └───util
└───scenario_objects # Loaded from experiment yaml file to, e.g., define whether the sun or a beamer should be loaded
    └───heliostats                   
```
## How to use ARTIST
We plan to provide an official *ReadTheDocs* documentation including exemplary usage scripts.

## How to contribute
Check out our [contribution guidelines](CONTRIBUTING.md) if you are interested in contributing to the `ARTIST` project :fire:.
Please also carefully check our [code of conduct](CODE_OF_CONDUCT.md) :blue_heart:.

## License
Hopefully MIT.

## Documentation
Coming soon :rocket:!

## Acknowledgments
This work is supported by the [Helmholtz AI](https://www.helmholtz.ai/) platform grant.

-----------
<div align="center">
  <a href="https://www.dlr.de/EN/Home/home_node.html"><img src="./logos/logo_dlr.svg" height="50px" hspace="3%" vspace="20px"></a>
  <a href="https://www.fz-juelich.de/portal/EN/Home/home_node.html"><img src="./logos/logo_fzj.svg" height="50px" hspace="3%" vspace="20px"></a>
  <a href="http://www.kit.edu/english/index.php"><img src="./logos/logo_kit.svg" height="50px" hspace="3%" vspace="20px"></a>
  <a href="https://synhelion.com/"><img src="./logos/logo_synhelion.svg" height="50px" hspace="3%" vspace="20px"></a>
  <a href="https://www.helmholtz.ai/"><img src="./logos/logo_hai.svg" height="25px" hspace="3%" vspace="20px"></a>
</div>
