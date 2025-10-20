<p align="center">
<img src="https://raw.githubusercontent.com/ARTIST-Association/ARTIST/main/logos/artist_logo.svg" alt="logo" width="500"/>
</p>

# AI-enhanced differentiable Ray Tracer for Irradiation Prediction in Solar Tower Digital Twins

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17381222.svg)](https://doi.org/10.5281/zenodo.17381222)
![PyPI](https://img.shields.io/pypi/v/artist-csp)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![OpenSSF Best Practices](https://www.bestpractices.dev/projects/11131/badge)](https://www.bestpractices.dev/projects/11131)
[![fair-software.eu](https://img.shields.io/badge/fair--software.eu-%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8B-yellow)](https://fair-software.eu)
[![FAIR checklist badge](https://fairsoftwarechecklist.net/badge.svg)](https://fairsoftwarechecklist.net/v0.2?f=31&a=32113&i=32221&r=133)
[![](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![codecov](https://codecov.io/gh/ARTIST-Association/ARTIST/graph/badge.svg?token=AEUYvTNXz1)](https://codecov.io/gh/ARTIST-Association/ARTIST)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/ARTIST-Association/ARTIST/main.svg)](https://results.pre-commit.ci/latest/github/ARTIST-Association/ARTIST/main)
[![Documentation Status](https://readthedocs.org/projects/artist/badge/?version=latest)](https://artist.readthedocs.io/en/latest/?badge=latest)
[![](https://img.shields.io/badge/Contact-artist%40lists.kit.edu-orange?label=Contact)](artist@lists.kit.edu)


## What ``ARTIST`` can do for you

The ``ARTIST`` package provides an implementation of a differentiable ray tracer using the `PyTorch` machine-learning
framework in `Python`. Leveraging automatic differentiation and GPU computation, it facilitates the optimization of
heliostats, towers, and camera parameters within a solar field by combining gradient-based optimization methods with
smooth parametric descriptions of heliostats.

**Our contributions include:**

- **Efficient heliostat calibration:** We develop a parallelized geometric kinematic model that enables efficient
    calibration via either ray tracing-based or motor position data. This offers a flexible and robust approach to
    heliostat calibration.

- **Surface reconstruction and flux density prediction:** Leveraging learning Non-Uniform Rational B-Splines (NURBS),
  `ARTIST` reconstructs heliostat surfaces accurately using calibration images commonly available in solar thermal power plants.
  Thus, we can achieve sub-millimeter accuracy in mirror reconstruction from focal spot images, contributing to improved
  operational safety and efficiency. The reconstructed surfaces can be used for predicting unique heliostat flux densities
  with state-of-the-art accuracy. Check out [this paper](https://doi.org/10.21203/rs.3.rs-2554998/v1) for more details.

- **Immediate deployment**: `ARTIST` enables deployment at the beginning of a solar thermal plant's operation,
  allowing for in situ calibration and subsequent improvements in energy efficiencies and cost reductions.

- **Optimized flux density:** ``ARTIST`` enables flux density optimization across an entire heliostat field by optimizing
  the motor positions of all heliostats to distribute the flux optimally over the receiver.


## Installation
We heavily recommend installing the `ARTIST` package in a dedicated `Python3.10+` virtual environment. You can
install ``ARTIST`` directly from the GitHub repository via:
```bash
pip install artist
```
Alternatively, you can install ``ARTIST`` locally. To achieve this, there are two steps you need to follow:
1. Clone the `ARTIST` repository:
   ```bash
   git clone https://github.com/ARTIST-Association/ARTIST.git
   ```
2. Install the package from the main branch. There are multiple installation options available:
   - Install basic dependencies: ``pip install .``
   - Install with optional dependencies to run the tutorials:  ``pip install ."[tutorials]"``
   - Install an editable version with developer dependencies: ``pip install -e ."[dev]"``

## Structure
The ``ARTIST`` repository is structured as shown below:
```
.
├── artist # Parent package
│   ├── core # Core functionality of ARTIST, e.g. raytracing, optimizers etc.
│   ├── data_loader # Deals with loading data into ARTIST from different sources.
│   ├── field # Objects in the field, e.g. heliostats and target areas like receivers and calibration targets.
│   ├── scenario # Functionality to create and load scenarios in ARTIST.
│   ├── scene # Light sources and factors influencing the surroundings.
│   └── util
├── tests
│   ├── data
│   │   ├── field_data # Real measurements from the PAINT database and STRAL that can be used in ARTIST.
│   │   ├── scenarios # Scenarios describing an environment that can be loaded by ARTIST.
│   │   └── ...
│   ├── core
│   ├── data_loader
│   └── ...
└── tutorials # Tutorials to help you get started with ARTIST.
    ├── data # Data accessed in the tutorials.
    │   ├── paint # Real measurements from the PAINT database.
    │   ├── scenarios # Scenarios describing an environment that can be loaded by ARTIST.
    │   └── stral Real # Measurements from STRAL.
    └── ...
```

## Documentation
You can check out the full ``ARTIST`` documentation at [https://artist.readthedocs.io/en/latest/index.html](https://artist.readthedocs.io/en/latest/index.html) :rocket:!
The ``ARTIST`` documentation includes:
- Installation instructions
- Tutorials
- Some theoretical background information
- API reference

## How to contribute
Check out our [contribution guidelines](CONTRIBUTING.md) if you are interested in contributing to the `ARTIST` project :fire:.
Please also carefully check our [code of conduct](CODE_OF_CONDUCT.md) :blue_heart:.

## Acknowledgments
This work is supported by the [Helmholtz AI](https://www.helmholtz.ai/) platform grant.

-----------
<div align="center">
  <a href="https://www.dlr.de/EN/Home/home_node.html"><img src="https://raw.githubusercontent.com/ARTIST-Association/ARTIST/main/logos/logo_dlr.svg" height="50px" hspace="3%" vspace="25px"></a>
  <a href="https://www.fz-juelich.de/portal/EN/Home/home_node.html"><img src="https://raw.githubusercontent.com/ARTIST-Association/ARTIST/main/logos/logo_fzj.svg" height="50px" hspace="3%" vspace="25px"></a>
  <a href="http://www.kit.edu/english/index.php"><img src="https://raw.githubusercontent.com/ARTIST-Association/ARTIST/main/logos/logo_kit.svg" height="50px" hspace="3%" vspace="25px"></a>
  <a href="https://synhelion.com/"><img src="https://raw.githubusercontent.com/ARTIST-Association/ARTIST/main/logos/logo_synhelion.svg" height="50px" hspace="3%" vspace="25px"></a>
</div>

<div align="center">
<a href="https://www.helmholtz.ai/"><img src="https://raw.githubusercontent.com/ARTIST-Association/ARTIST/main/logos/logo_hai.svg" height="25px" hspace="3%" vspace="25px"></a>
</div>
