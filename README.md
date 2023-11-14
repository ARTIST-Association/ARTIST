![ARTIST Logo](./LOGO.svg)

# AI-enhanced differentiable Ray Tracer for Irradiationprediction in Solar Tower Dig

[![fair-software.eu]
![PyPI]
![PyPI - Downloads]
[![OpenSSF Best Practices]
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![](https://img.shields.io/badge/Python-3.6+-blue.svg)](https://www.python.org/downloads/)
[![](https://img.shields.io/badge/Contact-max.pargmann%40dlr.de-orange)](mailto:max.pargmann@dlr.de)
[![Documentation Status](https://readthedocs.org/projects/propulate/badge/?version=latest)](https://propulate.readthedocs.io/en/latest/?badge=latest)
![](./coverage.svg)

## Description

This Python program implements a differentiable ray tracer within the PyTorch Machine Learning framework. Leveraging automatic differentiation and GPU computation, it facilitates the optimization of heliostats, towers, and camera parameters within a solar field by combining gradient-based optimization methods with smooth parametric descriptions of heliostats.

**Key capabilities of this program include:**

- **Neural Network Driven Heliostat Calibration**: A two-layer hybrid model for most efficient heliostat calibration. It inherits a robust geometric model for pre-alignment and a neural network disturbance model, which gradually adapts its impact via regularization sweeps. On this way, high data requirements of data-centric methods are overcome while maintaining flexibility for modeling complex real-world systems. For more details see[here](https://doi.org/10.1016/j.solener.2023.111962).  

- **Surface Reconstruction and Flux Density Prediction**: Leveraging learning Non-Uniform Rational B-Splines (NURBS), the program reconstructs heliostat surfaces accurately using calibration images commonly available in solar thermal power plants. Achieving sub-millimeter accuracy in mirror reconstruction from focal spot images, contributing to improved operational safety and efficiency. The reconstructed surfaces can be used for prediction unique heliostat flux densities with the same accuracy as state of the art. For more details see[here](https://doi.org/10.21203/rs.3.rs-2554998/v1).

- **Advanced Data Set Sampling Strategies** Utilizing a time independet data set sampling strategy based on Euler angles to improve accuracy by minimizing the needed data for calibration. For more details see[here](https://doi.org/10.21203/rs.3.rs-2898838/v1).

- **Immediate Deployment**: Enabling deployment at the beginning of a solar thermal plant's operation, allowing for in-situ calibration and subsequent improvements in energy efficiencies and cost reductions.

- **Optimize Flux Density** Coming Soon


## Installation Instructions: 
run setup.py without any commands

## Folder Structure
├───artist&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;# Folder includes the file packages                   \
│   ├───io&nbsp;&nbsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;# ???                              \
│   │   └───tests&emsp;&emsp;&emsp;&emsp;&emsp;&nbsp;# includes unit tests?                       \
│   ├───physics_objects&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# includes all physical objects inside the &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;raytracing environment e.g. Heliostats or the Receiver                 \
│   │   └───heliostats                  \
│   │       ├───alignment               \
│   │       │   └───tests               \
│   │       │       └───bitmaps         \
│   │       └───surface                 \
│   │           └───facets              \
│   ├───raytracing                      \
│   ├───scenario                        \
│   │   └───light_source                \
│   │       └───tests                   \
│   └───util                            \
└───scenario_objects&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;#Szenario Objects are ment to be &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;loaded from the experiment yaml file. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;For example to define wether the sun &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;or a beamer should be loaded                    \
    └───heliostats                      

## Usage: 
Blub


## Contributing: 
See in the contribute.md file

## Credits: 
Thanks to the Helmholtz HAICU Voucher Project for boosting the start of this Project
Thanks to Helmoltz ARTIST Project four founding

## License: 
Hopefully MIT
## Documentation: 
Coming Soon