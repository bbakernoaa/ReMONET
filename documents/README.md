# ReMONET

ReMONET is a powerful tool for regridding and vertical interpolation of geospatial data, designed to work seamlessly with xarray objects. It supports various regridding methods, including nearest neighbor, bilinear, and conservative methods, and provides utilities for vertical regridding.

## Features

- Regridding of xarray `DataArray` and `Dataset` objects
- Vertical regridding with automatic interface layer calculation
- Support for multiple regridding methods
- Easy integration with existing xarray workflows

## Installation

To install ReMONET, clone the repository and install the dependencies:

```bash
git clone https://github.com/bbakernoaa/ReMONET.git
cd ReMONET
pip install .
