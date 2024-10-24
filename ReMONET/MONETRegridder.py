
import xarray as xr
import logging
import os
import numpy as np
import dask.array as da
from sklearn.neighbors import BallTree
from scipy.spatial.distance import cdist
from typing import Tuple, Optional, Union
from math import radians, sin, cos, sqrt, atan2
class Regridder:
    """
    A class to perform regridding of data from a source grid to a target grid.

    Attributes:
    -----------
    source_grid : xarray.Dataset
        The source grid containing 'lon' and 'lat' coordinates.
    target_grid : xarray.Dataset
        The target grid containing 'lon' and 'lat' coordinates.
    method : str
        The interpolation method to use ('linear' by default).
    esmf_weights_file : str, optional
        Path to an ESMF weights file for regridding.

    Methods:
    --------
    _get_grid_points(grid):
        Extracts and flattens the grid coordinates.
    _compute_weights():
        Computes the regridding weights using KDTree or loads from ESMF file.
    _regrid_dataarray(data):
        Regrids a DataArray using the computed weights.
    regrid(data):
        Regrids a DataArray or Dataset.
    save_weights(filename):
        Saves the computed weights to a NetCDF file.
    load_weights(filename):
        Loads the weights from a NetCDF file.
    get_weights():
        Returns the computed weights.
    """
    def __init__(self, source_grid: Union[xr.Dataset, xr.DataArray], target_grid: Union[xr.Dataset, xr.DataArray], method: str, esmf_weights_file: str = "default_file.nc") -> None:
        self.source_grid = source_grid
        self.target_grid = target_grid
        self._validate_grids()
        if method not in ['nearest', 'linear', 'bilinear', 'cubic']:  # Add supported methods here
            raise ValueError("Unsupported regridding method specified.")
        self.method = method
        self.esmf_weights_file = esmf_weights_file
        self.weights = None
        self.indices = None

    def _validate_grids(self):
        """
        Validates the presence of 'lon' and 'lat' coordinates in both the source and target grids.

        Raises:
        -------
        KeyError: If 'lon' or 'lat' coordinates are missing in either grid.

        Logs:
        -----
        Logs an error if 'lon' or 'lat' coordinates are missing in the source or target grid.
        """
        try:
            source_lon_name, source_lat_name = self.determine_coordinate_names(self.source_grid)
            target_lon_name, target_lat_name = self.determine_coordinate_names(self.target_grid)

            if source_lon_name is None or source_lat_name is None or target_lon_name is None or target_lat_name is None:
                raise KeyError("Missing longitude or latitude coordinates in source or target grid.")
        except KeyError:
            logging.error("Missing longitude or latitude coordinates in source or target grid.")

    @staticmethod
    def determine_coordinate_names(grid: xr.Dataset) -> Tuple[str, str]:
        """
        Determines the names of longitude and latitude coordinates based on attributes.

        Parameters:
        -----------
        grid : xr.Dataset
            The grid containing coordinate variables.

        Returns:
        --------
        Tuple[str, str]
            A tuple containing the names of the longitude and latitude coordinates.
        """
        possible_lon_names = ['lon', 'longitude']
        possible_lat_names = ['lat', 'latitude']

        lon_name = None
        lat_name = None

        # Check for coordinate names using a set of possible names
        for coord_name in grid.coords:
            if coord_name in possible_lon_names:
                lon_name = coord_name
            elif coord_name in possible_lat_names:
                lat_name = coord_name

        # If not found, use attributes to determine coordinate names
        if lon_name is None or lat_name is None:
            attribute_to_coordinate = {
                'longitude': 'lon',
                'latitude': 'lat',
                'degrees_east': 'lon',
                'degrees_north': 'lat'
            }

            for var_name in grid.variables:
                var_attrs = grid[var_name].attrs
                for attr_name, coord_type in attribute_to_coordinate.items():
                    if attr_name in var_attrs.values():
                        if coord_type == 'lon':
                            lon_name = var_name
                        elif coord_type == 'lat':
                            lat_name = var_name

        if lon_name is None or lat_name is None:
            raise ValueError("Unable to determine longitude and latitude based on attributes.")

        return lon_name, lat_name

    def _get_grid_points(self, grid: xr.Dataset) -> np.ndarray:
        """
        Extracts and flattens the grid coordinates using `xr.DataArray.stack`.

        Parameters:
        -----------
        grid : xarray.Dataset
            The grid containing longitude and latitude coordinates.

        Returns:
        --------
        np.ndarray
            Flattened array of grid points.
        """
        lon_name, lat_name = self.determine_coordinate_names(grid)

        lon = np.radians(grid[lon_name].values)
        lat = np.radians(grid[lat_name].values)

        # Convert to 3D Cartesian coordinates
        x = np.cos(lat) * np.cos(lon)
        y = np.cos(lat) * np.sin(lon)
        z = np.sin(lat)

        return np.column_stack((x, y, z))

    @staticmethod
    def latlon_to_cartesian(lat, lon):
        # Earth's radius in meters
        R = 6371000

        # Convert lat/lon from degrees to radians
        lat_rad = np.deg2rad(lat)
        lon_rad = np.deg2rad(lon)

        # Cartesian coordinates
        x = R * np.cos(lat_rad) * np.cos(lon_rad)
        y = R * np.cos(lat_rad) * np.sin(lon_rad)
        z = R * np.sin(lat_rad)

        return x, y, z

    def generate_regridding_weights_cartesian(self, k=4):
        if 'lon' not in self.source_grid or 'lat' not in self.source_grid or 'lon' not in self.target_grid or 'lat' not in self.target_grid:
            raise KeyError("Missing 'lon' or 'lat' coordinates in the source or target grid.")

        source_lon_name, source_lat_name = self.determine_coordinate_names(self.source_grid)
        target_lon_name, target_lat_name = self.determine_coordinate_names(self.target_grid)

        # Convert source and target lat/lon to Cartesian coordinates and create points array
        source_points = self.convert_latlon_to_cartesian_points(self.source_grid, source_lat_name, source_lon_name)
        target_points = self.convert_latlon_to_cartesian_points(self.target_grid, target_lat_name, target_lon_name)

        # Initialize BallTree with Euclidean distance metric
        tree = BallTree(source_points, metric='euclidean')

        # Find k-nearest neighbors and their distances
        dist, ind = tree.query(target_points, k=k)

        # Calculate weights based on distances
        with np.errstate(divide='ignore'):  # Handle potential division by zero
            weights = 1 / dist
        weights /= np.linalg.norm(weights, axis=1)[:, np.newaxis]  # Normalize the weights

        # Convert arrays to Dask arrays for parallel processing
        self.weights = da.from_array(weights, chunks='auto')
        self.indices = da.from_array(ind, chunks='auto')

    @staticmethod
    def reshape_with_lat_lon(data, lat_dim='lat', lon_dim='lon'):
        # Extract the shape of the data
        data_shape = data.shape
        dim_names = data.dims

        # Find the positions of lat and lon dimensions
        lat_pos = dim_names.index(lat_dim)
        lon_pos = dim_names.index(lon_dim)

        # Reshape data
        other_dims = [d for d in data.dims if d not in [lat_dim, lon_dim]]
        new_shape = (data.shape[lat_pos] * data.shape[lon_pos],) + tuple(data.shape[d] for d in other_dims)
        data_flat = data.transpose(lat_dim, lon_dim, *other_dims).values.reshape(new_shape)

        return data_flat
    @staticmethod
    def apply_weights(data_flat: np.ndarray, weights: np.ndarray, indices: np.ndarray) -> np.ndarray:
        # Validate input types
        if not isinstance(data_flat, np.ndarray) or not isinstance(weights, np.ndarray) or not isinstance(indices, np.ndarray):
            raise TypeError("Inputs must be numpy arrays.")

        # Validate indices range
        if np.any(indices < 0) or np.any(indices >= len(data_flat)):
            raise ValueError("Indices are out of range.")

        # Validate input shapes
        print(data_flat.shape, weights.shape, indices.shape)
        if data_flat.shape[-1] != weights.shape[0] or weights.shape[1] != indices.shape[1]:
            raise ValueError("Input data shapes are not compatible for regridding.")

        reshaped_data = data_flat[..., np.newaxis]
        resampled = np.einsum('...ij,...j->...i', reshaped_data[..., indices], weights)
        return resampled

    def resample_data(self, data, k=4):

        source_lon_name, source_lat_name = self.determine_coordinate_names(self.source_grid)
        target_lon_name, target_lat_name = self.determine_coordinate_names(self.target_grid)

        self.generate_regridding_weights_cartesian(k=k)

        src_dims = [dim for dim in self.source_grid[source_lon_name].dims]
        if len(src_dims) == 1:
            dim1 = src_dims[0]
            dim2 = src_dims[0]
        else:
            dim1 = src_dims[0]
            dim2 = src_dims[1]
        tgt_dims = [dim for dim in self.target_grid[target_lon_name].dims]

        weights_data = self.weights.compute()
        indices_data = self.indices.compute()

        # Validate indices range
        if np.any(indices_data < 0) or np.any(indices_data >= len(data)):
            print(np.min(indices_data), np.max(indices_data), len(data))
            logging.warning("Some indices are out of range. Adjusting or skipping regridding for those points.")

        try:
            data_flat = self.reshape_with_lat_lon(data, lat_dim=dim1, lon_dim=dim2)
            resampled_data = xr.apply_ufunc(
                self.apply_weights, data_flat, weights_data, indices_data,
                input_core_dims=[src_dims, ['points', 'k'], ['points', 'k']],
                output_core_dims=[tgt_dims],
                vectorize=True, dask='parallelized',
                output_dtypes=[data_flat.dtype],
            )
        except Exception as e:
            logging.error(f"Error during xr.apply_ufunc: {e}")
            raise

        resampled_data = resampled_data.transpose(*tgt_dims, ...)
        resampled_data.coords[target_lat_name] = self.target_grid[target_lat_name]
        resampled_data.coords[target_lon_name] = self.target_grid[target_lon_name]

        return resampled_data

    @staticmethod
    def convert_latlon_to_cartesian_points(grid, lat_name, lon_name):
        # Convert lat/lon to Cartesian coordinates
        x, y, z = Regridder.latlon_to_cartesian(grid[lat_name].values.ravel(), grid[lon_name].values.ravel())
        # Create points array
        points = np.vstack([x, y, z]).T
        return points