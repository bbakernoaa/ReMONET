import xarray as xr
import numpy as np
from scipy.spatial import cKDTree
import dask.array as da
from typing import Tuple
from dask import delayed
import dask

class KDTreeRegrid:
    def __init__(self, source: xr.DataArray, target: xr.DataArray, k: int = 4):
        self.source = source
        self.target = target
        self.k = k
        self.lat_dim, self.lon_dim = self.find_lat_lon_dims(source)
        self.lat_tgt, self.lon_tgt = self.find_lat_lon_coords(target)

    @staticmethod
    def find_lat_lon_dims(data: xr.DataArray) -> Tuple[str, str]:
        lat_dim = None
        lon_dim = None
        for dim in data.dims:
            if 'lat' in dim.lower() or 'latitude' in dim.lower():
                lat_dim = dim
            if 'lon' in dim.lower() or 'longitude' in dim.lower():
                lon_dim = dim
        if lat_dim is None or lon_dim is None:
            raise ValueError("Latitude and/or Longitude dimensions not found in the data")
        return lat_dim, lon_dim

    @staticmethod
    def find_lat_lon_coords(data: xr.DataArray) -> Tuple[np.ndarray, np.ndarray]:
        lat_coord = None
        lon_coord = None
        for coord in data.coords:
            if 'lat' in coord.lower() or 'latitude' in coord.lower():
                lat_coord = data.coords[coord].values
            if 'lon' in coord.lower() or 'longitude' in coord.lower():
                lon_coord = data.coords[coord].values
        if lat_coord is None or lon_coord is None:
            raise ValueError("Latitude and/or Longitude coordinates not found in the data")
        return lat_coord, lon_coord

    @staticmethod
    def latlon_to_cartesian(lat: np.ndarray, lon: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        R = 6371000  # Earth's radius in meters
        lat_rad = np.deg2rad(lat)
        lon_rad = np.deg2rad(lon)
        x = R * np.cos(lat_rad) * np.cos(lon_rad)
        y = R * np.cos(lat_rad) * np.sin(lon_rad)
        z = R * np.sin(lat_rad)
        return x, y, z

    def generate_weights_indices_dask(self) -> Tuple[da.Array, da.Array]:
        x_src, y_src, z_src = self.latlon_to_cartesian(self.source[self.lat_dim].values.ravel(), self.source[self.lon_dim].values.ravel())
        x_tgt, y_tgt, z_tgt = self.latlon_to_cartesian(self.lat_tgt.ravel(), self.lon_tgt.ravel())

        source_points = np.vstack([x_src, y_src, z_src]).T
        target_points = np.vstack([x_tgt, y_tgt, z_tgt]).T

        num_workers = dask.config.get('scheduler')['num_workers']
        chunks = len(target_points) // num_workers if num_workers else len(target_points)
        dask_tasks = [delayed(self.parallel_weight_generation)(source_points, target_points[i:i + chunks], self.k) 
                      for i in range(0, len(target_points), chunks)]
        results = dask.compute(*dask_tasks)

        dist_list, ind_list = zip(*results)
        dist_dask = da.from_array(np.vstack(dist_list), chunks=(chunks, self.k))
        ind_dask = da.from_array(np.vstack(ind_list), chunks=(chunks, self.k))

        def normalize_weights(dist):
            weights = 1 / dist
            weights /= weights.sum(axis=1, keepdims=True)
            return weights

        weights_dask = da.map_blocks(normalize_weights, dist_dask, dtype=float)

        return weights_dask, ind_dask

    @staticmethod
    def parallel_weight_generation(source_points: np.ndarray, target_points: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        tree = cKDTree(source_points)
        dist, ind = tree.query(target_points, k=k)
        return dist, ind

    @staticmethod
    def resample_block(data_block: np.ndarray, weights: da.Array, indices: da.Array) -> np.ndarray:
        reshaped_block = data_block.reshape((data_block.shape[0], -1))
        resampled_block = np.sum(reshaped_block[:, indices] * weights[:, :, np.newaxis], axis=1)
        return resampled_block

    def resample_data_dask(self) -> xr.DataArray:
        weights_dask, indices_dask = self.generate_weights_indices_dask()

        data_dask = self.source.chunk({self.lat_dim: self.source[self.lat_dim].size//2, self.lon_dim: self.source[self.lon_dim].size//2})

        resampled_data = data_dask.map_blocks(
            self.resample_block, weights_dask, indices_dask,
            dtype=self.source.dtype, drop_axis=[0, 1]
        )

        resampled_data = xr.DataArray(
            resampled_data, dims=[self.lat_dim, self.lon_dim] + list(self.source.dims[2:]), 
            coords={self.lat_dim: self.lat_tgt[:, 0], self.lon_dim: self.lon_tgt[0, :]}
        )

        return resampled_data

    def save_weights(self, weights: da.Array, indices: da.Array, filename: str):
        """
        Save the weights and indices to a NetCDF file.

        Parameters:
        - weights (da.Array): The weights array.
        - indices (da.Array): The indices array.
        - filename (str): The filename to save the NetCDF file.
        """
        weights = weights.compute()
        indices = indices.compute()
        ds = xr.Dataset({
            'weights': (('target_points', 'k'), weights),
            'indices': (('target_points', 'k'), indices)
        })
        ds.to_netcdf(filename)

    def load_weights(self, filename: str) -> Tuple[da.Array, da.Array]:
        """
        Load the weights and indices from a NetCDF file.

        Parameters:
        - filename (str): The filename of the NetCDF file.

        Returns:
        - Tuple[da.Array, da.Array]: The weights and indices arrays.
        """
        ds = xr.open_dataset(filename)
        weights = da.from_array(ds['weights'].values, chunks=ds['weights'].shape)
        indices = da.from_array(ds['indices'].values, chunks=ds['indices'].shape)
        return weights, indices

# Example usage:
lat_src = np.array([[10, 20], [30, 40]])  # Source latitudes (2D array)
lon_src = np.array([[30, 40], [50, 60]])  # Source longitudes (2D array)
lat_tgt = np.array([[15, 25], [35, 45]])  # Target latitudes (2D array)
lon_tgt = np.array([[35, 45], [55, 65]])  # Target longitudes (2D array)

data_2d = xr.DataArray(np.random.rand(2, 2), dims=["lat", "lon"], coords={"lat": lat_src[:, 0], "lon": lon_src[0, :]})
data_3d = xr.DataArray(np.random.rand(2, 2, 3), dims=["lat", "lon", "time"], coords={"lat": lat_src[:, 0], "lon": lon_src[0, :], "time": [1, 2, 3]})
data_4d = xr.DataArray(np.random.rand(2, 2, 3, 4), dims=["lat", "lon", "z", "time"], coords={"lat": lat_src[:, 0], "lon": lon_src[0, :], "z": [0, 1, 2], "time": [1, 2, 3, 4]})

regridder = KDTreeRegrid(data_2d, data_2d)
weights, indices = regridder.generate_weights_indices_dask()
regridder.save_weights(weights, indices, 'weights_indices_2d.nc')
loaded_weights, loaded_indices = regridder.load_weights('weights_indices_2d.nc')
resampled_data_2d = regridder.resample_data_dask()
print(resampled_data_2d)

regridder = KDTreeRegrid(data_3d, data_3d)
weights, indices = regridder.generate_weights_indices_dask()
regridder.save_weights(weights, indices, 'weights_indices_3d.nc')
loaded_weights, loaded_indices = regridder.load_weights('weights_indices_3d.nc')
resampled_data_3d = regridder.resample_data_dask()
print(resampled_data_3d)

regridder = KDTreeRegrid(data_4d, data_4d)
weights, indices = regridder.generate_weights_indices_dask()
regridder.save_weights(weights, indices, 'weights_indices_4d.nc')
loaded_weights, loaded_indices = regridder.load_weights('weights_indices_4d.nc')
resampled_data_4d = regridder.resample_data_dask()
print(resampled
