import xarray as xr
import numpy as np
import numpy.ma as ma
from scipy.spatial import cKDTree
from scipy.interpolate import NearestNDInterpolator, LinearNDInterpolator
import dask.array as da
from typing import Tuple, Union
from dask import delayed
import dask
from scipy import stats

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

        def normalize_weights(dist: np.ndarray) -> np.ndarray:
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
    def resample_block(data_block: np.ndarray, weights: da.Array, indices: da.Array, statistic: str) -> np.ndarray:
        reshaped_block = data_block.reshape((data_block.shape[0], -1))
        if statistic == 'mean':
            resampled_block = np.sum(reshaped_block[:, indices] * weights[:, :, np.newaxis], axis=1)
        elif statistic == 'median':
            resampled_block = np.median(reshaped_block[:, indices], axis=1)
        elif statistic == 'mode':
            mode_result = stats.mode(reshaped_block[:, indices], axis=1, nan_policy='omit')
            resampled_block = mode_result.mode
        elif statistic == 'std':
            resampled_block = np.std(reshaped_block[:, indices], axis=1)
        else:
            raise ValueError("Unsupported statistic. Use 'mean', 'median', 'mode', or 'std'.")
        return resampled_block

    def resample_data_dask(self, statistic: str = 'mean') -> xr.DataArray:
        weights_dask, indices_dask = self.generate_weights_indices_dask()

        data_dask = self.source.chunk({self.lat_dim: self.source[self.lat_dim].size//2, self.lon_dim: self.source[self.lon_dim].size//2})

        def handle_nan(data_block: np.ndarray, weights: da.Array, indices: da.Array) -> np.ndarray:
            mask = np.isnan(data_block)
            data_block = np.ma.masked_array(data_block, mask)
            resampled_block = self.resample_block(data_block, weights, indices, statistic)
            return np.ma.filled(resampled_block, np.nan)

        resampled_data = data_dask.map_blocks(
            handle_nan, weights_dask, indices_dask,
            dtype=self.source.dtype, drop_axis=[0, 1]
        )

        resampled_data = xr.DataArray(
            resampled_data, dims=[self.lat_dim, self.lon_dim] + list(self.source.dims[2:]),
            coords={self.lat_dim: self.lat_tgt[:, 0], self.lon_dim: self.lon_tgt[0, :]},
            attrs=self.source.attrs  # Retain attributes
        )

        return resampled_data

    def save_weights(self, weights: da.Array, indices: da.Array, filename: str) -> None:
        weights = weights.compute()
        indices = indices.compute()
        ds = xr.Dataset({
            'weights': (('target_points', 'k'), weights),
            'indices': (('target_points', 'k'), indices)
        })
        ds.to_netcdf(filename)

    def load_weights(self, filename: str) -> Tuple[da.Array, da.Array]:
        ds = xr.open_dataset(filename)
        weights = da.from_array(ds['weights'].values, chunks=ds['weights'].shape)
        indices = da.from_array(ds['indices'].values, chunks=ds['indices'].shape)
        return weights, indices



def interpolate_scipy(self, method: str = 'linear') -> xr.DataArray:
    x_src, y_src, z_src = self.latlon_to_cartesian(np.asarray(self.source[self.lat_dim].values).ravel(), np.asarray(self.source[self.lon_dim].values).ravel())
    x_tgt, y_tgt, z_tgt = self.latlon_to_cartesian(self.lat_tgt.ravel(), self.lon_tgt.ravel())

    source_points = np.vstack([x_src, y_src, z_src]).T
    target_points = np.vstack([x_tgt, y_tgt, z_tgt]).T

    data_values = da.from_array(np.asarray(self.source.values).ravel(), chunks=self.source.values.size//2)
    mask = ~np.isnan(data_values)
    data_values = data_values[mask]
    source_points = source_points[mask]

    # Handle masked arrays
    if isinstance(data_values, ma.MaskedArray):
        mask = ~data_values.mask
        data_values = data_values.data[mask]

    # Flatten additional dimensions in data values
    data_values_flat = data_values.reshape(data_values.shape[0], -1)

    try:
        if method == 'nearest':
            tree = cKDTree(source_points)
            _, indices = tree.query(target_points, k=1)
            interpolated_data = self.source.values.ravel()[indices].reshape(self.lat_tgt.shape)
        else:
            interpolated_values = griddata(source_points, data_values_flat, target_points, method=method)
            interpolated_data = interpolated_values.reshape(self.lat_tgt.shape)
    except ValueError as e:
        print(f"Interpolation error: {e}")
        interpolated_data = np.full(self.lat_tgt.shape, np.nan)

    if self.source.attrs is not None:
        return xr.DataArray(interpolated_data, dims=[self.lat_dim, self.lon_dim], coords={self.lat_dim: self.lat_tgt[:, 0], self.lon_dim: self.lon_tgt[0, :]}, attrs=self.source.attrs)
    else:
        return xr.DataArray(interpolated_data, dims=[self.lat_dim, self.lon_dim], coords={self.lat_dim: self.lat_tgt[:, 0], self.lon_dim: self.lon_tgt[0, :]})
# Example usage:
lat_src = np.array([[10, 20], [30, 40]])  # Source latitudes (2D array)
lon_src = np.array([[30, 40], [50, 60]])  # Source longitudes
lat_src = np.array([[10, 20], [30, 40]])  # Source latitudes (2D array)
lon_src = np.array([[30, 40], [50, 60]])  # Source longitudes (2D array)
lat_tgt = np.array([[15, 25], [35, 45]])  # Target latitudes (2D array)
lon_tgt = np.array([[35, 45], [55, 65]])  # Target longitudes (2D array)

data_2d = xr.DataArray(np.random.rand(2, 2), dims=["lat", "lon"], coords={"lat": lat_src[:, 0], "lon": lon_src[0, :]})
data_3d = xr.DataArray(np.random.rand(2, 2, 3), dims=["lat", "lon", "time"], coords={"lat": lat_src[:, 0], "lon": lon_src[0, :], "time": [1, 2, 3]})
data_4d = xr.DataArray(np.random.rand(2, 2, 3, 4), dims=["lat", "lon", "z", "time"], coords={"lat": lat_src[:, 0], "lon": lon_src[0, :], "z": [0, 1, 2], "time": [1, 2, 3, 4]})

regridder = KDTreeRegrid(data_2d, data_2d)
weights, indices = regridder.generate_weights_indices_dask()
# regridder.save_weights(weights, indices
