import xarray as xr
import numpy as np
import dask.array as da
from scipy.interpolate import interp1d

class VerticalRegridder:
    def __init__(self, source: xr.DataArray, target_levels: xr.DataArray, source_vertical_levels: xr.DataArray, source_vertical_interfaces: xr.DataArray, target_vertical_interfaces: xr.DataArray, method: str = 'linear', vertical_coord: str = 'height'):
        """
        Initialize the VerticalRegridder class.

        Parameters:
        source (xr.DataArray): Source data array.
        target_levels (xr.DataArray): Target vertical levels.
        source_vertical_levels (xr.DataArray): Source vertical levels.
        source_vertical_interfaces (xr.DataArray): Source vertical interface levels.
        target_vertical_interfaces (xr.DataArray): Target vertical interface levels.
        method (str): Interpolation method ('linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic', 'conservative').
        vertical_coord (str): Name of the vertical coordinate ('height', 'pressure', or 'log_pressure').
        """
        self.source = source
        self.target_levels = target_levels
        self.source_vertical_levels = source_vertical_levels
        self.source_vertical_interfaces = source_vertical_interfaces
        self.target_vertical_interfaces = target_vertical_interfaces
        self.method = method
        self.vertical_coord = vertical_coord
        self.vertical_dim = self.get_vertical_dim(source, vertical_coord)
        self.check_monotonicity(self.source_vertical_levels)
        self.check_monotonicity(self.source_vertical_interfaces)
        self.check_monotonicity(self.target_vertical_interfaces)

    def get_vertical_dim(self, ds: xr.DataArray, vertical_coord: str) -> str:
        """
        Get the dimension name for the vertical coordinate.

        Parameters:
        ds (xr.DataArray): Input data array.
        vertical_coord (str): Name of the vertical coordinate.

        Returns:
        str: Dimension name for the vertical coordinate.
        """
        for dim in ds.dims:
            if vertical_coord in ds[dim].attrs.get('units', ''):
                return dim
        raise ValueError(f"No dimension with vertical coordinate '{vertical_coord}' found.")

    def check_monotonicity(self, vertical_levels: xr.DataArray):
        """
        Check if the vertical coordinate is monotonic.

        Parameters:
        vertical_levels (xr.DataArray): Vertical levels to check.

        Raises:
        ValueError: If the vertical levels are not monotonic.
        """
        if not (np.all(np.diff(vertical_levels) > 0) or np.all(np.diff(vertical_levels) < 0)):
            raise ValueError("The vertical coordinate must be monotonic.")

    def regrid(self) -> xr.DataArray:
        """
        Perform the vertical regridding operation.

        Returns:
        xr.DataArray: Regridded data array.
        """
        regridded_data = xr.apply_ufunc(
            self.regrid_slice,
            self.source,
            input_core_dims=[[self.vertical_dim]],
            output_core_dims=[[self.vertical_dim]],
            vectorize=True,
            dask='parallelized',
            output_dtypes=[self.source.dtype]
        )
        return regridded_data

    def regrid_slice(self, source_slice: np.ndarray) -> np.ndarray:
        """
        Regrid a slice of the data.

        Parameters:
        source_slice (np.ndarray): Source data slice.

        Returns:
        np.ndarray: Regridded data slice.
        """
        source_levels = self.source_vertical_levels.values
        target_levels = self.target_levels.values

        if self.method in ['linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic']:
            interpolator = interp1d(source_levels, source_slice, kind=self.method, bounds_error=False, fill_value='extrapolate')
            regridded_values = interpolator(target_levels)
        elif self.method == 'conservative':
            regridded_values = self.conservative_regrid(source_slice)
        else:
            raise ValueError(f"Unsupported interpolation method: {self.method}")

        return regridded_values

    def conservative_regrid(self, source_slice: np.ndarray) -> np.ndarray:
        """
        Perform conservative regridding.

        Parameters:
        source_slice (np.ndarray): Source data slice.

        Returns:
        np.ndarray: Conservatively regridded data slice.
        """
        source_interfaces = self.source_vertical_interfaces.values
        target_interfaces = self.target_vertical_interfaces.values

        source_volumes = np.diff(source_interfaces)
        target_volumes = np.diff(target_interfaces)

        regridded_values = np.zeros_like(self.target_levels.values)

        for i, target_level in enumerate(self.target_levels.values):
            overlap = np.minimum(source_interfaces[1:], target_interfaces[i+1]) - np.maximum(source_interfaces[:-1], target_interfaces[i])
            overlap = np.maximum(overlap, 0)
            weights = overlap / source_volumes
            regridded_values[i] = np.sum(weights * source_slice)

        return regridded_values

# # Example usage with chunking
# source_data = xr.DataArray(
#     da.random.random((10, 5, 5), chunks=(5, 5, 5)),
#     dims=['height', 'lat', 'lon'],
#     coords={'height': np.linspace(0, 10000, 10), 'lat': np.linspace(-90, 90, 5), 'lon': np.linspace(-180, 180, 5)}
# )
# source_vertical_levels = xr.DataArray(np.linspace(0, 10000, 10), dims=['height'], coords={'height': np.linspace(0, 10000, 10)})
# source_vertical_interfaces = xr.DataArray(np.linspace(0, 10000, 11), dims=['interface'], coords={'interface': np.linspace(0, 10000, 11)})
# target_levels = xr.DataArray(np.linspace(0, 10000, 20), dims=['height'], coords={'height': np.linspace(0, 10000, 20)})
# target_vertical_interfaces = xr.DataArray(np.linspace(0, 10000, 21), dims=['interface'], coords={'interface': np.linspace(0, 10000, 21)})

# regridder = VerticalRegridder(source_data, target_levels, source_vertical_levels, source_vertical_interfaces, target_vertical_interfaces, method='conservative', vertical_coord='height')
# regridded_data = regridder.regrid()
# print(regridded_data)
