import numpy as np
import xarray as xr
import dask.array as da
from sklearn.neighbors import KNeighborsRegressor

class Regridder:
    """
    A class to perform regridding of data from a source grid to a target grid using KNeighborsRegressor.

    Attributes:
    -----------
    source_grid : xarray.Dataset
        The source grid containing 'lon' and 'lat' coordinates.
    target_grid : xarray.Dataset
        The target grid containing 'lon' and 'lat' coordinates.
    n_neighbors : int
        Number of neighbors to use for KNeighborsRegressor.
    weights : str
        Weight function used in prediction ('uniform' or 'distance').

    Methods:
    --------
    _get_grid_points(grid):
        Extracts and flattens the grid coordinates.
    _train_model(data):
        Trains the KNeighborsRegressor on the source grid data.
    _regrid_dataarray(data):
        Regrids a DataArray using the trained model.
    regrid(data):
        Regrids a DataArray or Dataset.
    """

    def __init__(self, source_grid, target_grid, n_neighbors=5, weights='uniform'):
        """
        Initializes the Regridder with source and target grids, number of neighbors, and weight function.

        Parameters:
        -----------
        source_grid : xarray.Dataset
            The source grid containing 'lon' and 'lat' coordinates.
        target_grid : xarray.Dataset
            The target grid containing 'lon' and 'lat' coordinates.
        n_neighbors : int
            Number of neighbors to use for KNeighborsRegressor.
        weights : str
            Weight function used in prediction ('uniform' or 'distance').
        """
        self.source_grid = source_grid
        self.target_grid = target_grid
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.model = None

    def _get_grid_points(self, grid):
        """
        Extracts and flattens the grid coordinates.

        Parameters:
        -----------
        grid : xarray.Dataset
            The grid containing 'lon' and 'lat' coordinates.

        Returns:
        --------
        np.ndarray
            Flattened array of grid points.
        """
        lon = grid['lon'].values
        lat = grid['lat'].values
        return np.array([lon.flatten(), lat.flatten()]).T

    def _train_model(self, data):
        """
        Trains the KNeighborsRegressor on the source grid data.

        Parameters:
        -----------
        data : xarray.DataArray
            The data to be used for training.
        """
        source_points = self._get_grid_points(self.source_grid)
        data_flat = data.values.flatten()
        self.model = KNeighborsRegressor(n_neighbors=self.n_neighbors, weights=self.weights)
        self.model.fit(source_points, data_flat)

    def _regrid_dataarray(self, data):
        """
        Regrids a DataArray using the trained model.

        Parameters:
        -----------
        data : xarray.DataArray
            The data to be regridded.

        Returns:
        --------
        xarray.DataArray
            The regridded data.
        """
        if self.model is None:
            self._train_model(data)
        
        target_points = self._get_grid_points(self.target_grid)
        regridded_data = self.model.predict(target_points)
        regridded_data = regridded_data.reshape(self.target_grid['lon'].shape)
        
        regridded_da = xr.DataArray(regridded_data, coords=self.target_grid.coords, dims=self.target_grid.dims)
        
        # Copy attributes from the original data
        regridded_da.attrs = data.attrs
        
        return regridded_da

    def regrid(self, data):
        """
        Regrids a DataArray or Dataset.

        Parameters:
        -----------
        data : xarray.DataArray or xarray.Dataset
            The data to be regridded.

        Returns:
        --------
        xarray.DataArray or xarray.Dataset
            The regridded data.
        """
        if isinstance(data, xr.DataArray):
            return self._regrid_dataarray(data)
        elif isinstance(data, xr.Dataset):
            regridded_vars = {var: self._regrid_dataarray(data[var]) for var in data.data_vars}
            regridded_ds = xr.Dataset(regridded_vars, coords=self.target_grid.coords)
            
            # Copy attributes from the original dataset
            regridded_ds.attrs = data.attrs
            
            return regridded_ds
        else:
            raise TypeError("Input data must be an xarray DataArray or Dataset")

# Example usage
source_grid = xr.Dataset({'lon': (['y', 'x'], np.random.rand(10, 10)),
                          'lat': (['y', 'x'], np.random.rand(10, 10))})
target_grid = xr.Dataset({'lon': (['y', 'x'], np.linspace(0, 1, 20).reshape(4, 5)),
                          'lat': (['y', 'x'], np.linspace(0, 1, 20).reshape(4, 5))})
data = xr.DataArray(np.random.rand(10, 10), dims=['y', 'x'], attrs={'units': 'K', 'description': 'Sample data'})

regridder = Regridder(source_grid, target_grid, n_neighbors=5, weights='distance')
regridded_data = regridder.regrid(data)
print(regridded_data)
