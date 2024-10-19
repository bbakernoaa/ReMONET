import xarray as xr
from .regridder import Regridder  # Adjust the import based on your actual module structure
from .vertregridder import VertRegridder  # Adjust the import based on your actual module structure

@xr.register_dataarray_accessor("custom")
class CustomDataArrayAccessor:
    def __init__(self, xarray_obj: xr.DataArray):
        """
        Custom accessor for xarray DataArray objects.

        Parameters
        ----------
        xarray_obj : xr.DataArray
            The xarray DataArray object to which this accessor is attached.
        """
        self._obj = xarray_obj

    def regrid(self, target_grid: xr.DataArray, method: str = 'bilinear', **kwargs) -> xr.DataArray:
        """
        Regrid the DataArray to a target grid.

        Parameters
        ----------
        target_grid : xr.DataArray
            The target grid to which the DataArray will be regridded.
        method : str, optional
            The regridding method to use (default is 'bilinear').
        **kwargs : dict
            Additional keyword arguments to pass to the regridder.

        Returns
        -------
        xr.DataArray
            The regridded DataArray.
        """
        regridder = Regridder(self._obj, target_grid, method, **kwargs)
        return regridder.regrid()

    def vert_regrid(self, target_levels: xr.DataArray, method: str = 'linear', **kwargs) -> xr.DataArray:
        """
        Vertically regrid the DataArray to target levels.

        Parameters
        ----------
        target_levels : xr.DataArray
            The target vertical levels to which the DataArray will be regridded.
        method : str, optional
            The vertical regridding method to use (default is 'linear').
        **kwargs : dict
            Additional keyword arguments to pass to the vertregridder.

        Returns
        -------
        xr.DataArray
            The vertically regridded DataArray.
        """
        vert_regridder = VertRegridder(self._obj, target_levels, method, **kwargs)
        return vert_regridder.regrid()

@xr.register_dataset_accessor("custom")
class CustomDatasetAccessor:
    def __init__(self, xarray_obj: xr.Dataset):
        """
        Custom accessor for xarray Dataset objects.

        Parameters
        ----------
        xarray_obj : xr.Dataset
            The xarray Dataset object to which this accessor is attached.
        """
        self._obj = xarray_obj

    def regrid(self, target_grid: xr.DataArray, method: str = 'bilinear', **kwargs) -> xr.Dataset:
        """
        Regrid the Dataset to a target grid.

        Parameters
        ----------
        target_grid : xr.DataArray
            The target grid to which the Dataset will be regridded.
        method : str, optional
            The regridding method to use (default is 'bilinear').
        **kwargs : dict
            Additional keyword arguments to pass to the regridder.

        Returns
        -------
        xr.Dataset
            The regridded Dataset.
        """
        regridder = Regridder(self._obj, target_grid, method, **kwargs)
        return regridder.regrid()

    def vert_regrid(self, target_levels: xr.DataArray, method: str = 'linear', **kwargs) -> xr.Dataset:
        """
        Vertically regrid the Dataset to target levels.

        Parameters
        ----------
        target_levels : xr.DataArray
            The target vertical levels to which the Dataset will be regridded.
        method : str, optional
            The vertical regridding method to use (default is 'linear').
        **kwargs : dict
            Additional keyword arguments to pass to the vertregridder.

        Returns
        -------
        xr.Dataset
            The vertically regridded Dataset.
        """
        vert_regridder = VertRegridder(self._obj, target_levels, method, **kwargs)
        return vert_regridder.regrid()
