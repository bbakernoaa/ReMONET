# import xarray as xr
# # from .Regridder import Regridder  # Adjust the import based on your actual module structure
# # from .VertRegridder import VerticalRegridder  # Adjust the import based on your actual module structure
# # from . import grid

# @xr.register_dataarray_accessor("custom")
# class DataArrayRegrid:
#     def __init__(self, xarray_obj: xr.DataArray):
#         """
#         Custom accessor for xarray DataArray objects.

#         Parameters
#         ----------
#         xarray_obj : xr.DataArray
#             The xarray DataArray object to which this accessor is attached.
#         """
#         if not isinstance(xarray_obj, xr.DataArray):
#             raise TypeError("Expected an xarray DataArray object.")
#         self._obj = xarray_obj

#     def regrid(self, target_grid: xr.DataArray, method: str = 'bilinear', **kwargs) -> xr.DataArray:
#         """
#         Regrid the DataArray to a target grid.

#         Parameters
#         ----------
#         target_grid : xr.DataArray
#             The target grid to which the DataArray will be regridded.
#         method : str, optional
#             The regridding method to use (default is 'bilinear').
#         kwargs : dict
#             Additional arguments to pass to the regridding function.

#         Returns
#         -------
#         xr.DataArray
#             The regridded DataArray.
#         """
#         try:
#             return Regridder(self._obj, target_grid, method, **kwargs).regrid()
#         except Exception as e:
#             raise RuntimeError(f"Regridding failed: {e}")

#     def vert_regrid(self, target_levels: xr.DataArray, method: str = 'linear', **kwargs) -> xr.DataArray:
#         """
#         Vertically regrid the DataArray to target levels.

#         Parameters
#         ----------
#         target_levels : xr.DataArray
#             The target vertical levels to which the DataArray will be regridded.
#         method : str, optional
#             The vertical regridding method to use (default is 'linear').
#         **kwargs : dict
#             Additional keyword arguments to pass to the vertregridder.

#         Returns
#         -------
#         xr.DataArray
#             The vertically regridded DataArray.
#         """
#         try:
#             return VerticalRegridder(self._obj, target_levels, method, **kwargs).regrid()
#         except Exception as e:
#             raise RuntimeError(f"Vertical regridding failed: {e}")

# @xr.register_dataset_accessor("mregrid")
# class DatasetRegrid:
#     def __init__(self, xarray_obj: xr.Dataset):
#         """
#         Custom accessor for xarray Dataset objects.

#         Parameters
#         ----------
#         xarray_obj : xr.Dataset
#             The xarray Dataset object to which this accessor is attached.
#         """
#         if not isinstance(xarray_obj, xr.Dataset):
#             raise TypeError("Expected an xarray Dataset object.")
#         self._obj = xarray_obj

#     def regrid(self, target_grid: xr.DataArray, method: str = 'bilinear', **kwargs) -> xr.Dataset:
#         """
#         Regrid the Dataset to a target grid.

#         Parameters
#         ----------
#         target_grid : xr.DataArray
#             The target grid to which the Dataset will be regridded.
#         method : str, optional
#             The regridding method to use (default is 'bilinear').
#         kwargs : dict
#             Additional arguments to pass to the regridding function.

#         Returns
#         -------
#         xr.Dataset
#             The regridded Dataset.
#         """
#         regridded_data = {var_name: Regridder(da, target_grid, method, **kwargs).regrid() for var_name, da in self._obj.data_vars.items()}
#         return xr.Dataset(regridded_data, attrs=self._obj.attrs)

#     def vert_regrid(self, target_levels: xr.DataArray, method: str = 'linear', **kwargs) -> xr.Dataset:
#         """
#         Vertically regrid the Dataset to target levels.

#         Parameters
#         ----------
#         target_levels : xr.DataArray
#             The target vertical levels to which the Dataset will be regridded.
#         method : str, optional
#             The vertical regridding method to use (default is 'linear').
#         **kwargs : dict
#             Additional keyword arguments to pass to the vertregridder.

#         Returns
#         -------
#         xr.Dataset
#             The vertically regridded Dataset.
#         """
#         regridded_data = {var_name: VerticalRegridder(da, target_levels, method, **kwargs).regrid() for var_name, da in self._obj.data_vars.items()}
#         return xr.Dataset(regridded_data, attrs=self._obj.attrs)