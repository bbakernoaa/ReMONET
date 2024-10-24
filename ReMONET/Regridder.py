import xarray as xr
import numpy as np
import esmpy
from typing import Optional, List, Tuple, Union
import dask
# Removed unused import 'dask.array as da'

try:
    import cf_xarray
    CF_XARRAY_AVAILABLE = True
except ImportError:
    CF_XARRAY_AVAILABLE = False

class Regridder:
    def __init__(self, source: Union[xr.DataArray, xr.Dataset], target: Union[xr.DataArray, xr.Dataset], method: str = 'conservative',
                 weight_file: Optional[str] = None, locstream: Optional[esmpy.LocStream] = None):
        """
        Initialize the Regridder class.

        Parameters:
        source (Union[xr.DataArray, xr.Dataset]): Source data array or dataset.
        target (Union[xr.DataArray, xr.Dataset]): Target data array or dataset.
        method (str): Regridding method ('conservative', 'bilinear', 'nearest_s2d', 'patch', 'conservative2nd', etc.).
        weight_file (Optional[str]): Path to the ESMF weight file.
        locstream (Optional[esmpy.LocStream]): ESMF LocStream for observational data.
        """
        self.source = source
        self.target = target
        self.method = method
        self.weight_file = weight_file
        self.locstream = locstream
        self.source_lat, self.source_lon = self.get_lat_lon_vars(source)
        self.target_lat, self.target_lon = self.get_lat_lon_vars(target)
        self.lat_dim, self.lon_dim = self.get_lat_lon_dims(self.source_lat, self.source_lon)
        self.extra_dims = [dim for dim in source.dims if dim not in [self.lat_dim, self.lon_dim]]

    def get_lat_lon_vars(self, ds: Union[xr.DataArray, xr.Dataset]) -> Tuple[xr.DataArray, xr.DataArray]:
        """
        Get latitude and longitude variables from the dataset using CF conventions or units attributes.

        Parameters:
        ds (Union[xr.DataArray, xr.Dataset]): Input dataset or data array.

        Returns:
        Tuple[xr.DataArray, xr.DataArray]: Latitude and longitude data arrays.
        """
        if CF_XARRAY_AVAILABLE:
            lat_var = ds.cf['latitude']
            lon_var = ds.cf['longitude']
        else:
            lat_var = self.find_coord_by_units(ds, 'degrees_north')
            lon_var = self.find_coord_by_units(ds, 'degrees_east')
        return lat_var, lon_var

    def find_coord_by_units(self, ds: Union[xr.DataArray, xr.Dataset], units: str) -> xr.DataArray:
        """
        Find coordinate variable by units attribute.

        Parameters:
        ds (Union[xr.DataArray, xr.Dataset]): Input dataset or data array.
        units (str): Units attribute to search for.

        Returns:
        xr.DataArray: Coordinate variable with the specified units.
        """
        for var in ds.coords:
            if 'units' in ds[var].attrs and ds[var].attrs['units'] == units:
                return ds[var]
        raise ValueError(f"No coordinate with units '{units}' found.")

    def get_lat_lon_dims(self, lat: xr.DataArray, lon: xr.DataArray) -> Tuple[str, str]:
        """
        Get the dimension names for latitude and longitude coordinates.

        Parameters:
        lat (xr.DataArray): Latitude data array.
        lon (xr.DataArray): Longitude data array.

        Returns:
        Tuple[str, str]: Dimension names for latitude and longitude.
        """
        lat_dim = lat.dims[0]
        lon_dim = lon.dims[0]
        return lat_dim, lon_dim

    def create_locstream_from_xarray(self, ds: Union[xr.DataArray, xr.Dataset], lat_coord: str = 'lat', lon_coord: str = 'lon',
                                     extra_dims: Optional[List[str]] = None) -> esmpy.LocStream:
        """
        Create an ESMF LocStream from an xarray dataset or data array.

        Parameters:
        ds (Union[xr.DataArray, xr.Dataset]): Input dataset or data array.
        lat_coord (str): Name of the latitude coordinate.
        lon_coord (str): Name of the longitude coordinate.
        extra_dims (Optional[List[str]]): List of extra dimensions.

        Returns:
        esmpy.LocStream: ESMF LocStream object.
        """
        lat, lon = self.get_lat_lon_vars(ds)
        common_dim = list(set(lat.dims) & set(lon.dims))[0]

        if extra_dims is None:
            extra_dims = [dim for dim in ds.dims if dim != common_dim]

        flattened_data = ds.stack(points=extra_dims + [common_dim])

        locstream = esmpy.LocStream(len(flattened_data.points))
        locstream["ESMF:Lon"] = lat[flattened_data[common_dim].values].values
        locstream["ESMF:Lat"] = lon[flattened_data[common_dim].values].values

        for dim in extra_dims:
            locstream[dim] = flattened_data[dim].values

        return locstream

    def regrid_slice(self, source_slice: xr.DataArray, target_slice: xr.DataArray) -> xr.DataArray:
        """
        Regrid a slice of the data.

        Parameters:
        source_slice (xr.DataArray): Source data slice.
        target_slice (xr.DataArray): Target data slice.

        Returns:
        xr.DataArray: Regridded data slice.
        """
        if self.method == 'nearest_neighbor':
            return self.nearest_neighbor_regrid(source_slice, target_slice)
        elif self.weight_file:
            return self.regrid_with_weights(target_slice)
        else:
            return self.regrid_without_weights(source_slice, target_slice)

    def regrid_with_weights(self, target_chunk: xr.DataArray) -> xr.DataArray:
        """
        Regrid using an ESMF weight file.

        Parameters:
        target_chunk (xr.DataArray): Target data chunk.

        Returns:
        xr.DataArray: Regridded data chunk.
        """
        # Assuming self.source_lat and self.source_lon are xarray.DataArray
        source_lat = self.source_lat.values.flatten()
        source_lon = self.source_lon.values.flatten()
        target_lat = self.target_lat.values.flatten()
        target_lon = target_chunk[self.lon_dim].values.flatten()

        # Create the source and destination grids
        source_grid = esmpy.Grid(np.array([source_lat, source_lon]), staggerloc=esmpy.StaggerLoc.CENTER)
        target_grid = esmpy.Grid(np.array([target_lat, target_lon]), staggerloc=esmpy.StaggerLoc.CENTER)

        # Create the source and destination fields
        srcfield = esmpy.Field(source_grid)
        dstfield = esmpy.Field(target_grid)

        regrid = esmpy.RegridFromFile(srcfield=srcfield, dstfield=dstfield, filename=self.weight_file)
        regridded_chunk = regrid(srcfield, dstfield)
        return regridded_chunk.data

    def regrid_without_weights(self, source_slice: xr.DataArray, target_chunk: xr.DataArray) -> xr.DataArray:
        """
        Regrid without using an ESMF weight file.

        Parameters:
        source_slice (xr.DataArray): Source data slice.
        target_chunk (xr.DataArray): Target data chunk.

        Returns:
        xr.DataArray: Regridded data chunk.
        """
        target_coords = np.array(np.meshgrid(target_chunk[self.lat_dim].values, target_chunk[self.lon_dim].values, indexing='ij')).reshape(2, -1).T
        source_coords = np.array(np.meshgrid(self.source_lat.values, self.source_lon.values, indexing='ij')).reshape(2, -1).T

        if self.method == 'conservative':
            regrid_method = esmpy.RegridMethod.CONSERVE
        elif self.method == 'bilinear':
            regrid_method = esmpy.RegridMethod.BILINEAR
        elif self.method == 'nearest_s2d':
            regrid_method = esmpy.RegridMethod.NEAREST_STOD
        elif self.method == 'patch':
            regrid_method = esmpy.RegridMethod.PATCH
        elif self.method == 'conservative2nd':
            regrid_method = esmpy.RegridMethod.CONSERVE_2ND
        else:
            raise ValueError(f"Unsupported regridding method: {self.method}")

        srcfield = esmpy.Field(esmpy.Grid(np.array([self.source_lat.values, self.source_lon.values]), staggerloc=esmpy.StaggerLoc.CENTER))
        dstfield = esmpy.Field(esmpy.Grid(np.array([target_chunk[self.lat_dim].values, target_chunk[self.lon_dim].values]), staggerloc=esmpy.StaggerLoc.CENTER))
        regrid = esmpy.Regrid(srcfield, dstfield, regrid_method=regrid_method)
        regridded_chunk = regrid(srcfield, dstfield)
        return regridded_chunk.data

    def create_weight_file(self) -> str:
        """
        Create the ESMF weight file using dask for parallel processing.

        Returns:
        str: Path to the created weight file.
        """
        def compute_weights():
            # Flatten the latitude and longitude arrays to ensure they are 1-dimensional
            source_lat = self.source_lat.values.flatten()
            source_lon = self.source_lon.values.flatten()
            target_lat = self.target_lat.values.flatten()
            target_lon = self.target_lon.values.flatten()

            # Create the source and destination grids
            source_grid = esmpy.Grid(np.array([source_lat, source_lon]), staggerloc=esmpy.StaggerLoc.CENTER)
            target_grid = esmpy.Grid(np.array([target_lat, target_lon]), staggerloc=esmpy.StaggerLoc.CENTER)

            # Create the source and destination fields
            srcfield = esmpy.Field(source_grid)
            dstfield = esmpy.Field(target_grid)

            # Create the regrid object and generate weights
            regrid = esmpy.Regrid(srcfield=srcfield, dstfield=dstfield, regrid_method=self.method, filename=self.weight_file)
            regrid(srcfield, dstfield)  # Perform the regridding to generate weights

            return regrid

        weights = dask.compute(compute_weights)
        return weights