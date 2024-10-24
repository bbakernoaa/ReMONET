"""
Backend for ReMONET. This module wraps ESMPy's complicated API and can create
ESMPy Grid and Regrid objects only using basic numpy arrays.

General idea:

1) Only use pure numpy array in this low-level backend. xarray should only be
used in higher-level APIs which interface with this low-level backend.

2) Use simple, procedural programming here. Because ESMpy Classes are
complicated enough, building new Classes will make debugging very difficult.

3) Add some basic error checking in this wrapper level.
ESMPy is hard to debug because the program often dies in the Fortran level.
So it would be helpful to catch some common mistakes in Python level.
"""

import os
from typing import Optional, Tuple
import warnings
import numpy as np
import dask.array as da
import dask
import numpy.lib.recfunctions as nprec
try:
    import esmpy as ESMF
except ImportError:
    import ESMF


def check_f_contiguous(a: np.ndarray) -> None:
    """
    Give a warning if input array if not Fortran-ordered.

    ESMPy expects Fortran-ordered array. Passing C-ordered array will slow down
    performance due to memory rearrangement.

    Parameters
    ----------
    a : numpy array

    Returns
    -------
    numpy array
    """
    if not a.flags['F_CONTIGUOUS']:
        warnings.warn(f'Input array of shape {a.shape} is not F_CONTIGUOUS. This will affect performance.', stacklevel=2)
    return a


def warn_lat_range(lat: np.ndarray) -> None:
    """
    Give a warning if latitude is outside of [-90, 90]

    Longitude, on the other hand, can be in any range,
    since the transform is done in (x, y, z) space.

    Parameters
    ----------
    lat : numpy array
    """
    if np.any(lat > 90.0) or np.any(lat < -90.0):
        warnings.warn(f'Latitude is outside of [-90, 90]. Actual range: [{lat.min()}, {lat.max()}]')


class Grid(ESMF.Grid):
    @classmethod
    def from_xarray(cls, lon: np.ndarray, lat: np.ndarray, periodic: bool = False, mask: Optional[np.ndarray] = None) -> ESMF.Grid:
        """
        Create an ESMF.Grid object, for constructing ESMF.Field and ESMF.Regrid.

        Parameters
        ----------
        lon, lat : 2D numpy array
             Longitute/Latitude of cell centers.

             Recommend Fortran-ordering to match ESMPy internal.

             Shape should be ``(Nlon, Nlat)`` for rectilinear grid,
             or ``(Nx, Ny)`` for general quadrilateral grid.

        periodic : bool, optional
            Periodic in longitude? Default to False.
            Only useful for source grid.

        mask : 2D numpy array, optional
            Grid mask. According to the ESMF convention, masked cells
            are set to 0 and unmasked cells to 1.

            Shape should be ``(Nlon, Nlat)`` for rectilinear grid,
            or ``(Nx, Ny)`` for general quadrilateral grid.

        Returns
        -------
        grid : ESMF.Grid object
        """

        # ESMPy expects Fortran-ordered array.
        # Passing C-ordered array will slow down performance.
        for a in [lon, lat]:
            check_f_contiguous(a)
            if not a.flags['F_CONTIGUOUS']:
                raise ValueError(f'{a} must be Fortran-ordered for optimal performance.')

        warn_lat_range(lat)

        # ESMF.Grid can actually take 3D array (lon, lat, radius),
        # but regridding only works for 2D array
        if lon.ndim != 2:
            raise ValueError('Input grid must be 2D array')
        if lon.shape != lat.shape:
            raise ValueError('lon and lat must have the same shape')

        staggerloc = ESMF.StaggerLoc.CENTER  # actually just integer 0

        if periodic:
            num_peri_dims = 1
        else:
            num_peri_dims = None

        # ESMPy documentation claims that if staggerloc and coord_sys are None,
        # they will be set to default values (CENTER and SPH_DEG).
        # However, they actually need to be set explicitly,
        # otherwise grid._coord_sys and grid._staggerloc will still be None.
        grid = cls(
            np.array(lon.shape),
            staggerloc=staggerloc,
            coord_sys=ESMF.CoordSys.SPH_DEG,
            num_peri_dims=num_peri_dims,
        )

        # The grid object points to the underlying Fortran arrays in ESMF.
        # To modify lat/lon coordinates, need to get pointers to them
        lon_pointer = grid.get_coords(coord_dim=0, staggerloc=staggerloc)
        lat_pointer = grid.get_coords(coord_dim=1, staggerloc=staggerloc)

        # Use [...] to avoid overwritting the object. Only change array values.
        lon_pointer[...] = lon
        lat_pointer[...] = lat

        # Follows SCRIP convention where 1 is unmasked and 0 is masked.
        # See https://github.com/NCPP/ocgis/blob/61d88c60e9070215f28c1317221c2e074f8fb145/src/ocgis/regrid/base.py#L391-L404
        if mask is not None:
            # remove fractional values and clip to 0 or 1
            mask = np.clip(mask, 0, 1)
            # convert array type to integer (ESMF compat)
            grid_mask = mask.astype(np.int32)
            if mask.shape != lon.shape:
                raise ValueError(
                    f'mask must have the same shape as the latitude/longitude '
                    f'coordinates, got: mask.shape = {mask.shape}, lon.shape = {lon.shape}'
                )
            grid.add_item(ESMF.GridItem.MASK, staggerloc=ESMF.StaggerLoc.CENTER, from_file=False)
            grid.mask[0][:] = grid_mask

        return grid

    def get_shape(self, loc: ESMF.StaggerLoc = ESMF.StaggerLoc.CENTER) -> Tuple[int, ...]:
        """Return shape of grid for specified StaggerLoc"""
        # We cast explicitly to python's int (numpy >=2)
        return tuple(map(int, self.size[loc]))


class LocStream(ESMF.LocStream):
    @classmethod
    def from_xarray(cls, lon: np.ndarray, lat: np.ndarray) -> ESMF.LocStream:
        """
        Create an ESMF.LocStream object, for contrusting ESMF.Field and ESMF.Regrid

        Parameters
        ----------
        lon, lat : 1D numpy array
             Longitute/Latitude of cell centers.

        Returns
        -------
        locstream : ESMF.LocStream object
        """
        if len(lon) == 0 or len(lat) == 0:
            raise ValueError('Input arrays `lon` and `lat` cannot be empty')

        if len(lon.shape) > 1:
            raise ValueError('lon can only be 1d')
        if len(lat.shape) > 1:
            raise ValueError('lat can only be 1d')

        if lon.shape != lat.shape:
            raise ValueError('lon and lat shapes do not match.')

        location_count = len(lon)

        locstream = cls(location_count, coord_sys=ESMF.CoordSys.SPH_DEG)

        locstream['ESMF:Lon'] = lon.astype(np.float64)
        locstream['ESMF:Lat'] = lat.astype(np.float64)

        return locstream

    def get_shape(self) -> Tuple[int, int]:
        """
        Return the shape of the LocStream object.

        Returns
        -------
        tuple
            A tuple containing the size of the LocStream and 1.
        """
        return (self.size, 1)


def add_corner(grid: ESMF.Grid, lon_b: np.ndarray, lat_b: np.ndarray) -> None:
    """
    Add corner information to ESMF.Grid for conservative regridding.

    Not needed for other methods like bilinear or nearest neighbour.

    Parameters
    ----------
    grid : ESMF.Grid object
        Generated by ``Grid.from_xarray()``. Will be modified in-place.

    lon_b, lat_b : 2D numpy array
        Longitute/Latitude of cell corner
        Recommend Fortran-ordering to match ESMPy internal.
        Shape should be ``(Nlon+1, Nlat+1)``, or ``(Nx+1, Ny+1)``
    """

    staggerloc = ESMF.StaggerLoc.CORNER  # actually just integer 3

    warnings.warn('\n'.join([f'Input array of shape {a.shape} is not F_CONTIGUOUS. This will affect performance.' for a in [lon_b, lat_b] if not a.flags['F_CONTIGUOUS']]), stacklevel=2)

    warn_lat_range(lat_b)

    try:
        if lon_b.ndim != 2:
            raise ValueError('Input grid must be 2D array')
        if lon_b.shape != lat_b.shape:
            raise ValueError('lon_b and lat_b must have the same shape')
        if not np.array_equal(lon_b.shape, grid.max_index + 1):
            raise ValueError('lon_b should be size (Nx+1, Ny+1)')
        if grid.num_peri_dims != 0 or grid.periodic_dim is not None:
            raise ValueError('Cannot add corner for periodic grid')
    except ValueError as e:
        print(f'Error: {e}')

    grid.add_coords(staggerloc=staggerloc)

    lon_b_pointer = grid.get_coords(coord_dim=0, staggerloc=staggerloc)
    lat_b_pointer = grid.get_coords(coord_dim=1, staggerloc=staggerloc)

    lon_b_pointer[...] = lon_b
    lat_b_pointer[...] = lat_b


class Mesh(ESMF.Mesh):
    @classmethod
    def from_polygons(cls, polys, element_coords='centroid'):
        """
        Create an ESMF.Mesh object from a list of polygons.

        Validate input `polys` to prevent errors and unnecessary computations.
        All exterior ring points are added to the mesh as nodes and each polygon
        is added as an element, with the polygon centroid as the element's coordinates.

        Parameters
        ----------
        polys : sequence of shapely Polygon
           Holes are not represented by the Mesh.
        element_coords : array or "centroid", optional
            If "centroid", the polygon centroids will be used (default)
            If an array of shape (len(polys), 2) : the element coordinates of the mesh.
            If None, the Mesh's elements will not have coordinates.

        Returns
        -------
        mesh : ESMF.Mesh
            A mesh where each polygon is represented as an Element.
        """
        if not polys:
            raise ValueError('Input `polys` is empty. Provide a non-empty sequence of polygons.')

        for poly in polys:
            if not poly.is_valid:
                raise ValueError('Invalid polygon detected. Check polygon validity before processing.')

        node_num = sum(len(e.exterior.coords) - 1 for e in polys)
        elem_num = len(polys)
        # Pre alloc arrays. Special structure for coords makes the code faster.
        crd_dt = np.dtype([('x', np.float32), ('y', np.float32)])
        node_coords = np.empty(node_num, dtype=crd_dt)
        node_coords[:] = np.nan
        element_types = np.empty(elem_num, dtype=np.uint32)
        element_conn = np.empty(node_num, dtype=np.uint32)
        # Flag for centroid calculation
        calc_centroid = isinstance(element_coords, str) and element_coords == 'centroid'
        if calc_centroid:
            element_coords = np.empty(elem_num, dtype=crd_dt)
        inode = 0
        iconn = 0
        node_coord_map = {}
        for ipoly, poly in enumerate(polys):
            ring = poly.exterior
            if calc_centroid:
                element_coords[ipoly] = poly.centroid.coords[0]
            element_types[ipoly] = len(ring.coords) - 1
            unique_nodes = set()
            for coord in ring.coords[:-1] if ring.is_ccw else ring.coords[:0:-1]:
                crd = np.asarray(coord, dtype=crd_dt)  # Cast so we can compare
                if crd not in unique_nodes:  # New node
                    unique_nodes.add(crd)
                    node_index = node_coord_map.get(tuple(crd), None)
                    if node_index is None:  # New node
                        node_coords[inode] = crd
                        element_conn[iconn] = inode
                        node_coord_map[tuple(crd)] = inode
                        inode += 1
                    else:  # Node already exists
                        element_conn[iconn] = node_index
                else:  # Node already exists
                    element_conn[iconn] = list(unique_nodes).index(crd)
                iconn += 1
        node_num = inode  # With duplicate nodes, inode < node_num

        mesh = cls(2, 2, coord_sys=ESMF.CoordSys.SPH_DEG)
        mesh.add_nodes(
            node_num,
            np.arange(node_num) + 1,
            nprec.structured_to_unstructured(node_coords[:node_num]).ravel(),
            np.zeros(node_num),
        )
        if calc_centroid:
            element_coords = nprec.structured_to_unstructured(element_coords)
        if element_coords is not None:
            element_coords = element_coords.ravel()
        try:
            mesh.add_elements(
                elem_num,
                np.arange(elem_num) + 1,
                element_types,
                element_conn,
                element_coords=element_coords,
            )
        except ValueError as err:
            raise ValueError(
                'ESMF failed to create the Mesh, this usually happen when some polygons are invalid (test with `poly.is_valid`)'
            ) from err
        return mesh

    def get_shape(self, loc=ESMF.MeshLoc.ELEMENT):
        """Return the shape of the Mesh at specified MeshLoc location."""
        return (self.size[loc], 1)


@dask.delayed
def esmf_regrid_build_parallel(sourcegrid, destgrid, method, filename=None, extra_dims=None, extrap_method=None, extrap_dist_exponent=None, extrap_num_src_pnts=None, ignore_degenerate=None):
    """
    Create an ESMF.Regrid object, containing regridding weights, in parallel using Dask.

    Parameters
    ----------
    sourcegrid, destgrid : ESMF.Grid or ESMF.Mesh object
        Source and destination grids.

    method : str
        Regridding method.

    filename : str, optional
        Offline weight file.

    extra_dims : list of integers, optional
        Extra dimensions in the data field.

    extrap_method : str, optional
        Extrapolation method.

    extrap_dist_exponent : float, optional
        The exponent for extrapolation.

    extrap_num_src_pnts : int, optional
        Number of source points for extrapolation.

    ignore_degenerate : bool, optional
        Whether to ignore degenerate cells.

    Returns
    -------
    regrid : ESMF.Regrid object
    """

    # Your existing code for creating regridding weights goes here...

    regrid = esmf_regrid_build(sourcegrid, destgrid, method, filename, extra_dims, extrap_method, extrap_dist_exponent, extrap_num_src_pnts, ignore_degenerate)

    return regrid

# Example usage
regrid = esmf_regrid_build_parallel(sourcegrid, destgrid, method, filename, extra_dims, extrap_method, extrap_dist_exponent, extrap_num_src_pnts, ignore_degenerate)

def esmf_regrid_build(
    sourcegrid,
    destgrid,
    method,
    filename=None,
    extra_dims=None,
    extrap_method=None,
    extrap_dist_exponent=None,
    extrap_num_src_pnts=None,
    ignore_degenerate=None,
):
    """
    Create an ESMF.Regrid object, containing regridding weights.

    Parameters
    ----------
    sourcegrid, destgrid : ESMF.Grid or ESMF.Mesh object
        Source and destination grids.

        Should create them by ``Grid.from_xarray()``
        (with optionally ``add_corner()``),
        instead of ESMPy's original API.

    method : str
        Regridding method. Options are

        - 'bilinear'
        - 'conservative', **need grid corner information**
        - 'conservative_normed', **need grid corner information**
        - 'patch'
        - 'nearest_s2d'
        - 'nearest_d2s'

    filename : str, optional
        Offline weight file. **Require ESMPy 7.1.0.dev38 or newer.**
        With the weights available, we can use Scipy's sparse matrix
        multiplication to apply weights, which is faster and more Pythonic
        than ESMPy's online regridding. If None, weights are stored in
        memory only.

    extra_dims : a list of integers, optional
        Extra dimensions (e.g. time or levels) in the data field

        This does NOT affect offline weight file, only affects online regrid.

        Extra dimensions will be stacked to the fastest-changing dimensions,
        i.e. following Fortran-like instead of C-like conventions.
        For example, if extra_dims=[Nlev, Ntime], then the data field dimension
        will be [Nlon, Nlat, Nlev, Ntime]

    extrap_method : str, optional
        Extrapolation method. Options are

        - 'inverse_dist'
        - 'nearest_s2d'

    extrap_dist_exponent : float, optional
        The exponent to raise the distance to when calculating weights for the
        extrapolation method. If none are specified, defaults to 2.0

    extrap_num_src_pnts : int, optional
        The number of source points to use for the extrapolation methods
        that use more than one source point. If none are specified, defaults to 8

    ignore_degenerate : bool, optional
        If False (default), raise error if grids contain degenerated cells
        (i.e. triangles or lines, instead of quadrilaterals)

    Returns
    -------
    grid : ESMF.Grid object

    """

    # use shorter, clearer names for options in ESMF.RegridMethod
    method_dict = {
        'bilinear': ESMF.RegridMethod.BILINEAR,
        'conservative': ESMF.RegridMethod.CONSERVE,
        'conservative_normed': ESMF.RegridMethod.CONSERVE,
        'patch': ESMF.RegridMethod.PATCH,
        'nearest_s2d': ESMF.RegridMethod.NEAREST_STOD,
        'nearest_d2s': ESMF.RegridMethod.NEAREST_DTOS,
    }
    try:
        esmf_regrid_method = method_dict[method]
    except Exception:
        raise ValueError('method should be chosen from ' '{}'.format(list(method_dict.keys())))

    # use shorter, clearer names for options in ESMF.ExtrapMethod
    extrap_dict = {
        'inverse_dist': ESMF.ExtrapMethod.NEAREST_IDAVG,
        'nearest_s2d': ESMF.ExtrapMethod.NEAREST_STOD,
        None: None,
    }
    try:
        esmf_extrap_method = extrap_dict[extrap_method]
    except KeyError:
        raise KeyError(
            f'`extrap_method` should be chosen from {list(extrap_dict.keys())}'
        ) from exc

    # until ESMPy updates ESMP_FieldRegridStoreFile, extrapolation is not possible
    # if files are written on disk
    if (extrap_method is not None) & (filename is not None):
        raise ValueError('`extrap_method` cannot be used along with `filename`.')

    # conservative regridding needs cell corner information
    if method in ['conservative', 'conservative_normed']:
        if not isinstance(sourcegrid, ESMF.Mesh) and not sourcegrid.has_corners:
            raise ValueError(
                'source grid has no corner information. ' 'cannot use conservative regridding.'
            )
        if not isinstance(destgrid, ESMF.Mesh) and not destgrid.has_corners:
            raise ValueError(
                'destination grid has no corner information. ' 'cannot use conservative regridding.'
            )

    # ESMF.Regrid requires Field (Grid+data) as input, not just Grid.
    # Extra dimensions are specified when constructing the Field objects,
    # not when constructing the Regrid object later on.
    if isinstance(sourcegrid, ESMF.Mesh):
        sourcefield = ESMF.Field(sourcegrid, meshloc=ESMF.MeshLoc.ELEMENT, ndbounds=extra_dims)
    else:
        sourcefield = ESMF.Field(sourcegrid, ndbounds=extra_dims)
    if isinstance(destgrid, ESMF.Mesh):
        destfield = ESMF.Field(destgrid, meshloc=ESMF.MeshLoc.ELEMENT, ndbounds=extra_dims)
    else:
        destfield = ESMF.Field(destgrid, ndbounds=extra_dims)

    # ESMF bug? when using locstream objects, options src_mask_values
    # and dst_mask_values produce runtime errors
    allow_masked_values = True
    if isinstance(sourcefield.grid, ESMF.api.locstream.LocStream):
        allow_masked_values = False
    if isinstance(destfield.grid, ESMF.api.locstream.LocStream):
        allow_masked_values = False

    # ESMPy will throw an incomprehensive error if the weight file
    # already exists. Better to catch it here!
    if filename is not None:
        assert not os.path.exists(
            filename
        ), 'Weight file already exists! Please remove it or use a new name.'

    # re-normalize conservative regridding results
    # https://github.com/JiaweiZhuang/xESMF/issues/17
    if method == 'conservative_normed':
        norm_type = ESMF.NormType.FRACAREA
    else:
        norm_type = ESMF.NormType.DSTAREA

    # Calculate regridding weights.
    # Must set unmapped_action to IGNORE, otherwise the function will fail,
    # if the destination grid is larger than the source grid.
    kwargs = dict(
        filename=filename,
        regrid_method=esmf_regrid_method,
        unmapped_action=ESMF.UnmappedAction.IGNORE,
        ignore_degenerate=ignore_degenerate,
        norm_type=norm_type,
        extrap_method=esmf_extrap_method,
        extrap_dist_exponent=extrap_dist_exponent,
        extrap_num_src_pnts=extrap_num_src_pnts,
        factors=filename is None,
    )
    if allow_masked_values:
        kwargs.update(dict(src_mask_values=[0], dst_mask_values=[0]))

    regrid = ESMF.Regrid(sourcefield, destfield, **kwargs)

    return regrid


def esmf_regrid_apply(regrid, indata):
    """
    Apply existing regridding weights to the data field,
    using ESMPy's built-in functionality.

    xESMF use Scipy to apply weights instead of this.
    This is only for benchmarking Scipy's result and performance.

    Parameters
    ----------
    regrid : ESMF.Regrid object
        Contains the mapping from the source grid to the destination grid.

        Users should create them by esmf_regrid_build(),
        instead of ESMPy's original API.

    indata : numpy array of shape ``(Nlon, Nlat, N1, N2, ...)``
        Extra dimensions ``(N1, N2, ...)`` are specified in
        ``esmf_regrid_build()``.

        Recommend Fortran-ordering to match ESMPy internal.

    Returns
    -------
    outdata : numpy array of shape ``(Nlon_out, Nlat_out, N1, N2, ...)``

    """

    # Passing C-ordered input data will be terribly slow,
    # since indata is often quite large and re-ordering memory is expensive.
    check_f_contiguous(indata)

    # Get the pointers to source and destination fields.
    # Because the regrid object points to its underlying field&grid,
    # we can just pass regrid from ESMF_regrid_build() to ESMF_regrid_apply(),
    # without having to pass all the field&grid objects.
    sourcefield = regrid.srcfield
    destfield = regrid.dstfield

    # pass numpy array to the underlying Fortran array
    sourcefield.data[...] = indata

    # apply regridding weights
    destfield = regrid(sourcefield, destfield)

    return destfield.data


def esmf_regrid_finalize(regrid):
    """
    Free the underlying Fortran array to avoid memory leak.

    After calling ``destroy()`` on regrid or its fields, we cannot use the
    regrid method anymore, but the input and output data still exist.

    Parameters
    ----------
    regrid : ESMF.Regrid object

    Raises
    ------
    RuntimeError: If `regrid` is already finalized or if any of the `destroy()` calls fail.

    """

    if regrid is not None:
        try:
            regrid.destroy()
            if regrid.srcfield is not None:
                regrid.srcfield.destroy()
            if regrid.dstfield is not None:
                regrid.dstfield.destroy()

            # double check
            if not (regrid.finalized and regrid.srcfield.finalized and regrid.dstfield.finalized):
                raise RuntimeError("Regrid finalization failed.")
        except Exception as e:
            print(f"Error occurred during destruction: {e}")



def esmf_regrid_apply_parallel(regrid, indata):
    """
    Apply existing regridding weights to the data field using Dask for parallel computation.

    Parameters
    ----------
    regrid : ESMF.Regrid object
        Contains the mapping from the source grid to the destination grid.

    indata : numpy array of shape ``(Nlon, Nlat, N1, N2, ...)``
        Extra dimensions ``(N1, N2, ...)`` are specified in ``esmf_regrid_build()``.

    Returns
    -------
    outdata : numpy array of shape ``(Nlon_out, Nlat_out, N1, N2, ...)``

    """
    # Passing C-ordered input data will be terribly slow,
    # since indata is often quite large and re-ordering memory is expensive.
    check_f_contiguous(indata)

    # Convert input data to Dask array for parallel processing
    dask_array = da.from_array(indata, chunks='auto')

    # Define a function to apply regridding weights to a chunk of data
    def apply_regridding_chunk(chunk):
        sourcefield = regrid.srcfield
        destfield = regrid.dstfield
        sourcefield.data[...] = chunk
        destfield = regrid(sourcefield, destfield)
        return destfield.data

    # Apply regridding to each chunk of data in parallel using map_blocks
    outdata = dask_array.map_blocks(apply_regridding_chunk, dtype=indata.dtype)

    # Compute the result to get the final output data
    outdata = outdata.compute()

    return outdata

class Regrid:
    def __init__(self, sourcegrid, destgrid, method, filename=None, extra_dims=None, extrap_method=None, extrap_dist_exponent=None, extrap_num_src_pnts=None, ignore_degenerate=None):
        self.sourcegrid = sourcegrid
        self.destgrid = destgrid
        self.method = method
        self.filename = filename
        self.extra_dims = extra_dims
        self.extrap_method = extrap_method
        self.extrap_dist_exponent = extrap_dist_exponent
        self.extrap_num_src_pnts = extrap_num_src_pnts
        self.ignore_degenerate = ignore_degenerate
        self.regrid = None

    def create_weights(self):
        self.regrid = esmf_regrid_build(self.sourcegrid, self.destgrid, self.method, self.filename, self.extra_dims, self.extrap_method, self.extrap_dist_exponent, self.extrap_num_src_pnts, self.ignore_degenerate)

    def apply_regridding(self, indata):
        if self.regrid is None:
            raise ValueError("Regrid weights have not been created. Call create_weights() first.")
        return esmf_regrid_apply(self.regrid, indata)

    def finalize(self):
        if self.regrid is None:
            raise ValueError("Regrid weights have not been created. Call create_weights() first.")
        esmf_regrid_finalize(self.regrid)

    def write_weights_to_file(self, output_file):
        if self.regrid is None:
            raise ValueError("Regrid weights have not been created. Call create_weights() first.")
        if self.filename is not None:
            raise ValueError("Weights have already been written to a file.")
        self.filename = output_file
        self.regrid = esmf_regrid_build(self.sourcegrid, self.destgrid, self.method, self.filename, self.extra_dims, self.extrap_method, self.extrap_dist_exponent, self.extrap_num_src_pnts, self.ignore_degenerate)

    def get_regrid_weights(self):
        if self.regrid is None:
            raise ValueError("Regrid weights have not been created. Call create_weights() first.")
        return self.regrid