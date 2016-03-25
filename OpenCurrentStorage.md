**Still todo:**
  * Point sampling support
  * boundary condition handling support
  * NetCDF support
  * reduction support
  * co-array support (when added)
  * templated operator support (when added)

# Introduction #

OCUStorage provides data structures and basic operations over those data structures.  Currently, the supported data structures include 1D, 2D, and 3D array classes callde "Grids".

The grid classes are defined in the header files [ocustorage/grid1d.h](http://code.google.com/p/opencurrent/source/browse/src/ocustorage/grid1d.h), [ocustorage/grid2d.h](http://code.google.com/p/opencurrent/source/browse/src/ocustorage/grid2d.h), and [ocustorage/grid3d.h](http://code.google.com/p/opencurrent/source/browse/src/ocustorage/grid3d.h).

Grids are named according to the naming convention:

Grid **X** d {**Host** | **Device** } { **I** | **F** | **D** }

  * **X** is the dimension of the grid
  * **Host** means the grid memory is allocated on the host while **Device** means it is allocated on the device,
  * the atoms stored in the grid are 32-bit integer (**I**), 32-bit float (**F**) or 64-bit float (**D**).

For example, !Grid1DHostF would be a 1-dimensional grid of floats that resides on the host.

The lifecycle of a grid object is to first create it, which does not allocate any memory.  Next, you must initialize the grid via a call to either `init()` or `init_congruent()` (more on this below).  Then all other routines may be called.  The grid's destructor cleans up all memory allocations.  Grid may not be reused - that is, you cannot adjust the grid's dimension after it has been initialized.  In order to change the dimension of a grid, you must actually delete it and then allocate a new one.

All arrays are stored in z,y,x-major order.  That is, z is the fastest changing axis, then y, then x.  Note that this is the opposite of what the CUDA model uses for thread ID, where x is the fastest changing axis, then y, then z.

# Grid Padding #

To allow for straightforward handling of boundary conditions, the Grid classes allow for out-of-bounds indexing via a padding scheme.

For example, the `init()` routines to the `Grid3DHostF` class takes both the dimension and the number of "ghost" cells in all directions:
```
  Grid3DHostF g;
  g.init(64, 64, 64, 1, 1, 1);
```
This would allocate a grid that is 64 x 64 x 64 with a single row of "ghost" cells in all directions.  In this case, the "interior" dimensions would be 64 x 64 x 64, and are denoted `nx`, `ny`, and `nz`.  The "ghost dimensions would be 1, and are denoted `gx`, `gy`, and `gz`.  Because indexing is zero-based, you can access elements at index -1 or at 64.  For example:
```
  float a = g.at(10,10,10); // valid
  float b = g.at(-1,-1,-1); // valid
  float d = g.at(64, 0, 0); // valid
  float c = g.at(-2, 0, 0); // NOT VALID
  float e = g.at(65, 0, 0); // NOT VALID
```
Padding can be considered similar to creating fortran arrays that are not 1-based.

# Class Hierarchy #

For each dimensionality, the grid classes follow a class hierarchy (note that this isn't totally implemented yet, so some dimensions are missing some features).

GridUntyped
  * Base class.
  * Provides functionality for determining offset for a particular element in the buffer.


GridDimension
  * subclass of GridUntyped
  * No additional functionality - used to express dimensions and layout of a grid without any allocated memory.

GridBase
  * subclass of GridUntyped
  * templated on stored data type
  * Base class for grids that provide allocated buffers.
  * Provides pointers to start of linear buffers, adds functionality for determining actual position of given element.

GridHost
  * subclass of GridBase
  * templated on stored data type
  * Manages buffers allocated on host via `cudaHostAlloc`

GridDevice
  * subclass of GridBase
  * templated on stored data type
  * Managers buffers allocated on device via `cudaMalloc`.

# Memory layout #

The memory layout is described in [A Fast Double Precision CFD Code Using CUDA](http://jcohen.name/papers/Cohen_Fast_2009_final.pdf), and is summarized here.

![http://jcohen.name/images/cuda_layout.png](http://jcohen.name/images/cuda_layout.png)

Because of CUDA's coalescing rules, it is convenient to lay out memory in an array so that z-columns in the grid start on 16-element boundaries (64 bytes for a 32-bit structure, 128 bytes for a 64-bit structure). Grids are "pre-padded" so that the column z=0 begins at a 16-element boundary (this is possible because `cudaMalloc` returns aligned buffers).  Additionally, each z-column is padded so that the total number of bytes between adjacent z-columns is a multiple of 16 elements (in addition to any padding necessary for the ghost cells).  Consequently, `&at(i,j,0)` will always begin at a 16-element boundary, for any `i` and `j`.

The full padded size of each dimension is `pnx`, `pny`, and `pnz`.  Therefore, the stride between adjacent elements in z is 1, stride between adjacent elements in y is `pnz`, and the stride between adjacent elements in x is `pny * pnz`.  These strides can be accessed via the `xstride()`, `ystride()`, and `zstride()` accessors.

It is sometimes convenient to bind a grid to a 1D texture so that its elements can be accessed via the `texfetch1d()` routine and receive the benefits of the texture cache.  Any linear buffer can be bound as a 1d texture, provided it obeys certain (hardware dependent) alignment rules.  Pointers returned from `cudaMalloc` are guaranteed to be properly aligned for texture binding.  Therefore, when binding a grid's buffer as a texture, we always bind the pointer at the beginning of the buffer, and then in software offset our texture index in order to undo the pre-padding.  The number of elements between the start of the buffer and the position of element (0,0,0) is stored as `shift_amount`.

It is convenient to enforce identical layout for all grids of the same dimensions.  Therefore, host grids are laid out in memory identically, even though they do not necessarily benefit from the 16-element alignment.


# Congruent Padding #

Oftentimes, a single CUDA routine will access several different grids at the same time.  These grids may have different dimensions.  For example, a staggered MAC grid in 3D is stored in 3 grids, u,v, and w.  For a nx x ny x nz discretization, u would have dimensions (nx+1) x ny x nz, which v would have nx x (ny+1) x nz, and w would have nx x ny x (nz+1).

However, by padding all 3 grids appropriately, it is possible to store them in such a way that their xstride, ystride, and zstride all match.  This allows us to write more efficient CUDA routines that access particular elements from multiple grids.  We refer to this process of padding multiple grids so that their strides match as **congruent padding**.  For this reason, most solvers expect that their input grids will be congruently padded and will return an error upon initialization if this is not the case.

There are 3 ways to perform congruent padding.  One is to do it by hand, passing the appropriate `pnx` `pnx` and `pnz` values into the `init` routines.  This is not recommended.

The second applies when the grids will have identical dimensions (i.e. not the case described above).  In this case, you can pass one grid into the `init_congruent` routine of another grid, and the second grid will be allocated with identical dimensions and layout.  This is useful, for example, when copying from a device to a host grid for debugging:
```
   Grid3DDeviceF d_grid;
   // perform calculation on d_grid
   Grid3DHostF h_grid;
   h_grid.init_congruent(d_grid); // will have the same layout
   h_grid.copy_all_data(d_grid); // this requires identical layout, which is now guaranteed
   // examine contents of h_grid
```

The third and most general way is to use the routine `Grid3DDimension::pad_for_congruence`.  It returns the appropriate inputs to the `init` routine so that the grids will be congruent, although they may have different interior and ghost dimensions.

For example, the following code illustrates how a staggered grid can be created:

```
  std::vector<Grid3DDimension> dimensions(3);
  dimensions[0].init(nx+1,ny,nz,1,1,1); // nx,ny,nz and gx,gy,gz specified,
  dimensions[1].init(nx,ny+1,nz,1,1,1); // but pnx, pny, and pnz left blank.
  dimensions[2].init(nx,ny,nz+1,1,1,1);  
  Grid3DDimension::pad_for_congruence(dimensions); // this modifies dimensions in place

  Grid3DDeviceD d_u, d_v, d_w;
  d_u.init_congruent(dimensions[0]);
  d_v.init_congruent(dimensions[1]);
  d_w.init_congruent(dimensions[2]);
  // grids will now be congruent.
```