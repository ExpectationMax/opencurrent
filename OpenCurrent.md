## Introduction ##

OpenCurrent is an open source C++ library for solving Partial Differential Equations (PDEs) over regular grids using the CUDA platform from NVIDIA. It breaks down a PDE into 3 basic objects, “Grids”, “Solvers,” and “Equations.” “Grid” data structures efficiently implement regular 1D, 2D, and 3D arrays in both double and single precision. Grids support operations like computing linear combinations, managing host-device memory transfers, interpolating values at non-grid points, and performing array-wide reductions. “Solvers” use these data structures to calculate terms arising from discretizations of PDEs, such as finite-difference based advection and diffusion schemes, and a multigrid solver for Poisson equations. These computational building blocks can be assembled into complete “Equation” objects that solve time-dependent PDEs. One such Equation solver is an incompressible Navier-Stokes solver that uses a second-order Boussinesq model. This equation solver is fully validated, and has been used to study Rayleigh-Benard convection under a variety of different regimes [(citation)](http://www.jcohen.name/papers/Cohen_Fast_2009.pdf). Benchmarks show it to perform about 8 times faster than an equivalent Fortran code running on an 8-core Xeon.

The OpenCurrent infrastructure includes support for profiling both the CPU or GPU, support for reading and writing NetCDF data files, and the ability to generate simple plots. It includes a complete validation and unit testing framework that allows for easy and automatic validation of numerical methods and solvers. OpenCurrent uses CMake for cross-platform development and has been tested on both Windows and Linux-based systems. Via a compile-time option, OpenCurrent can be configured to support older hardware (pre-GT200) that does not handle double precision, but on newer hardware all routines are available in both double and single precision.

## Documentation ##

Please see [the main documentation page](MainDoc.md)

## Getting Started ##

### Requirements ###

OpenCurrent is designed to run on Windows or Linux systems (not currently MacOS).  It has been tested on Windows XP-32 with Visual Studio 2005 and Ubuntu-64 with gcc 4.x.  It should run on other systems, so if you successfully build it & pass the testsuite on a different system, please let me know.  While it is theoretically possible to build CUDA applications on Windows using gcc with cygwin, I have so far been unsuccessful at getting this to work with CMake.  If someone has better luck, please let me know how you did it.

OpenCurrent requires CUDA 2.3 or later from NVIDIA, which can be downloaded from http://www.nvidia.com/object/cuda_get.html.  CUDA 2.3 requires a NVIDIA graphics driver [r190](https://code.google.com/p/opencurrent/source/detail?r=190) or later, which can be downloaded from the same page.  CUDA 3.0 requires driver version [r195](https://code.google.com/p/opencurrent/source/detail?r=195) or later.

Building OpenCurrent requires CMake 2.6.4 or later, which can be downloaded from http://www.cmake.org/ for all platforms.

OpenCurrent currently has only a single optional external dependency, which is NetCDF, a file IO library used in the ocean and atmospheric sciences communities for storing gridded data.  OpenCurrent will run with either NetCDF 3.6 or NetCDF 4.0, as long as the appropriate flag is set in the System.cmake file.  When using NetCDF 3.6, file compression will be automatically disabled because it is not supported by this older version of the library.

NetCDF can be downloaded from http://www.unidata.ucar.edu/software/netcdf/.  I highly recommend downloading the entire project and building it yourself, as the pre-built versions seem to be a bit flaky in my testing.

Optionally, all NetCDF features may be disabled via a compile-time flag.  In this case, the project will build without requiring NetCDF.

### Building ###

To build OpenCurrent, you must first copy the file in the top-level directory System.cmake.src to System.cmake, and then edit it as per your system setup.  The default should properly detect whether you are on Windows or Linux, but if it gets it wrong, you can just modify the logic as required.  Also, you need to make sure the following variables are set correctly in that file:
  * OCU\_TARGET\_SM - what sm version you are targeting.  For a GT200-based GPU, this will be sm\_13, for example.
  * CUDA\_TOOLKIT\_ROOT\_DIR - where CUDA is installed.  The defaults will normally be correct, but you may have done something funky.
  * OCU\_NETCDF\_ENABLED - set to TRUE of FALSE depending on whether you want to compile with NetCDF support for file i/o.  If set to FALSE, then the project can be built without requiring NetCDF to be installed.
  * OCU\_OMP\_ENABLED - set to TRUE or FALSE depending on whether your compiler supports OpenMP pragmas.  If set to FALSE, multi-GPU routines will not be supported.
  * OCU\_NETCDF4\_SUPPORT\_ENABLED - set to TRUE or FALSE depending on whether you have NetCDF 4.x (TRUE) or 3.6 (FALSE) installed.
  * NetCDF\_INCLUDE\_DIR - optionally, you can hardcode where NetCDF header files will be found, or you can let CMake's default path searching find them.
  * NetCDF\_LIBRARY - optionally, you can hardcode where NetCDF library will be found, or you can let CMake's default path searching find them.

Once the System.cmake file is properly set up, you can now build the library.  I **highly** recommend an out-of-source build, where you place all of the binary & non-source files into a separate directory.

#### Linux ####

From the top level (opencurrent) directory:

```
 > mkdir sm13-rel  (or whatever you want to call it)
 > cd sm13-rel
 > cmake ../src
 > make
```

This will create an out-of-source build in a sibling directory to 'src' called 'sm13-rel'.  Naming the build directory like this will allow you to have multiple versions of the same source tree built with different configurations.  For example, to build with debug symbols:

```
 > mkdir sm13-dbg 
 > cd sm13-dbg
 > cmake ../src
 > [Edit CMakeCache.txt to change CMAKE_BUILD_TYPE to Debug]
 > make
```

Similarly, to build for a different target sm, such as sm\_10, modify the OCU\_TARGET\_SM variable to be sm\_10.

#### Windows ####

From the top-level (opencurrent) directory, create a new directory called msvc-sm13 (or whatever).  CMake will generate a Visual Studio project that contains the Debug and Release builds in it, so you don't need a separate directory for them as with the Linux version.  However, you will need separate directories if you want to have different sm's supported.

Using the CMake GUI, point it to the opencurrent/src directory as the input, and opencurrent/msvc-sm13 as the build directory. Then hit "Configure" and then "Ok." when prompted, Choose the appropriate generator (probably Visual Studio 2005). There should now be a file called opencurrent.sln in the build directory.  Open this with Visual Studio and build the project.

### Testing ###

To test everything out, you can run the unit test.  OpenCurrent contains an extensive unit test that is integrated with CMake.  On Linux, just type 'make test' from the build directory.  On Windows, build the RUN\_TESTS project in Visual Studio.  Hopefully everything will work fine.

## Example ##

This example source code (found in apps/incompress.cpp) demonstrates a program that loads a staggered grid via a netCDF file, projects it to be incompressible, then writes the resulting file to disk.


First, we read the file name & open it:

```
int main(int argc, const char **argv) 
{
  const char *input = argv[1];
  const char *output = argv[2];
  double tolerance = atof(argv[3]);

  NetCDFGrid3DReader reader;
  reader.open(input);

  int nx = reader.nx();
  int ny = reader.ny();
  int nz = reader.nz();
  float hx = reader.hx();
  float hy = reader.hy();
  float hz = reader.hz();
```

Next, we need to allocate grids to hold the u,v,w grids, on both the host and device.  For efficiency, OpenCurrent requires that all grids are allocated to be congruent via padding.  Congruent means that the data layout in memory is identical, even though grids may have different dimensions.  The following code will create the 3 arrays required to hold a staggered vector field, and additionally require that they all have a single row of ghost cells in all dimensions.  Through a call to Grid3DDimension::pad\_for\_congruence, the padding parameters are set so that all 3 grids will have identical memory layout.  Finally, we will allocate the actual memory for these grids on the host and device.
```
  std::vector<Grid3DDimension> dimensions(3);
  dimensions[0].init(nx+1,ny,nz,1,1,1); // x,y,z dim, followed by x,y,z ghost cells
  dimensions[1].init(nx,ny+1,nz,1,1,1);
  dimensions[2].init(nx,ny,nz+1,1,1,1);  
  Grid3DDimension::pad_for_congruence(dimensions);

  Grid3DHostD h_u, h_v, h_w;
  Grid3DDeviceD d_u, d_v, d_w;

  d_u.init_congruent(dimensions[0]);
  d_v.init_congruent(dimensions[1]);
  d_w.init_congruent(dimensions[2]);
  h_u.init_congruent(dimensions[0],true); // true => use pinned memory
  h_v.init_congruent(dimensions[1],true);
  h_w.init_congruent(dimensions[2],true);
```

We will perform the actual divergence-free projection using a mixed precision (iterative refinement) multigrid solver.  We need to tell it what boundary conditions to use for the projection, and in this case we will set no-slip wall boundaries.  Then we need to initialize it, telling it which grids to process.

```
  Sol_ProjectDivergenceMixed3DDeviceD projector;
  BoundaryCondition bc;
  bc.type = BC_FORCED_INFLOW_VARIABLE_SLIP;
  projector.bc = BoundaryConditionSet(bc);
  projector.initialize_storage(nx, ny, nz, hx, hy, hz, &d_u, &d_v, &d_w);
```

Next we will set up a NetCDF file writer, telling it which grids to write, what type of grid they are, and what they are called.
```
  NetCDFGrid3DWriter writer;
  writer.open(output, nx, ny, nz, hx, hy, hz);
  writer.define_variable("u", NC_DOUBLE, GS_U_FACE);
  writer.define_variable("v", NC_DOUBLE, GS_V_FACE);
  writer.define_variable("w", NC_DOUBLE, GS_W_FACE);
```

Finally, we will step through all the time levels to perform the actual projection.  First, we load the staggered grid from disk into the host grids.  then we copy from host to device.  We tell the projector to solve, which will actually calculate the projection, updating the u,v, and w grids in place.  We then copy the results back to the CPU, and write them to disk.

```
  int nsteps = reader.num_time_levels();
  for (int step = 0; step < nsteps; step++) {
    reader.read_variable("u", h_u, step);
    reader.read_variable("v", h_v, step);
    reader.read_variable("w", h_w, step);

    d_u.copy_all_data(h_u);
    d_v.copy_all_data(h_v);
    d_w.copy_all_data(h_w);

    GPUTimer solve_timer;
    solve_timer.start();
    projector.solve(tolerance);
    solve_timer.stop();

    h_u.copy_all_data(d_u);
    h_v.copy_all_data(d_v);
    h_w.copy_all_data(d_w);

    size_t out_level;
    writer.add_time_level(reader.get_time(step), out_level);
    writer.add_data("u", h_u, out_level);
    writer.add_data("v", h_v, out_level);
    writer.add_data("w", h_w, out_level);
  }
```