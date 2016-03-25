## What is OpenCurrent? ##

OpenCurrent is an open source C++ library for solving Partial Differential Equations (PDEs) over regular grids using the [CUDA](http://www.nvidia.com/cuda) platform from [NVIDIA](http://www.nvidia.com).

Browse the [source tree](http://code.google.com/p/opencurrent/source/browse/) or [download](http://code.google.com/p/opencurrent/downloads/list) the code.  Read the [documentation here](MainDoc.md).

## News ##

**See Jonathan Cohen's talk on OpenCurrent at GTC:**

http://developer.download.nvidia.com/compute/cuda/docs/GTC_2010_Archives.htm#RANGE!A48

The video feed apparently wasn't working, so you can follow along in the [slide deck](http://opencurrent.googlecode.com/files/GTC_2010_OpenCurrent.pdf).

**Version 1.1.0 released**

New features:
  * Multi-GPU communication library
  * Multi-GPU versions of Multigrid solver, Incompressible Navier-Stokes solver, and more
  * NetCDF support now optional
  * Support for Fermi/CUDA 3.0
  * Numerous bug fixes and enhancements

## Contact ##

OpenCurrent was developed by [Jonathan Cohen](http://www.jcohen.name) at [NVIDIA Research](http://www.nvidia.com/research).  Please contact the [opencurrent-users google group](http://groups.google.com/group/opencurrent-users) with any questions.