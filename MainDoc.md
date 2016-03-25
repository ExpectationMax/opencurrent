## Overview ##

OpenCurrent is a library for solving PDEs over structured grids using CUDA-capable devices.  See the OpenCurrent wiki page for instructions on how to build the test suite and for a quick introduction to the library.

While some example applications are included, the goal of OpenCurrent is to provide utilities and functionality that can be used to build useful solvers in a variety of application domains.

## Design Principles ##

The OpenCurrent library follows a small number of design principles throughout the codebase:
  * **Static Communciation Patterns**:  In a complex system, managing communication is one of the most difficult and important aspects for correctness and performance.  OpenCurrent follows the philosophy that all communication patterns must be declared up-front statically, and cannot be changed once the data structures are created.  This greatly simplifies the run-time system, since it allows communication patterns to be hard-coded at start-up, which puts the burden for choosing how best to arrange communcation onto the programmer.  This allows OpenCurrent to have a very lightweight run-time system, which improves performance and makes the overall code-base simpler.
  * **Seperation between Topological and Differential Code**
  * Todo: Others...

## Library Documentation ##


OpenCurrent is divided into 3 libraries:

**OpenCurrentUtil**
  * Functionality that is not related to any particular algorithm or data structure.

**OpenCurrentStorage**
  * Grid data structures and basic operations over those data structures.

**OpenCurrentEquation**
  * Components needed to solve equations.

There is also a **[unit testing framework](UnitTesting.md)**.

## Double Precision Support ##

OpenCurrent is designed to run on all CUDA-capable devices, with certain functionality automatically disabled based on the compute capabilities of the target platform.  In particular, double precision is not support in kernels prior to compute 1.3.  In order to maintain uniformity of the head files across different compilation targets, the library automatically does not define symbols that would require double-precision support when the compute target is < 1.3.  This means that attempting to call a double-precision routine will result in a link-time error, typically 'undefined symbol.'  In order to avoid these errors, application code should avoid calling double-precision routines when linked against a version of OpenCurrent built for compute caoability prior to 1.3.

Essentially all routines or data structures that are templated on type and have a 'Device' version are not defined.  Host routines, however, are unaffected.  For example, calling routines on a Grid3DDeviceD object will result in link errors, while calling routines on Grid3DHostD will not.