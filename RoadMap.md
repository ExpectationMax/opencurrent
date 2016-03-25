# Multi-GPU #

Support single system, multi-GPU configurations
  * Based on OpenMP (for cross-platform support)
  * [Co-array](http://www.co-array.org/) model for inter-gpu communication
  * Lightweight thread management, some basic multithreading routines (reductions, barriers, etc.)
  * OpenCurrent philosophy: static communication patterns for efficiency.  Solvers work unchanged (operate on local grids).  Equations manage communication between multiple solvers.

# MPI-enabled #

Support for clusters
  * Built on top of multi-gpu framework.  New backend for the co-array model.

# Templated operators #

Allow for 'apply to grid'-type operations
  * Built using C++ templates, and possibly metaprogramming (pending compiler support).
  * Allow for more natural expression of per-cell operations, remove most of boiler-plate code.
  * rewrite entire library using this framework - should shrink codebase considerably.

# Solver enhancements #

Implement better stencils
  * Lots of good papers on efficient stencils in cuda.  Implement some of them.

Iterative solvers
  * New multigrid solvers for Poisson equations
  * connect to cusp iterative solvers
  * multi-gpu / multi-node support for all such solvers

# Application areas #

Ocean modeling
  * Add sigma coordinate support to different solvers
  * vertically integrated N.S. equations (2d)
  * baroclinic / barotropic mode splitting solver

Others? (If interested, send mail to the list)