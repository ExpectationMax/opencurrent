# Introduction #

The main numerical components of OpenCurrent are split between Solver objects and Equation objects.  The idea is that Solvers compute local differential quantities by reading and writing from individual buffers, while Equations solve entire equations by managing communication, boundary condition enforcement, data transfer, and the appropriate application of Solvers.  The point of this division is to split the code between local-aware code (Solvers), versus globally-aware Equations.  A single Solver might be useful in a number of different contexts, which encourages code reuse.

Solvers can operate on input buffers either in place or not, depending on their implementation.  The rule with solvers is that they should not perform any data movement on their input, since it is the job of the owning Equation object to move data around as is appropriate.  In other words, solvers can be thought of as simple operations that read from some buffers, perform a calculation, and write the results into another buffer.  (Currently the `Sol_MultigridPressure3DDeviceD` solver is an exception to this rule, but that needs to be revisited at some point.)

The Equation objects act at a higher level, orchestrating the Solvers in order to solve a particular equation across a particular set of execution units.  Equation objects should not modify buffers directly via CUDA routines or otherwise, but should call either grid-based routines or Solvers to modify values.

# Naming Convention #

All Solver and Equation objects are named according to a simple convention.  Solvers are named:

Sol`_` **Operator** **Dimension** D {**Host** | **Device**} {**D** | **F** }

where **Operator** is the operation performed, **Dimension** is the dimension grid that this operator works on (1D, 2D, or 3D), **Host** or **Device** indicates which execution unit this solver runs on, and the suffix indicates the precision of the output.  Solver names should not be named based on what precision they operate, but based on the precision of their output.  For example, the [Sol\_MultigridPressureMixed3DDeviceD](http://code.google.com/p/opencurrent/source/browse/src/ocuequation/sol_mgpressuremixed3d.h) solver internally runs at both single and double precision, but its output is double precision accuracy.  Therefore, it is suffixed with a **D**.

There may be multiple operators that compute the same term, but using different methods.  This is why the method name is often folded into the **Operator** specification.  For example, the `Sol_MultigridPressure3DDeviceD` and `Sol_PCGPressure3DDeviceD` solvers both calculate the same term, but using different methods (multigrid versus pre-conditioned conjugate gradients).

Equations are named similarly, except they are not required to specify device or host:

Eqn`_` **Equation** **Dimension** D {**D** | **F** }.

For example, the double precision incompressible Navier-Stokes solver is named `Eqn_IncompressibleNS3DD`.

All source files are prefixed with `eqn_` or `sol_` based on whether they define solvers or equations.  Originally, implementation was split between CUDA-specific routines (.cu files) and other routines (.cpp), but more recently all source code goes into .cu files.  Over time, the plan is to migrate all source files to be .cu and pass everything through the nvcc compiler.


# Solver Usage #

Solvers are subclasses of the [`ErrorHandler`](http://code.google.com/p/opencurrent/source/browse/src/ocuequation/error_handler.h) class which provides functionality for recording whether a run-time error happened, then querying this state.  Solver also contain a [`KernelWrapper`](http://code.google.com/p/opencurrent/source/browse/src/ocuutil/kernel_wrapper.h) member that provides functionality for recording timing infomation and checking CUDA return codes.


The lifecycle of a Solver object consists of:
  * Constructor - takes no arguments
  * Set global variables that control behavior (solver-dependent)
  * Initialize buffers that solver reads/writes (solver-dependent)
  * Call `initialize_storage`
  * Call `solve` (repeatedly)
  * (optionally) Extract results via data accessors

For example, the [Sol\_Gradient3DDevice](http://code.google.com/p/opencurrent/source/browse/src/ocuequation/sol_gradient3d.h) object computes the gradient of a scalar field.


(todo: walk through example)

# Equation Usage #

All Equation objects have a corresponding `Params` struct that is used to specify all of the parameters for the given equation, including its initial conditions, coefficients, numerical methods, and any other parameters that affect execution.

Equations are subclasses of the [`ErrorHandler`](http://code.google.com/p/opencurrent/source/browse/src/ocuequation/error_handler.h) class which provides functionality for recording whether a run-time error happened, then querying this state.

The lifecycle of an Equation object consists of:
  * Constructor - takes no arguments
  * Create corresponding `Params` object and initialize with appropriate values
  * Call `set_parameters()` passing in `Params` object.
  * Call `advance()` repeatedly to advance state of equation in time.
  * Extract results via data accessors.

Todo: finish this.


# List of Solvers #

Todo

Some notes here:
  * FluxConservativeAdvection
  * EnforcingBoundaryConditionsInMultigrid

# List of Equations #

Todo