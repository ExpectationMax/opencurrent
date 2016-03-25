## Discretization of the Laplacian ##

Let **u** be the unknown and **b** be the right hand side of a Poisson equation,

<img src='http://latex.codecogs.com/gif.latex?f_x \frac{\partial^2 u}{\partial x^2} + f_y \frac{\partial^2 u}{\partial y^2}  + f_z \frac{\partial^2 U}{\partial z^2} = B%.png' />

over a rectangular domain with grid spacing **h** and dimensions nx, ny, and nz.  At the faces of the domain (i.e. positive x, negative x, positive y, negative y, positive z, and negative z) there are boundary conditions, either Dirichelet or Neumann.

The discretization used is a discrete 7-point stencil.  This is simpler to write in indexless notation, so at a given point i,j,k, I will refer to the value of the **u** grid as **U**, and its 6 neighbors as **U<sub>E</sub>** (for East, or the point at i+1,j,k), **U<sub>W</sub>**,**U<sub>N</sub>**, **U<sub>S</sub>**, **U<sub>U</sub>**, **U<sub>D</sub>** (west, north, south, up, down).  In this notation, the stencil looks like this:

<img src='http://latex.codecogs.com/gif.latex?B =\frac{f_x \left((U_E - U) -(U - U_W)\right)}{h^2} + \frac{f_y \left((U_N - U) -(U - U_S)\right)}{h^2} + \frac{f_z \left((U_U - U) -(U - U_D)\right)}{h^2}%.png' />

which simplifies to

<img src='http://latex.codecogs.com/gif.latex?\frac{1}{h^2} (f_x (U_E + U_W) + f_y(U_N + U_S) + f_z(U_U + U_D) - (2f_x + 2f_y + 2f_z)U) = B%.png' />


Away from the boundary, the inner loop of the multigrid solver is a Gauss-Seidel relaxation step (see http://code.google.com/p/opencurrent/source/browse/src/ocuequation/sol_mgpressure3ddevf.cu) which involves simply solving for U to get

<img src='http://latex.codecogs.com/gif.latex?U = \frac{-h^2 B + f_x (U_E + U_W) + f_y(U_N + U_S) + f_z(U_U + U_D)}{(2f_x + 2f_y + 2f_z)}%.png' />

In the case of periodic boundary conditions, this is correct as well, with the additional logic of wrapping **U**'s neighbors around the domain periodically, so for example **U<sub>E</sub>** is i+1,j,k, but in the periodic case, it comes from (i+1)%nx,j,k.

## Ghost points ##

A common technique for dealing with boundary conditions is to simply allocate extra cells along each boundary of the domain, which are called "ghost points."  For example, rather than allocating a 3D array of size (nx,ny,nz), we allocate an array of size (nx+2,ny+2,nz+2).  And we call the first element in this array (-1,-1,-1) rather than (0,0,0).  This padding lets us access points that are 1 place outside of the domain, and we can fill them with arbitrary values.  In the case of periodic boundary conditions, for example, we would set ghost point (-1,j,k) = (nx-1,j,k), and (nx,j,k) = (0,j,k) for all j,k.

The advantage of this scheme is that we never need to put any kind of special handling of boundary conditions in our inner loops.  When a stencil requires reading from an out-of-bounds (ghost) point, it will simply read whatever value has been written into that ghost point previously.  And therefore to enforce different types of boundary conditions, we simply need to fill in the ghost points using different rules, but any code that relies on the stencil doesn't need to change.

To enforce Dirichelet conditions, assuming a cell-centered grid, the boundary value needs to be enforce half-way between cell 0 and cell -1.  In the following ascii art, diagram, cell 0 has value **U** (in its center).  In order to enforce that the interpolated value (at the cell edge) is 0, we need to set the ghost point to -**U**.

```
   -1     0
 +-----+-----+
 | -U  |  U  |
 |  .  |  .  |
 |     |     |
 +-----+-----+
```

In the more general case that we want to enforce that the interpolated value is A, then we must set the ghost point to -**U** + 2 **A** .  The half-way value is then

<img src='http://latex.codecogs.com/gif.latex?\frac{U + -U + 2A}{2} = A%.png' />


as desired.

The enforce Neumann boundary conditions that the derivative of **U** at the boundary is 0, we simply set the ghost point to U, as in this diagram:
```
   -1     0
 +-----+-----+
 |  U  |  U  |
 |  .  |  .  |
 |     |     |
 +-----+-----+
```
In the more general case that we want to the derivative to be **A**, we set it to **U** - h **A**.  Then the derivative is

<img src='http://latex.codecogs.com/gif.latex?\frac{U - (U-hA)}{h} = A%.png' />


as desired.

In all three cases (periodic, Dirichelet, and Neumann) we can simply enforce the boundary conditions via setting values on the ghost nodes.  For more complex or higher order conditions, we may require more ghost points.  Also, for a larger stencil, we will need a larger band of ghost points.  But in general, this is a good approach to enforcing boundary conditions in finite difference schemes.

One note about a CUDA implementation of these schemes: shifting the arrays by ghost points can result in unaligned loads and kill coalescing on G80 or require 2x the number of coalesced loads on GT200.  Therefore, case must be taken so that threadId values line up properly with 16-word boundaries.  Typically, this requires shifting the threadIdx over by the number of ghost points so that thread 0 operates on cell -1, thread 1 on cell 0, etc.

## Boundary Conditions for relaxation ##

The exception to the nice property that inner loops can remain boundary condition-unaware is code that inverts a stencil, such as the relaxation scheme outlined above.

We can rewrite the relaxation update in the following form:

<img src='http://latex.codecogs.com/gif.latex?U = U + \frac{-h^2 B + f_x (U_E + U_W) + f_y(U_N + U_S) + f_z(U_U + U_D) - (2f_x + 2f_y + 2f_z)U}{(2f_x + 2f_y + 2f_z) + M}%.png' />


All we've done is add **U** to the right hand side, and include an extra term

<img src='http://latex.codecogs.com/gif.latex?-(2f_x + 2f_y + 2f_z)U%.png' />


in the numerator, which then cancels out the additional **U** term.  Notice also the additional of a modifier **M** to the denominator.  In the case that we are not near any boundary points, or we have periodic boundary conditions, **M** is set to 0 and we have the original update equation.  However, in the case of Dirichelet or Neumann conditions, we can obtain the correctly modified update rules by varying **M** as follows.

For Dirichelet conditions, say we set **U<sub>E</sub>** = -**U** + 2 **A** as above.  Then the Laplacian operator becomes

<img src='http://latex.codecogs.com/gif.latex?\frac{1}{h^2} (f_x (-U + 2A + U_W) + f_y(U_N + U_S) + f_z(U_U + U_D) - (2f_x + 2f_y + 2f_z)U) = B%.png' />


and the correct update rule is now

<img src='http://latex.codecogs.com/gif.latex?U = \frac{-h^2 B + f_x (2A + U_W) + f_y(U_N + U_S) + f_z(U_U + U_D)}{(2f_x + 2f_y + 2f_z)}%.png' />


If we set the ghost point for **U<sub>E</sub>** to -**U** + 2 **A** and set **M** to -**f<sub>x</sub>**, the modified relaxation rule becomes

<img src='http://latex.codecogs.com/gif.latex?U = U + \frac{-h^2 B + f_x (-U + 2A + U_W) + f_y(U_N + U_S) + f_z(U_U + U_D) - (2f_x + 2f_y + 2f_z)U}{(2f_x + 2f_y + 2f_z) - f_x}%.png' />


which simplifies to exactly the correct rule.

For Neumann conditions, we set **U<sub>E</sub>** = **U** + h **A**.  The Laplacian is then

<img src='http://latex.codecogs.com/gif.latex?\frac{1}{h^2} (f_x (U + hA + U_W) + f_y(U_N + U_S) + f_z(U_U + U_D) - (2f_x + 2f_y + 2f_z)U) = B%.png' />


and the correct update rule is now

<img src='http://latex.codecogs.com/gif.latex?U = \frac{-h^2 B + f_x (hA + U_W) + f_y(U_N + U_S) + f_z(U_U + U_D)}{(3f_x + 2f_y + 2f_z)}.%.png' />


If we set the value as a ghost point, and set **M** = **f<sub>x</sub>**, the modified relaxation rule becomes

<img src='http://latex.codecogs.com/gif.latex?U = U + \frac{-h^2 B + f_x (U + hA + U_W) + f_y(U_N + U_S) + f_z(U_U + U_D) - (2f_x + 2f_y + 2f_z)U}{(2f_x + 2f_y + 2f_z) + f_x}%.png' />


which again simplifies to the correct rule.

By using the modified form of the update rule, we can achieve either Neumann or Dirichelet conditions by simply modifying the denominator.  This means that our inner loop can retain a single simple form, and we only need to adjust **M** when we are near a boundary to obtain different update rules depending on the boundary conditions.