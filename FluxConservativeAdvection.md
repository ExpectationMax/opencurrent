# Flux-Conservative Advection #

The flux-conservative form of advection/conservation of momentum is nice because it allows you to exactly conserve integrals of a given quantity.

## Staggered "MAC" Grids ##

In 2D, the fluid velocity field can be thought of as 2 scalar fields, _U_ and _V_ (in 3D there is also _W_).  For simplicity, I will often write _u_ (lower-case "u") to represent the collection of all 2 (or 3) fields.  Rather than storing a regular array of vectors (i.e. U and V co-located), the three grids are shifted by a half-cell each.  So a single grid cell _i,j_ would look like this:

```
 +---Vij+1--+
 |          |
 Uij  Aij Ui+1j
 |          |
 +---Vij----+
```

Here, _A_<sub>i,j</sub> is a scalar value associated with the center of cell i,j and the _U_ velocities are on the center of the X-faces of cells, the _V_ velocities are on the centers of the Y-faces of cells.  Because of their position, _U_ and _V_ can be thought of as representing net flux of the fluid across the faces between adjacent cells (when multiplied by the surface area of the face).  For this reason, staggered grids naturally lead to a Finite Volume interpretation.

## Passive-scalar advection ##

Typically, a passive scalar _A_ will be stored in a cell-centered way as above.  The advection equation is normally written:

<img src='http://latex.codecogs.com/gif.latex?\frac{\partial A}{\partial t} = -(u \cdot \nabla) A  = -U \frac{\partial A}{\partial x} - V \frac{\partial A}{\partial y}%.png' />

The flux-conservative form of this equation is written as:

<img src='http://latex.codecogs.com/gif.latex?\frac{\partial A}{\partial t} = -\frac{\partial (U\cdot A)}{\partial x} -\frac{\partial (V\cdot A)}{\partial y} + A (\nabla \cdot u)%.png' />

This equivalence of the above two equations can be seen by application of the product rule.  When ''u'' has zero divergence (i.e. incompressible flow), the last term drops out and we are left with

<img src='http://latex.codecogs.com/gif.latex?\frac{\partial A}{\partial t} = -\frac{\partial (U\cdot A)}{\partial x} -\frac{\partial (V\cdot A)}{\partial y}%.png' />

It turns out this is very naturally discretized on a staggered grid.  The _dA/dt_ term need to be cell-centered.  Therefore _-dUA/dx_ needs to be cell-centered.  Which means that <img src='http://latex.codecogs.com/gif.latex?U\cdot A%.png' /> needs to be staggered in the x-direction by .5 grid cells, which aligns it with the _U_-grid.  So the only interpolation needed is to interpolate _A_ onto the _U_-grid.  This is where the finite-differencing scheme comes in - there are several options of this half-grid shift, and each option results in a different finite-difference scheme.  For this write-up, we'll just consider the simplest, first-order upwinding.  The interpolation function is:

<img src='http://latex.codecogs.com/gif.latex?A_{i+\frac{1}{2},j} \approx (u < 0) ? A_{i+1,j} : A_{i,j}%.png' />

So in C-pseudocode, here is what the advection routine looks like:

```
 float UpwindInterp(float U, float A0, float A1) {
   return U < 0? A1 : A0; 
 }
 for each i,j {
   float dUA = U[i+1,j] * UpwindInterp(U[i+1,j], A[i,j], A[i+1,j]) - U[i,j] * UpwindInterp(U[i,j], A[i-1,j], A[i,j]);
   float dVA = V[i,j+1] * UpwindInterp(V[i,j+1], A[i,j], A[i,j+1]) - V[i,j] * UpwindInterp(V[i,j], A[i,j-1], A[i,j]);
   dAdt[i,j] = -dUA/dx - dVA/dy;
 }
```

In 3D, the _W_ term is just added in a similar manner.
This is a simple 1st order upwind interpolation - typically, a more complex and wider stencil would be used.  But the basic code structure is unchanged.

## Momentum advection ##

Things get trickier when the quantity being advected is _u_ itself.  That is because we need to match each term of _u_ by each other term of _u_, for a total a 9 (in 3D) terms.

<img src='http://latex.codecogs.com/gif.latex?\frac{\partial U}{\partial t} = -\frac{\partial (U\cdot U)}{\partial x} -\frac{\partial (V\cdot U)}{\partial y} -\frac{\partial (W\cdot U)}{\partial z}%.png' />

<img src='http://latex.codecogs.com/gif.latex?\frac{\partial V}{\partial t} = -\frac{\partial (U\cdot V)}{\partial x} -\frac{\partial (V\cdot V)}{\partial y} -\frac{\partial (W\cdot V)}{\partial z}%.png' />

<img src='http://latex.codecogs.com/gif.latex?\frac{\partial W}{\partial t} = -\frac{\partial (U\cdot W)}{\partial x} -\frac{\partial (V\cdot W)}{\partial y} -\frac{\partial (W\cdot W)}{\partial z}%.png' />

The complexity comes in because the difference terms are all on the wrong grids.  Therefore, we need to interpolate between the 3 grids differently for each term, and the indexing
gets somewhat messy.  I'm going to write it out once and for all here, so hopefully I never need to rederive this again.
Below is what that looks like, again assuming there is a routine UpwindInterp as above:

```
 // dUdt calculated over the U-grid
 for each i,j,k {
    float u_iph_j_k = .5 * (u[i,j,k] + u[i+1,j,k]); 
    float u_imh_j_k = .5 * (u[i,j,k] + u[i-1,j,k]);
    float duu = (u_iph_j_k * Upwind(u_iph_j_k, u[i,j,k], u[i+1,j,k])) - 
                (u_imh_j_k * Upwind(u_imh_j_k, u[i-1mj,k], u[i,j,k]));
    float v_atu_i_jph_k = .5* (v[i  ,j+1,k  ] + v[i-1,j+1,k  ]);
    float v_atu_i_jmh_k = .5* (v[i  ,j  ,k  ] + v[i-1,j  ,k  ]);
    float dvu = (v_atu_i_jph_k * Upwind(v_atu_i_jph_k, u[i,j,k], u[i,j+1,k])) - 
                (v_atu_i_jmh_k * Upwind(v_atu_i_jmh_k, u[i,j-1,k], u[i,j,k]));
    float w_atu_i_j_kph = .5* (w[i  ,j  ,k+1] + w[i-1,j  ,k+1]);
    float w_atu_i_j_kmh = .5* (w[i  ,j  ,k  ] + w[i-1,j  ,k  ]);
    float dwu = (w_atu_i_j_kph * Upwind(w_atu_i_j_kph, u[i,j,k], u[i,j,k+1])) - 
                (w_atu_i_j_kmh * Upwind(w_atu_i_j_kmh, u[i,j,k-1], u[i,j,k]));   
    dudt[i,j,k] = -duu/dx - dvu/dy - dwu/dz;
 }

 for each i,j,k {
    float u_atv_iph_j_k = .5* (u[i+1,j  ,k  ] + u[i+1,j-1,k  ]);
    float u_atv_imh_j_k = .5* (u[i  ,j  ,k  ] + u[i  ,j-1,k  ]);
    float duv = (u_atv_iph_j_k * Upwind(u_atv_iph_j_k, v[i,j,k], v[i+1,j,k])) - 
                (u_atv_imh_j_k * Upwind(u_atv_imh_j_k, v[i-1,j,k], v[i,j,k]));
    float v_i_jph_k = .5 * (v[i,j,k] + v[i,j+1,k]); 
    float v_i_jmh_k = .5 * (v[i,j,k] + v[i,j-1,k]);
    float dvv = (v_i_jph_k * Upwind(v_i_jph_k, v[i,j,k], v[i,j+1,k])) - 
                (v_i_jmh_k * Upwind(v_i_jmh_k, v[i,j-1,k], v[i,j,k]));
    float w_atv_i_j_kph = .5* (w[i  ,j  ,k+1] + w[i  ,j-1,k+1]);
    float w_atv_i_j_kmh = .5* (w[i  ,j  ,k  ] + w[i  ,j-1,k  ]);
    float dwv = (w_atv_i_j_kph * Upwind(w_atv_i_j_kph, v[i,j,k], v[i,j,k+1])) - 
                (w_atv_i_j_kmh * Upwind(w_atv_i_j_kmh, v[i,j,k-1], v[i,j,k]))
    dvdt[i,j,k] = -duv/dx - dvv/dy - dwv/dz;
 }

 for each i,j,k {
    float u_atw_iph_j_k = .5* (u[i+1,j  ,k  ] + u[i+1,j  ,k-1]);
    float u_atw_imh_j_k = .5* (u[i  ,j  ,k  ] + u[i  ,j  ,k-1]);
    float duw = (u_atw_iph_j_k * Upwind(u_atw_iph_j_k, w[i,j,k], w[i+1,j,k])) - 
                (u_atw_imh_j_k * Upwind(u_atw_imh_j_k, w[i-1,j,k], w[i,j,k]));    
    float v_atw_i_jph_k = .5* (v[i  ,j+1,k  ] + v[i  ,j+1,k-1]);
    float v_atw_i_jmh_k = .5* (v[i  ,j  ,k  ] + v[i  ,j  ,k-1]);
    float dvw = (v_atw_i_jph_k * Upwind(v_atw_i_jph_k, w[i,j,k], w[i,j+1,k])) - 
                (v_atw_i_jmh_k * Upwind(v_atw_i_jmh_k, w[i,j-1,k], w[i,j,k]));    
    float w_i_j_kph = .5 * (w[i,j,k] + w[i,j,k+1]); 
    float w_i_j_kmh = .5 * (w[i,j,k] + w[i,j,k-1]);
    float dww = (w_i_j_kph * Upwind(w_i_j_kph, w[i,j,k], w[i,j,k+1])) - 
                (w_i_j_kmh * Upwind(w_i_j_kmh, w[i,j,k-1], w[i,j,k]));
    dwdt[i,j,k] = -duw/dx - dvw/dy - dww/dz;
 }
```

## Implementation ##

This is implemented in two different Solver classes, one for the passive advection case and one for the self-advection case.  The source code is at

http://code.google.com/p/opencurrent/source/browse/src/ocuequation/sol_selfadvection3d.h

and

http://code.google.com/p/opencurrent/source/browse/src/ocuequation/sol_passiveadvection3d.h