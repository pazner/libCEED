# libCEED.jl Docs

Documentation for the [libCEED.jl](https://github.com/pazner/libCEED.jl) Julia
interfce to the [libCEED](https://github.com/ceed/libceed) library.

For further information, see also the [libCEED
documentation](https://libceed.readthedocs.io/).

Several [short examples](Examples.md) are included to demonstrate the
functionality.

A key feature of libCEED.jl is the ability to [define user
Q-functions](UserQFunctions.md) natively in Julia.

## Advantages of a high-level interface for libCEED


#### User Q-functions

With libCEED.jl, it is much easier to write dimension-independent user-defined
Q-functions that automatically work on the GPU. See the [related
documentation](UserQFunctions.md) for more information.

#### Safe access to CeedVector objects

When accessing [`CeedVector`](@ref) objects, the C interface requires the user
to manually call `CeedVectorGetArray`, paired with `CeedVectorRestoreArray`. If
the user wants read-only access, then the user must call
`CeedVectorGetArrayRead`, paired with `CeedVectorRestoreArrayRead`. This can
possibly be bug-prone, because the user may forget to restore the array, or may
match the `Read` version to get the array with non-`Read` version to restore the
array (or vice versa).

In libCEED.jl, this difficulty is mitigated using the [`witharray`](@ref)
function and [`@witharray`](@ref) macro. There are also read-only versions,
[`witharray_read`](@ref) and [`@witharray_read`](@ref). When using this
functionality, it is impossible to forget to restore the array, and the correct
version is always paired properly.

For example, in `ex1-volume`, the following C code
```c
// Compute and print the sum of the entries of 'v' giving the mesh volume.
const CeedScalar *v_host;
CeedVectorGetArrayRead(v, CEED_MEM_HOST, &v_host);
CeedScalar vol = 0.;
for (CeedInt i = 0; i < sol_size; i++) {
  vol += v_host[i];
}
CeedVectorRestoreArrayRead(v, &v_host);
```
is replaced with the following equivalent Julia code
```julia
# Compute and print the sum of the entries of 'v' giving the mesh volume.
vol = witharray_read(sum, v, MEM_HOST)
```

In `ex2-surface`, the following C code
```c
// Initialize 'u' with sum of coordinates, x+y+z.
CeedScalar *u_host;
const CeedScalar *x_host;
CeedVectorGetArray(u, CEED_MEM_HOST, &u_host);
CeedVectorGetArrayRead(mesh_coords, CEED_MEM_HOST, &x_host);
for (CeedInt i = 0; i < sol_size; i++) {
  u_host[i] = 0;
  for (CeedInt d = 0; d < dim; d++)
    u_host[i] += x_host[i+d*sol_size];
}
CeedVectorRestoreArray(u, &u_host);
CeedVectorRestoreArrayRead(mesh_coords, &x_host);
```
is replaced with the following equivalent Julia code
```julia
@witharray_read(x_host=mesh_coords, size=(mesh_size÷dim, dim),
    @witharray(u_host=u, size=(sol_size,1),
        sum!(u_host, x_host)))
```
The macro version can provide better performance if a closure is required, and
allow for convenient reshaping of the vector into equivalently sized matrices
or tensors.

### Ceed objects
```@contents
Pages = [
   "Ceed.md",
   "CeedVector.md",
   "ElemRestriction.md",
   "Basis.md",
   "QFunction.md",
   "Operator.md",
]
```

### Utilities
```@contents
Pages = [
   "Misc.md",
   "Globals.md",
   "Quadrature.md",
]
```

```@contents
Pages = ["C.md"]
```
