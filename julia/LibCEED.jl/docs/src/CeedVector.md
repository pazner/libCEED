# CeedVector

```@docs
CeedVector
setvalue!
Base.setindex!(v::CeedVector, v2::CeedScalar)
Base.setindex!(v::CeedVector, v2::AbstractArray)
Base.Vector(v::CeedVector)
LinearAlgebra.norm(v::CeedVector, n::NormType)
LinearAlgebra.norm(v::CeedVector, p::Real)
Base.length(v::CeedVector)
@witharray
@witharray_read
witharray
witharray_read
setarray!
sync_array!
take_array!
```
