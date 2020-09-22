# Basis

```@docs
Basis
BasisCollocated
create_tensor_h1_lagrange_basis
create_tensor_h1_basis
create_h1_basis
apply!(b::Basis, nelem, tmode::TransposeMode, emode::EvalMode, u::libCEED.AbstractCeedVector, v::libCEED.AbstractCeedVector)
apply(c::Ceed, b::Basis, u::AbstractVector; nelem=1, tmode=NOTRANSPOSE, emode=EVAL_INTERP)
getdimension
gettopology
getnumcomponents
getnumnodes
getnumnodes1d
getnumqpts
getnumqpts1d
```
