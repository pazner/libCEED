# ElemRestriction

```@docs
ElemRestriction
ElemRestrictionNone
create_elem_restriction
create_elem_restriction_strided
apply!(r::ElemRestriction, u::CeedVector, ru::CeedVector; tmode=NOTRANSPOSE, request=RequestImmediate())
create_evector
create_lvector
create_vectors
getcompstride
getnumelements
getelementsize
getlvectorsize
getnumcomponents(r::ElemRestriction)
getmultiplicity
```
