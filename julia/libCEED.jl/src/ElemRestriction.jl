abstract type AbstractElemRestriction end

"""
    ElemRestrictionNone()

Returns the singleton object corresponding to libCEED's
`CEED_ELEMRESTRICTION_NONE`
"""
struct ElemRestrictionNone <: AbstractElemRestriction end
Base.getindex(::ElemRestrictionNone) = C.CEED_ELEMRESTRICTION_NONE[]

"""
    ElemRestriction

Wraps a `CeedElemRestriction` object, representing the restriction from local
vectors to elements.
"""
mutable struct ElemRestriction <: AbstractElemRestriction
    ref::Ref{C.CeedElemRestriction}
    function ElemRestriction(ref)
        obj = new(ref)
        finalizer(obj) do x
            # ccall(:jl_safe_printf, Cvoid, (Cstring, Cstring), "Finalizing %s.\n", repr(x))
            C.CeedElemRestrictionDestroy(x.ref)
        end
        return obj
    end
end
Base.getindex(r::ElemRestriction) = r.ref[]
Base.show(io::IO, ::MIME"text/plain", e::ElemRestriction) = ceed_show(io, e, C.CeedElemRestrictionView)

@doc raw"""
    create_elem_restriction(ceed::Ceed, nelem, elemsize, ncomp, compstride, lsize, mtype::MemType, cmode::CopyMode, offsets::AbstractArray{CeedInt})

Create a `CeedElemRestriction`.

!!! warning "Zero-based indexing"
    In the below notation, we are using **0-based indexing**. libCEED expects
    the offset indices to be 0-based.

# Arguments:
- `ceed`:       The [`Ceed`](@ref) object
- `nelem`:      Number of elements described in the `offsets` array
- `elemsize`:   Size (number of "nodes") per element
- `ncomp`:      Number of field components per interpolation node (1 for scalar
                fields)
- `compstride`: Stride between components for the same L-vector "node". Data for
                node $i$, component $j$, element $k$ can be found in the
                L-vector at index `offsets[i + k*elemsize] + j*compstride`.
- `lsize`:      The size of the L-vector. This vector may be larger than the
                elements and fields given by this restriction.
- `mtype`:      Memory type of the `offsets` array, see [`MemType`](@ref)
- `cmode`:      Copy mode for the `offsets` array, see [`CopyMode`](@ref)
- `offsets`:    Array of shape `(elemsize, nelem)`. Column $i$ holds the ordered
                list of the offsets (into the input [`CeedVector`](@ref)) for
                the unknowns corresponding to element $i$, where $0 \leq i <
                \textit{nelem}$. All offsets must be in the range $[0,
                \textit{lsize} - 1]$.
"""
function create_elem_restriction(c::Ceed, nelem, elemsize, ncomp, compstride, lsize, mtype::MemType, cmode::CopyMode, offsets::AbstractArray{CeedInt})
    ref = Ref{C.CeedElemRestriction}()
    C.CeedElemRestrictionCreate(c[], nelem, elemsize, ncomp, compstride, lsize, mtype, cmode, offsets, ref)
    ElemRestriction(ref)
end

@doc raw"""
    reate_elem_restriction_strided(ceed::Ceed, nelem, elemsize, ncomp, lsize, strides)

Create a strided `CeedElemRestriction`.

!!! warning "Zero-based indexing"
    In the below notation, we are using **0-based indexing**. libCEED expects
    the offset indices to be 0-based.

# Arguments:
- `ceed`:     The [`Ceed`](@ref) object
- `nelem`:    Number of elements described by the restriction
- `elemsize`: Size (number of "nodes") per element
- `ncomp`:    Number of field components per interpolation node (1 for scalar
              fields)
- `lsize`:    The size of the L-vector. This vector may be larger than the
              elements and fields given by this restriction.
- `strides`:  Array for strides between [nodes, components, elements]. Data for
              node $i$, component $j$, element $k$ can be found in the L-vector
              at index `i*strides[0] + j*strides[1] + k*strides[2]`.
              [`STRIDES_BACKEND`](@ref) may be used with vectors created by a
              Ceed backend.
"""
function create_elem_restriction_strided(c::Ceed, nelem, elemsize, ncomp, lsize, strides)
    ref = Ref{C.CeedElemRestriction}()
    C.CeedElemRestrictionCreateStrided(c[], nelem, elemsize, ncomp, lsize, strides, ref)
    ElemRestriction(ref)
end
