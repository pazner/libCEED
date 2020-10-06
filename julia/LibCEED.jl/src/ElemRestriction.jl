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
            destroy(x)
        end
        return obj
    end
end
destroy(r::ElemRestriction) = C.CeedElemRestrictionDestroy(r.ref) # COV_EXCL_LINE
Base.getindex(r::ElemRestriction) = r.ref[]
Base.show(io::IO, ::MIME"text/plain", e::ElemRestriction) =
    ceed_show(io, e, C.CeedElemRestrictionView)

@doc raw"""
    create_elem_restriction(
        ceed::Ceed,
        nelem,
        elemsize,
        ncomp,
        compstride,
        lsize,
        offsets::AbstractArray{CeedInt},
        mtype::MemType=MEM_HOST,
        cmode::CopyMode=COPY_VALUES,
    )

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
- `offsets`:    Array of shape `(elemsize, nelem)`. Column $i$ holds the ordered
                list of the offsets (into the input [`CeedVector`](@ref)) for
                the unknowns corresponding to element $i$, where $0 \leq i <
                \textit{nelem}$. All offsets must be in the range $[0,
                \textit{lsize} - 1]$.
- `mtype`:      Memory type of the `offsets` array, see [`MemType`](@ref)
- `cmode`:      Copy mode for the `offsets` array, see [`CopyMode`](@ref)
"""
function create_elem_restriction(
    c::Ceed,
    nelem,
    elemsize,
    ncomp,
    compstride,
    lsize,
    offsets::AbstractArray{CeedInt};
    mtype::MemType=MEM_HOST,
    cmode::CopyMode=COPY_VALUES,
)
    ref = Ref{C.CeedElemRestriction}()
    C.CeedElemRestrictionCreate(
        c[],
        nelem,
        elemsize,
        ncomp,
        compstride,
        lsize,
        mtype,
        cmode,
        offsets,
        ref,
    )
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

"""
    create_evector(r::ElemRestriction)

Return a new [`CeedVector`](@ref) E-vector.
"""
function create_evector(r::ElemRestriction)
    ref = Ref{C.CeedVector}()
    C.CeedElemRestrictionCreateVector(r[], C_NULL, ref)
    CeedVector(ref)
end

"""
    create_lvector(r::ElemRestriction)

Return a new [`CeedVector`](@ref) L-vector.
"""
function create_lvector(r::ElemRestriction)
    ref = Ref{C.CeedVector}()
    C.CeedElemRestrictionCreateVector(r[], ref, C_NULL)
    CeedVector(ref)
end

"""
    create_vectors(r::ElemRestriction)

Return an (L-vector, E-vector) pair.
"""
function create_vectors(r::ElemRestriction)
    l_ref = Ref{C.CeedVector}()
    e_ref = Ref{C.CeedVector}()
    C.CeedElemRestrictionCreateVector(r[], l_ref, e_ref)
    CeedVector(l_ref), CeedVector(e_ref)
end

"""
    getcompstride(r::ElemRestriction)

Get the L-vector component stride.
"""
function getcompstride(r::ElemRestriction)
    lsize = Ref{CeedInt}()
    C.CeedElemRestrictionGetCompStride(r[], lsize)
    lsize[]
end

"""
    getnumelements(r::ElemRestriction)

Get the total number of elements in the range of an [`ElemRestriction`](@ref).
"""
function getnumelements(r::ElemRestriction)
    result = Ref{CeedInt}()
    C.CeedElemRestrictionGetNumElements(r[], result)
    result[]
end

"""
    getelementsize(r::ElemRestriction)

Get the size of elements in the given [`ElemRestriction`](@ref).
"""
function getelementsize(r::ElemRestriction)
    result = Ref{CeedInt}()
    C.CeedElemRestrictionGetElementSize(r[], result)
    result[]
end

"""
    getlvectorsize(r::ElemRestriction)

Get the size of an L-vector for the given [`ElemRestriction`](@ref).
"""
function getlvectorsize(r::ElemRestriction)
    result = Ref{CeedInt}()
    C.CeedElemRestrictionGetLVectorSize(r[], result)
    result[]
end

"""
    getnumcomponents(r::ElemRestriction)

Get the number of components in the elements of an [`ElemRestriction`](@ref).
"""
function getnumcomponents(r::ElemRestriction)
    result = Ref{CeedInt}()
    C.CeedElemRestrictionGetNumComponents(r[], result)
    result[]
end

"""
    getmultiplicity(r::ElemRestriction, v::AbstractCeedVector)

Get the multiplicity of nodes in an [`ElemRestriction`](@ref). The
[`CeedVector`](@ref) `v` should be a L-vector (i.e. `length(v) ==
getlvectorsize(r)`).
"""
function getmultiplicity(r::ElemRestriction, v::AbstractCeedVector)
    @assert length(v) == getlvectorsize(r)
    C.CeedElemRestrictionGetMultiplicity(r[], v[])
end
