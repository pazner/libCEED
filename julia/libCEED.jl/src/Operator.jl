mutable struct Operator
    ref::Ref{C.CeedOperator}
    qf::AbstractQFunction
    dqf::AbstractQFunction
    dqfT::AbstractQFunction
    function Operator(ref, qf, dqf, dqfT)
        obj = new(ref, qf, dqf, dqfT)
        finalizer(obj) do x
            # ccall(:jl_safe_printf, Cvoid, (Cstring, Cstring), "Finalizing %s.\n", repr(x))
            destroy(x)
        end
        return obj
    end
end
destroy(op::Operator) = C.CeedOperatorDestroy(op.ref)
Base.getindex(op::Operator) = op.ref[]
Base.show(io::IO, ::MIME"text/plain", op::Operator) = ceed_show(io, op, C.CeedOperatorView)

"""
    Operator(ceed::Ceed; qf, dqf=QFunctionNone(), dqfT=QFunctionNone(), fields)

Creates a libCEED `CeedOperator` object using the given Q-function `qf`, and
optionally its derivative and derivative transpose.

An array of fields must be provided, where each element of the array is a tuple
containing the name of the field (as a string or symbol), the corresponding
element restriction, basis, and vector.

# Examples

Create the operator that builds the Q-data associated with the mass matrix.
```
build_oper = Operator(
    ceed,
    qf=build_qfunc,
    fields=[
        (:J, mesh_restr, mesh_basis, CeedVectorActive()),
        (:w, ElemRestrictionNone(), mesh_basis, CeedVectorNone()),
        (:qdata, sol_restr_i, BasisCollocated(), CeedVectorActive())
    ]
)
```
"""
function Operator(c::Ceed; qf, dqf=QFunctionNone(), dqfT=QFunctionNone(), fields)
    op = Operator(c, qf, dqf, dqfT)
    for f ∈ fields
        set_field!(op, String(f[1]), f[2], f[3], f[4])
    end
    op
end

function Operator(c::Ceed, qf::AbstractQFunction, dqf::AbstractQFunction, dqfT::AbstractQFunction)
    ref = Ref{C.CeedOperator}()
    C.CeedOperatorCreate(c[], qf[], dqf[], dqfT[], ref)
    Operator(ref, qf, dqf, dqfT)
end

function set_field!(op::Operator, fieldname::AbstractString, r::AbstractElemRestriction, b::AbstractBasis, v::AbstractCeedVector)
    C.CeedOperatorSetField(op[], fieldname, r[], b[], v[])
end

"""
    apply!(op::Operator, vin, vout, request=RequestImmediate())

Apply the action of the operator `op` to the input vector `vin`, and store the
result in the output vector `vout`.

For non-blocking application, the user can specify a request object. By default,
immediate (synchronous) completion is requested.
"""
function apply!(op::Operator, vin::AbstractCeedVector, vout::AbstractCeedVector, request::AbstractRequest=RequestImmediate())
    try
        C.CeedOperatorApply(op[], vin[], vout[], request[])
    catch e
        # Cannot recover from exceptions in operator apply
        printstyled(stderr, "libCEED.jl: ", color=:red, bold=true)
        println("error occurred when applying operator")
        Base.display_error(stderr, Base.catch_stack())
        # Exit without running atexit hooks or finalizers
        ccall(:exit, Cvoid, (Cint,), 1)
    end
end
