struct UserQFunction{F,K}
    f::F
    fptr::Ptr{Nothing}
    kf::K
    cuf::Union{Nothing,CUDA.HostKernel}
end

function UserQFunction(ceed::Ceed, f, kf, cuf)
    UserQFunction(f, kf, fptr, cuf)
end

@inline function extract_context(ptr, ::Type{T}) where T
    unsafe_load(Ptr{T}(ptr))
end

@inline function extract_array(ptr, idx, dims)
    UnsafeArray(Ptr{CeedScalar}(unsafe_load(ptr, idx)), dims)
end

function generate_user_qfunction(ceed, def_module, qf_name, Q, constants, array_names, ctx, arrays, dims_in, dims_out, body)
    const_assignments = []
    for c ∈ constants
        push!(const_assignments, :($(c[1]) = $(c[2])))
    end

    qf1 = gensym(qf_name)
    f = Core.eval(def_module, quote
        @inline function $qf1(ctx_ptr::Ptr{Cvoid}, $Q::CeedInt, in_ptr::Ptr{Ptr{CeedScalar}}, out_ptr::Ptr{Ptr{CeedScalar}})
            $(const_assignments...)
            $ctx
            $arrays
            $body
            CeedInt(0)
        end
    end)
    f_qn = QuoteNode(f)
    rt = :CeedInt
    at = :(Core.svec(Ptr{Cvoid}, CeedInt, Ptr{Ptr{CeedScalar}}, Ptr{Ptr{CeedScalar}}))
    fptr = eval(Expr(:cfunction, Ptr{Cvoid}, f_qn, rt, at, QuoteNode(:ccall)))

    qf2 = gensym(qf_name)
    kf = Core.eval(def_module, quote
        @inline function $qf2(ctx_ptr::Ptr{Cvoid}, $Q::CeedInt, $(array_names...))
            $(const_assignments...)
            $ctx
            $body
            nothing
        end
    end)
    cuf = mk_cufunction(ceed, def_module, qf_name, kf, dims_in, dims_out)

    UserQFunction(f, fptr, kf, cuf)
end

function meta_user_qfunction(ceed, def_module, qf, Q, args)
    qf_name = Meta.quot(qf)
    Q_name = Meta.quot(Q)

    body = nothing
    ctx = nothing
    constants = []
    arrays = []
    dims_in = []
    dims_out = []
    names_in = []
    names_out = []

    for a ∈ args[1:end-1]
        if Meta.isexpr(a, :(=))
            a1 = Meta.quot(a.args[1])
            a2 = esc(a.args[2])
            push!(constants, :(($a1, $a2)))
        elseif Meta.isexpr(a, :tuple)
            arr_name = a.args[1]
            inout = a.args[2].value
            ndim = length(a.args) - 3
            dims = Vector{Expr}(undef, ndim)
            for d=1:ndim
                dims[d] = :(Int($(a.args[d+3])))
            end
            if inout == :in
                ptr = :in_ptr
                arr = dims_in
                i_inout = length(dims_in) + 1
                push!(names_in, arr_name)
            elseif inout == :out
                ptr = :out_ptr
                arr = dims_out
                i_inout = length(dims_out) + 1
                push!(names_out, arr_name)
            else
                error("Array specification must be either :in or :out. Given $inout.")
            end
            push!(arrays, :($arr_name = extract_array($ptr, $i_inout, ($(dims...),))))
            push!(arr, :(Int[$(esc.(a.args[5:end])...)]))
        elseif Meta.isexpr(a, :(::))
            ctx_name = a.args[1]
            ctx_type = a.args[2]
            ctx = Meta.quot(:($ctx_name = extract_context(ctx_ptr, $ctx_type)))
        else
            error("Bad argument to @user_qfunction")
        end
    end

    body = Meta.quot(args[end])

    arrays = Meta.quot(quote
        $(arrays...)
    end)

    arr_names = [names_in ; names_out]

    return :(generate_user_qfunction(
        $ceed,
        $def_module,
        $qf_name,
        $Q_name,
        [$(constants...)],
        $arr_names,
        $ctx,
        $arrays,
        [$(dims_in...)],
        [$(dims_out...)],
        $body
    ))
end

"""
    @interior_qf name=def

Creates a user-defined interior (volumentric) Q-function, and assigns it to a
variable named `name`. The definition of the Q-function is given as:
```
@interior_qf user_qf=(
    ceed::CEED, Q,
    [const1=val1, const2=val2, ...],
    [ctx::ContextType],
    (I1, :in, EvalMode, Q, dims...),
    (I2, :in, EvalMode, Q, dims...),
    (O1, :out, EvalMode, Q, dims...),
    body
)
```
In the above, `Q` is the name of a variable which will be bound to the number of
Q-points being operated on.

The definitions of form `const=val` are used for definitions which will be
compile-time constants in the Q-function. For example, if `dim` is a variable
set to the dimension of the problem, then `dim=dim` will make `dim` available in
the body of the Q-function as a compile-time constant.

If the user wants to provide a context struct to the Q-function, that can be
achieved by optionally including `ctx::ContextType`, where `ContextType` is the
type of the context struct, and `ctx` is the name to which is will be bound in
the body of the Q-function.

This is followed by the definition of the input and output arrays, which take
the form `(arr_name, (:in|:out), EvalMode, Q, dims...)`. Each array will be
bound to a variable named `arr_name`. Input arrays should be tagged with :in,
and output arrays with :out. An `EvalMode` should be specified, followed by the
dimensions of the array. The first dimension is always `Q`.

# Examples

- Q-function to compute the "Q-data" for the mass operator, which is given by
  the quadrature weight times the Jacobian determinant. The mesh Jacobian (the
  gradient of the nodal mesh points) and the quadrature weights are given as
  input arrays, and the Q-data is the output array. `dim` is given as a
  compile-time constant, and [`CeedDim`](@ref) is used to select a specialized
  determinant implementation for the given dimension. Because `dim` is a
  constant, the dispatch based on `CeedDim(dim)` is static (type stable). The
  `@view` macro is used to avoid allocations when accessing the slices.
```
@interior_qf build_qfunc = (
    ceed, Q, dim=dim,
    (J, :in, EVAL_GRAD, Q, dim, dim),
    (w, :in, EVAL_WEIGHT, Q),
    (qdata, :out, EVAL_NONE, Q),
    @inbounds @simd for i=1:Q
        qdata[i] = w[i]*det(@view(J[i,:,:]), CeedDim(dim))
    end)
```
"""
macro interior_qf(args)
    if !Meta.isexpr(args, :(=))
        error("@interior_qf must be of form `qf = (body)`")
    end

    qf = args.args[1]
    user_qf = esc(qf)
    args = args.args[2].args
    ceed = esc(args[1])

    fields_in = []
    fields_out = []
    for a ∈ args
        if Meta.isexpr(a, :tuple)
            field_name = String(a.args[1])
            inout = a.args[2].value
            evalmode = a.args[3]
            # Skip first dim (num qpts)
            ndim = length(a.args) - 4
            dims = Vector{Expr}(undef, ndim)
            for d=1:ndim
                dims[d] = esc(:(Int($(a.args[d+4]))))
            end
            sz_expr = :(prod(($(dims...),)))
            if inout == :in
                push!(fields_in, :(add_input!($user_qf, $field_name, $sz_expr, $evalmode)))
            elseif inout == :out
                push!(fields_out, :(add_output!($user_qf, $field_name, $sz_expr, $evalmode)))
            end
        end
    end

    gen_user_qf = meta_user_qfunction(ceed, __module__, qf, args[2], args[3:end])

    quote
        $user_qf = create_interior_qfunction($ceed, $gen_user_qf)
        $(fields_in...)
        $(fields_out...)
    end
end
