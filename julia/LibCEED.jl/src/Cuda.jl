# COV_EXCL_START
using .CUDA

cuda_is_loaded = true

struct FieldsCuda
    inputs::NTuple{16,Int}
    outputs::NTuple{16,Int}
end

function generate_kernel(qf_name, kf, dims_in, dims_out)
    ninputs = length(dims_in)
    noutputs = length(dims_out)

    input_sz = prod.(dims_in)
    output_sz = prod.(dims_out)

    f_ins = [Symbol("rqi$i") for i = 1:ninputs]
    f_outs = [Symbol("rqo$i") for i = 1:noutputs]
    args = [f_ins; f_outs]

    def_ins = Vector{Expr}(undef, ninputs)
    f_ins_j = Vector{Union{Symbol,Expr}}(undef, ninputs)
    for i = 1:ninputs
        if length(dims_in[i]) == 0
            def_ins[i] = :(local $(f_ins[i]))
            f_ins_j[i] = f_ins[i]
        else
            def_ins[i] =
                :($(f_ins[i]) = LibCEED.MArray{Tuple{$(dims_in[i]...)},Float64}(undef))
            f_ins_j[i] = :($(f_ins[i])[j])
        end
    end
    def_outs = [
        :($(f_outs[i]) = LibCEED.MArray{Tuple{$(dims_out[i]...)},Float64}(undef))
        for i = 1:noutputs
    ]

    read_quads_in = [
        :(
            for j = 1:$(input_sz[i])
                $(f_ins_j[i]) = unsafe_load(
                    LibCEED.CUDA.DevicePtr(LibCEED.CuPtr{CeedScalar}(fields.inputs[$i])),
                    q + (j - 1)*Q,
                    a,
                )
            end
        ) for i = 1:ninputs
    ]

    write_quads_out = [
        :(
            for j = 1:$(output_sz[i])
                unsafe_store!(
                    LibCEED.CUDA.DevicePtr(LibCEED.CuPtr{CeedScalar}(fields.outputs[$i])),
                    $(f_outs[i])[j],
                    q + (j - 1)*Q,
                    a,
                )
            end
        ) for i = 1:noutputs
    ]

    qf = gensym(qf_name)

    quote
        function $qf(ctx_ptr, Q, fields)
            gd = LibCEED.gridDim()
            bi = LibCEED.blockIdx()
            bd = LibCEED.blockDim()
            ti = LibCEED.threadIdx()

            inc = bd.x*gd.x

            $(def_ins...)
            $(def_outs...)

            # Alignment for data read/write
            a = Val($(sizeof(CeedScalar)))

            for q = (ti.x+(bi.x-1)*bd.x):inc:Q
                $(read_quads_in...)
                $kf(ctx_ptr, CeedInt(1), $(args...))
                $(write_quads_out...)
            end
            return
        end
    end
end

function mk_cufunction(ceed, def_module, qf_name, kf, dims_in, dims_out)
    if !iscuda(ceed)
        return nothing
    end

    if !has_cuda()
        error("No valid CUDA installation found")
    end

    k_fn = Core.eval(def_module, generate_kernel(qf_name, kf, dims_in, dims_out))
    tt = Tuple{Ptr{Nothing},Int32,FieldsCuda}
    host_k = cufunction(k_fn, tt; maxregs=64)
    return host_k.fun.handle
end
# COV_EXCL_STOP
