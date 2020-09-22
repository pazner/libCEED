struct FieldsCuda
    inputs::NTuple{16, Int}
    outputs::NTuple{16, Int}
end

function generate_kernel(qf_name, kf, dims_in, dims_out)
    ninputs = length(dims_in)
    noutputs = length(dims_out)

    input_sz = prod.(dims_in)
    output_sz = prod.(dims_out)

    f_ins = [Symbol("rqi$i") for i=1:ninputs]
    f_outs = [Symbol("rqo$i") for i=1:noutputs]

    args = [f_ins ; f_outs]

    def_ins = [:($(f_ins[i]) = MArray{Tuple{1,$(dims_in[i]...)},Float64}(undef)) for i=1:ninputs]
    def_outs = [:($(f_outs[i]) = MArray{Tuple{1,$(dims_out[i]...)},Float64}(undef)) for i=1:noutputs]

    read_quads_in = [
        :(for j=1:$(input_sz[i])
            $(f_ins[i])[j] = unsafe_load(CUDA.DevicePtr(CuPtr{CeedScalar}(fields.inputs[$i])), q + (j-1)*Q, a)
          end)
        for i=1:ninputs
    ]

    write_quads_out = [
        :(for j=1:$(output_sz[i])
            unsafe_store!(CUDA.DevicePtr(CuPtr{CeedScalar}(fields.outputs[$i])), $(f_outs[i])[j], q+(j-1)*Q, a)
          end)
        for i=1:noutputs
    ]

    qf = gensym(qf_name)

    quote
        function $qf(ctx_ptr, Q, fields)
            gd = gridDim()
            bi = blockIdx()
            bd = blockDim()
            ti = threadIdx()

            inc = bd.x*gd.x

            $(def_ins...)
            $(def_outs...)

            # Alignment for data read/write
            a = Val($(sizeof(CeedScalar)))

            for q=(ti.x + (bi.x-1)*bd.x):inc:Q
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
    tt = Tuple{Ptr{Nothing}, Int32, FieldsCuda}
    cufunction(k_fn, tt, maxregs=64)
end
