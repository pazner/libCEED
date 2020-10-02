using Test, libCEED, LinearAlgebra, StaticArrays

@testset "libCEED" begin
    @testset "Ceed" begin
        res = "/cpu/self/ref/serial"
        c = Ceed(res)
        @test isdeterministic(c)
        @test getresource(c) == res
        @test !iscuda(c)
        @test get_preferred_memtype(c) == MEM_HOST
        @test_throws libCEED.CeedError create_interior_qfunction(c, "")
        io = IOBuffer()
        show(io, MIME("text/plain"), c)
        @test String(take!(io)) == """
            Ceed
              Ceed Resource: $res
              Preferred MemType: host"""
    end

    @testset "Context" begin
        c = Ceed()
        data = zeros(3)
        ctx = Context(c, data)
        io = IOBuffer()
        show(io, MIME("text/plain"), ctx)
        @test String(take!(io)) == """
            CeedQFunctionContext
              Context Data Size: $(sizeof(data))"""
        @test_throws Exception set_data!(ctx, MEM_HOST, OWN_POINTER, data)
    end

    @testset "CeedVector" begin
        n = 10
        c = Ceed()
        v = CeedVector(c, n)
        @test length(v) == n
        @test axes(v) == (1:n,)
        @test ndims(v) == 1
        @test ndims(CeedVector) == 1

        v[] = 0.0
        @test @witharray(a = v, all(a .== 0.0))

        v1 = rand(n)
        v2 = CeedVector(c, v1)
        @test @witharray_read(a = v2, mtype = MEM_HOST, a == v1)
        @test Vector(v2) == v1
        v[] = v1
        for p ∈ [1, 2, Inf]
            @test norm(v, p) ≈ norm(v1, p)
        end
        @test_throws Exception norm(v, 3)
        @test witharray_read(sum, v) == sum(v1)
        reciprocal!(v)
        @test @witharray(a = v, mtype = MEM_HOST, all(a .== 1.0 ./ v1))

        witharray(x -> x .= 1.0, v)
        @test @witharray(a = v, all(a .== 1.0))

        @test CeedVectorActive()[] == libCEED.C.CEED_VECTOR_ACTIVE[]
        @test CeedVectorNone()[] == libCEED.C.CEED_VECTOR_NONE[]

        io = IOBuffer()
        summary(io, v)
        @test String(take!(io)) == "$n-element CeedVector"
        summary(io, v)
        println(io, ":")
        @witharray_read(a = v, Base.print_array(io, a))
        s1 = String(take!(io))
        show(io, MIME("text/plain"), v)
        @test s1 == String(take!(io))
    end

    @testset "Basis" begin
        c = Ceed()
        dim = 3
        ncomp = 1
        p = 4
        q = 6
        b = create_tensor_h1_lagrange_basis(c, dim, ncomp, p, q, GAUSS_LOBATTO)

        @test getdimension(b) == 3
        @test gettopology(b) == HEX
        @test getnumcomponents(b) == ncomp
        @test getnumnodes(b) == p^dim
        @test getnumnodes1d(b) == p
        @test getnumqpts(b) == q^dim
        @test getnumqpts1d(b) == q

        q1d, w1d = lobatto_quadrature(3, AbscissaAndWeights)
        @test q1d ≈ [-1.0, 0.0, 1.0]
        @test w1d ≈ [1/3, 4/3, 1/3]

        q1d, w1d = gauss_quadrature(3)
        @test q1d ≈ [-sqrt(3/5), 0.0, sqrt(3/5)]
        @test w1d ≈ [5/9, 8/9, 5/9]

        @test BasisCollocated()[] == libCEED.C.CEED_BASIS_COLLOCATED[]
    end

    @testset "Request" begin
        @test RequestImmediate()[] == libCEED.C.CEED_REQUEST_IMMEDIATE[]
        @test RequestOrdered()[] == libCEED.C.CEED_REQUEST_ORDERED[]
    end

    @testset "Misc" begin
        for dim = 1:3
            D = CeedDim(dim)
            J = rand(dim, dim)
            @test det(J, D) ≈ det(J)
            J = J + J' # make symmetric
            @test setvoigt(SMatrix{dim,dim}(J)) == setvoigt(J, D)
            @test getvoigt(setvoigt(J, D)) == J
            V = zeros(dim*(dim + 1)÷2)
            setvoigt!(V, J, D)
            @test V == setvoigt(J, D)
            J2 = zeros(dim, dim)
            getvoigt!(J2, V, D)
            @test J2 == J
        end
    end

    @testset "QFunction" begin
        c = Ceed()

        id = create_identity_qfunction(c, 1, EVAL_INTERP, EVAL_INTERP)
        Q = 10
        v = rand(Q)
        v1 = CeedVector(c, v)
        v2 = CeedVector(c, Q)
        apply!(id, Q, [v1], [v2])
        @test @witharray(a = v2, a == v)

        io = IOBuffer()
        show(io, MIME("text/plain"), create_interior_qfunction(c, "Poisson3DApply"))
        @test String(take!(io)) == """
            Gallery CeedQFunction Poisson3DApply
              2 Input Fields:
                Input Field [0]:
                  Name: "du"
                  Size: 3
                  EvalMode: "gradient"
                Input Field [1]:
                  Name: "qdata"
                  Size: 6
                  EvalMode: "none"
              1 Output Field:
                Output Field [0]:
                  Name: "dv"
                  Size: 3
                  EvalMode: "gradient\""""

        @interior_qf id2 = (c, (a, :in, EVAL_INTERP), (b, :out, EVAL_INTERP), b.=a)
        v2[] = 0.0
        apply!(id2, Q, [v1], [v2])
        @test @witharray(a = v2, a == v)

        dim = 3
        @interior_qf qf = (
            c,
            dim=dim,
            (a, :in, EVAL_GRAD, dim),
            (b, :in, EVAL_INTERP),
            (c, :out, EVAL_NONE),
            nothing,
        )
        in_sz, out_sz = libCEED.get_field_sizes(qf)
        @test in_sz == [dim, 1]
        @test out_sz == [1]
        @test QFunctionNone()[] == libCEED.C.CEED_QFUNCTION_NONE[]
    end

    @testset "Operator" begin
        c = Ceed()
        id = create_identity_qfunction(c, 1, EVAL_INTERP, EVAL_INTERP)
        b = create_tensor_h1_lagrange_basis(c, 3, 1, 3, 3, GAUSS_LOBATTO)
        n = getnumnodes(b)
        offsets = Vector{CeedInt}(0:n-1)
        er = create_elem_restriction(c, 1, n, 1, 1, n, MEM_HOST, COPY_VALUES, offsets)
        op = Operator(
            c;
            qf=id,
            fields=[
                (:input, er, b, CeedVectorActive()),
                (:output, er, b, CeedVectorActive()),
            ],
        )
        io = IOBuffer()
        show(io, MIME("text/plain"), op)
        @test String(take!(io)) == """
            CeedOperator
              2 Fields
              1 Input Field:
                Input Field [0]:
                  Name: "input"
                  Active vector
              1 Output Field:
                Output Field [0]:
                  Name: "output"
                  Active vector"""

        v = rand(n)
        v1 = CeedVector(c, v)
        v2 = CeedVector(c, n)
        apply!(op, v1, v2)
        @test @witharray_read(a1 = v1, @witharray_read(a2 = v2, a1 == a2))
    end
end
