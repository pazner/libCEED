using Test, libCEED, LinearAlgebra, StaticArrays

@testset "libCEED" begin
    @testset "Ceed" begin
        res = "/cpu/self/ref/serial"
        c = Ceed(res)
        @test isdeterministic(c)
        @test getresource(c) == res
        @test !iscuda(c)
        @test get_preferred_memtype(c) == MEM_HOST
    end

    @testset "CeedVector" begin
        n = 10
        c = Ceed()
        v = CeedVector(c, n)
        @test length(v) == n
        @test axes(v) == (1:n,)

        v[] = 0.0
        @test @witharray(a=v, all(a .== 0.0))

        v1 = rand(n)
        v2 = CeedVector(c, v1)
        @test @witharray_read(a=v2, mtype=MEM_HOST, a == v1)
        @test Vector(v2) == v1
        v[] = v1
        for p ∈ [1,2,Inf]
            @test norm(v,p) ≈ norm(v1,p)
        end
        @test_throws Exception norm(v,3)
        @test witharray_read(sum, v) == sum(v1)
        reciprocal!(v)
        @test @witharray(a=v, mtype=MEM_HOST, all(a .== 1.0./v1))

        witharray(x -> x .= 1.0, v)
        @test @witharray(a=v, all(a .== 1.0))

        @test CeedVectorActive()[] == libCEED.C.CEED_VECTOR_ACTIVE[]
        @test CeedVectorNone()[] == libCEED.C.CEED_VECTOR_NONE[]
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
        for dim=1:3
            D = CeedDim(dim)
            J = rand(dim,dim)
            @test det(J,D) ≈ det(J)
            J = J + J' # make symmetric
            @test setvoigt(SMatrix{dim,dim}(J)) == setvoigt(J,D)
            @test getvoigt(setvoigt(J,D),D) == J
            V = zeros(dim*(dim+1)÷2)
            setvoigt!(V, J, D)
            @test V == setvoigt(J,D)
        end
    end
end
