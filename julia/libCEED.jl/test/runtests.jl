using Test, libCEED, LinearAlgebra

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
        v1 = rand(n)
        v[] = v1

        @test length(v) == n
        for p ∈ [1,2,Inf]
            @test norm(v,p) ≈ norm(v1,p)
        end
        @test witharray_read(sum, v) == sum(v1)
        reciprocal!(v)
        @test @witharray a=v all(a .== 1 ./v1)
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
    end
end
