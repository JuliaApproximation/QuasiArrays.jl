using QuasiArrays, DomainSets, StaticArrays, Test

@testset "Kron" begin
    @testset "InclusionKron" begin
        a = Inclusion(0:0.5:2)
        b = Inclusion([1,3,4])
        c = quasikron(a,b)
        @test eltype(c) == SVector{2,Float64}
        @test c[SVector(0.5,3)] == SVector(0.5,3)
        @test_throws BoundsError c[SVector(0.6,3)]
        @test_throws BoundsError c[SVector(0.5,2)]
    end

    @testset "QuasiKron" begin
        a = QuasiVector(randn(5), 0:0.5:2)
        b = QuasiVector([5,6,8], [1,3,4])
        K = quasikron(a, b)
        @test axes(K) == (quasikron(axes(a,1),axes(b,1)),)
        @test K[SVector(0.5,3)] == a[0.5]b[3]
        @test_throws BoundsError K[SVector(0.6,3)]
        @test_throws BoundsError K[SVector(0.5,2)]
    end
end