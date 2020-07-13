using QuasiArrays, Test
import QuasiArrays: InclusionKron

@testset "Kron" begin
    @testset "InclusionKron" begin
        a = Inclusion(0:0.5:2)
        b = Inclusion([1,3,4])
        c = InclusionKron(a,b)
        @test eltype(c) == Tuple{Float64,Int}
        @test c[(0.5,3)] == (0.5,3)
        @test_throws BoundsError c[(0.6,3)]
        @test_throws BoundsError c[(0.5,2)]
    end

    @testset "QuasiKron" begin
        a = QuasiVector(randn(5), 0:0.5:2)
        b = QuasiVector([5,6,8], [1,3,4])
        K = QuasiKron(a, b)
        @test axes(K) == (InclusionKron(axes(a,1),axes(b,1)),)
        @test K[(0.5,3)] == a[0.5]b[3]
        @test_throws BoundsError K[(0.6,3)]
        @test_throws BoundsError K[(0.5,2)]
    end
end