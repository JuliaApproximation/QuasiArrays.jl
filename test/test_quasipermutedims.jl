using QuasiArrays, LinearAlgebra, Test


@testset "permutedims" begin
    A = QuasiArray(randn(5,2), (0:0.5:2, [1,4]))
    P = permutedims(A)
    @test P isa PermutedDimsQuasiArray
    @test size(P) == reverse(size(A))
    @test axes(P) == reverse(axes(A))
    @test P[4,0.5] == A[0.5,4]
    @test similar(P) isa QuasiArray
    P[4,0.5] = 2
    @test A[0.5,4] ≡ 2.0

    @testset "Diag of view" begin
        # this is used in ContinuumArrays
        D = Diagonal(view(A, 0.0, [1,4]))
        @test permutedims(D) ≡ D
    end

    @testset "vec" begin
        a = QuasiVector(randn(3),[0,2,3])
        @test permutedims(a) == a'
        @test permutedims(a .+ im) == transpose(a .+ im)
    end
end