using QuasiArrays, Test


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
end