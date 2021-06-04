using QuasiArrays, Test

@testset "reducedim" begin
    A = QuasiArray(randn(2,3), (0:0.5:0.5, 1:0.5:2))
    @test sum(A) ≈ sum(A.parent)
    @test sum(A; dims=1) ≈ QuasiArray(sum(A.parent; dims=1), (1:1, 1:0.5:2))
    @test sum(A; dims=2) ≈ QuasiArray(sum(A.parent; dims=2), (0:0.5:0.5, 1:1))
end