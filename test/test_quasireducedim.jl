using QuasiArrays, Test

@testset "reducedim" begin
    A = QuasiArray(randn(2,3), (0:0.5:0.5, 1:0.5:2))
    @test sum(A) ≈ sum(A.parent)
    @test sum(A; dims=1) ≈ QuasiArray(sum(A.parent; dims=1), (1:1, 1:0.5:2))
    @test sum(A; dims=2) ≈ QuasiArray(sum(A.parent; dims=2), (0:0.5:0.5, 1:1))

    @test prod(A) ≈ prod(A.parent)
    @test prod(A; dims=1) ≈ QuasiArray(prod(A.parent; dims=1), (1:1, 1:0.5:2))
    @test prod(A; dims=2) ≈ QuasiArray(prod(A.parent; dims=2), (0:0.5:0.5, 1:1))

    @test any(isreal,A) ≈ any(isreal,A.parent)
    @test any(isreal,A; dims=1) ≈ QuasiArray(any(isreal,A.parent; dims=1), (1:1, 1:0.5:2))
    @test any(isreal,A; dims=2) ≈ QuasiArray(any(isreal,A.parent; dims=2), (0:0.5:0.5, 1:1))

    @test all(isreal,A) ≈ all(isreal,A.parent)
    @test all(isreal,A; dims=1) ≈ QuasiArray(all(isreal,A.parent; dims=1), (1:1, 1:0.5:2))
    @test all(isreal,A; dims=2) ≈ QuasiArray(all(isreal,A.parent; dims=2), (0:0.5:0.5, 1:1))

    @test minimum(A) ≈ minimum(A.parent)
    @test_broken minimum(A; dims=1) ≈ QuasiArray(minimum(A.parent; dims=1), (1:1, 1:0.5:2))
    @test_broken minimum(A; dims=2) ≈ QuasiArray(minimum(A.parent; dims=2), (0:0.5:0.5, 1:1))
end