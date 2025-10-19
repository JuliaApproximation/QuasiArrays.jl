using QuasiArrays, StatsBase, Random, Test

Random.seed!(243)

@testset "sample" begin
    w = QuasiVector([1,2,2,3,4], 0:0.5:2)
    @test sample(w) in axes(w,1)
    @test sample(Random.default_rng(), w) in axes(w,1)
    @test all(in(axes(w,1)), sample(w,1000))
    @test all(in(axes(w,1)), sample(Random.default_rng(),w,1000))
    @test mean(sample(w,10_000)) ≈ sum((0:0.5:2) .* parent(w))/sum(parent(w)) atol=1E-2

    W = QuasiMatrix([1; 2; 3 ;;], 0:0.5:1, Base.OneTo(1))
    @test sample(W) in axes(W,1)
    @test sample(Random.default_rng(), W) in axes(W,1)
    @test all(in(axes(W,1)), sample(W,5))
end

