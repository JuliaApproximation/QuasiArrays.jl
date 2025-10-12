using QuasiArrays, Test

@testset "quasireshapedarray" begin
    A = QuasiArray(randn(2,3), (0:0.5:0.5, 1:0.5:2))
    @test_throws ErrorException vec(A)
    @test_throws ErrorException reshape(A)
end