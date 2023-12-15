using QuasiArrays, SparseArrays

@testset "sparse" begin
    @test !issparse(Inclusion([0,1]))
end