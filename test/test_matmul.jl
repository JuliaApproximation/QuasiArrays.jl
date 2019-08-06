using QuasiArrays, Test
import QuasiArrays: ApplyQuasiArray

@testset "Multiplication" begin
    A = QuasiDiagonal(Inclusion(0:0.1:1))
    b = Inclusion(0:0.1:1)
    Ab = A*b
    @test Ab isa ApplyQuasiArray
    @test Ab[0.1] ≈ 0.1^2
    @test_throws DimensionMismatch A*Inclusion(1:2)
end