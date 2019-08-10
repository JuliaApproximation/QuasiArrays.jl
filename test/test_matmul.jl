using QuasiArrays, Test
import QuasiArrays: ApplyQuasiArray

@testset "Multiplication" begin
    @testset "Diag * Inclusion" begin
        A = QuasiDiagonal(Inclusion(0:0.1:1))
        b = Inclusion(0:0.1:1)
        Ab = A*b
        @test Ab isa ApplyQuasiArray
        @test Ab[0.1] ≈ 0.1^2
        @test_throws DimensionMismatch A*Inclusion(1:2)
    end
    @testset "Quasi * Quasi" begin
        A = QuasiArray(rand(3,3),(0:0.5:1,0:0.5:1))
        @test A*A isa ApplyQuasiArray
        @test QuasiArray(A*A) == QuasiArray(A.parent*A.parent,A.axes)
    end
    @testset "Quasi * Array" begin
        A = QuasiArray(rand(3,3),(0:0.5:1,Base.OneTo(3)))
        B = rand(3,3)
        @test A*B isa ApplyQuasiArray
        @test QuasiArray(A*B) == QuasiArray(A.parent*B,A.axes)
    end

    @testset "Triple" begin
        A = QuasiArray(rand(3,3),(0:0.5:1,0:0.5:1))
        @test length((A*A*A).applied.args) == 3
        @test (A*A*A) === apply(*,A,A,A)
    end
end