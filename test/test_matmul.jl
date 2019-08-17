using QuasiArrays, Test
import QuasiArrays: apply

@testset "Multiplication" begin
    @testset "Diag * Inclusion" begin
        A = QuasiDiagonal(Inclusion(0:0.1:1))
        b = Inclusion(0:0.1:1)
        Ab = A*b
        @test Ab isa QuasiArray
        @test Ab[0.1] ≈ 0.1^2
        @test_throws DimensionMismatch A*Inclusion(1:2)
    end
    @testset "Quasi * Quasi" begin
        A = QuasiArray(rand(3,3),(0:0.5:1,0:0.5:1))
        @test A*A isa QuasiArray
        @test QuasiArray(A*A) == QuasiArray(A.parent*A.parent,A.axes)
    end
    @testset "Quasi * Array" begin
        A = QuasiArray(rand(3,3),(0:0.5:1,Base.OneTo(3)))
        B = rand(3,3)
        @test A*B isa QuasiArray
        @test QuasiArray(A*B) == QuasiArray(A.parent*B,A.axes)
        @test B*A' isa QuasiArray
        (A*B)[0.5,1]
    end

    @testset "Triple" begin
        A = QuasiArray(rand(3,3),(0:0.5:1,0:0.5:1))
        @test A isa QuasiArray
        @test Array(A*A*A) ≈ Array(A)*Array(A)*Array(A)
    end
end