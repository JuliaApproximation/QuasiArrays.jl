using QuasiArrays, Test
import QuasiArrays: ApplyQuasiMatrix

@testset "Hcat" begin
    A = QuasiArray(randn(2,2), ([0,0.5], Base.OneTo(2)))
    B = [A A]
    @test B isa ApplyQuasiMatrix{Float64,typeof(hcat)}
    @test axes(B) == (axes(A,1), Base.OneTo(4))
    @test B[0.0,1] == B[0.0,3] == A[0.0,1]
    @test_throws BoundsError B[0.1,1]
    @test_throws BoundsError B[0.0,5]
    @test QuasiArray(B) == B
end