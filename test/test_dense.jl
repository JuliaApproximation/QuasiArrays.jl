using QuasiArrays, Test
import QuasiArrays: MemoryLayout, QuasiArrayLayout, ApplyStyle, QuasiArrayApplyStyle

@testset "Special functions" begin
    @testset "^" begin
        A = QuasiArray(randn(3,3), (0:0.5:1, 0:0.5:1))
        @test MemoryLayout(typeof(A)) isa QuasiArrayLayout
        @test ApplyStyle(*,typeof(A),typeof(A)) isa QuasiArrayApplyStyle
        @test A*A isa QuasiArray{Float64}
        @test A^2 == A^big(2) == A*A
        @test A^4 == A*A*A*A
        @test A^big(2) isa QuasiArray{Float64}
        @test A^(-1) == inv(A) == QuasiArray(inv(A.parent), A.axes)
    end
end