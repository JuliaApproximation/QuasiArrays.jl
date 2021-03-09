using QuasiArrays, ArrayLayouts, LazyArrays, Test
import QuasiArrays: QuasiArrayLayout, QuasiArrayApplyStyle
import LazyArrays: ApplyStyle, MulStyle

@testset "Special functions" begin
    @testset "^" begin
        A = QuasiArray(randn(3,3), (0:0.5:1, 0:0.5:1))
        @test MemoryLayout(typeof(A)) isa QuasiArrayLayout
        @test ApplyStyle(*,typeof(A),typeof(A)) isa MulStyle
        @test A*A isa QuasiArray{Float64}
        @test A^2 == A^big(2) == A*A
        @test A^4 ≈ A*A*A*A
        @test A^big(2) isa QuasiArray{Float64}
        @test A^(-1) == inv(A) == QuasiArray(inv(A.parent), A.axes)

        V = view(A,:,:)
        @test V^1 == A
        @test V^2 == A^2
        @test V^3 == A^3
        @test_broken V^(-1) == inv(A)

        A = QuasiArray([1 2; 3 4], (0:0.5:0.5, 0:0.5:0.5))
        V = view(A,:,:)
        @test V^1 == A
        @test V^2 == A^2
    end
end