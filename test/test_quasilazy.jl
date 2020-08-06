using QuasiArrays, LazyArrays, ArrayLayouts, Test
import QuasiArrays: QuasiLazyLayout, QuasiArrayApplyStyle, LazyQuasiMatrix
import LazyArrays: MulStyle, ApplyStyle

struct MyQuasiLazyMatrix <: LazyQuasiMatrix{Float64}
    A::QuasiArray
end

Base.axes(A::MyQuasiLazyMatrix) = axes(A.A)
Base.getindex(A::MyQuasiLazyMatrix, x::Float64, y::Float64) = A.A[x,y]

@testset "LazyQuasiArray" begin
    @testset "*" begin
        A = MyQuasiLazyMatrix(QuasiArray(rand(3,3),(0:0.5:1,0:0.5:1)))
        B = QuasiArray(rand(3,3),(0:0.5:1,0:0.5:1))
        C = BroadcastQuasiArray(exp, B)
        
        @test MemoryLayout(A) isa QuasiLazyLayout
        @test ApplyStyle(*, typeof(A), typeof(A)) isa MulStyle
        @test ApplyStyle(*, typeof(A), typeof(B)) isa MulStyle

        @test A*A isa ApplyQuasiArray
        @test A*B isa ApplyQuasiArray
        @test A*B*C isa ApplyQuasiArray

        @test A*A == A.A*A
        @test A*B == A.A*B
        @test A*B*C ≈ A.A*B*C
    end
    @testset "^" begin
        A = QuasiArray(rand(3,3),(0:0.5:1,0:0.5:1))
        b = QuasiArray(randn(3), (A.axes[1],))
        B = QuasiArray(randn(3,3),A.axes)
        App = Applied(^, A, 2)
        @test ndims(App) == ndims(typeof(App)) == 2
        @test size(App) == size(A)
        @test axes(App) == axes(A)
        @test eltype(App) == Float64

        @test A^2 == QuasiArray(parent(A)^2, A.axes)
        A² = ApplyQuasiArray(^,A,2)
        @test eltype(A²) == Float64
        @test A² == A^2
        @test A^2 * b ≈ A²*b
        @test A^2 * B ≈ A²*B

        App = Applied(^, A, 2.5)
        @test eltype(App) == ComplexF64
        Ap = ApplyQuasiArray(^,A,2.5)
        @test eltype(Ap) == ComplexF64
        @test Ap == A^2.5
        @test A^2.5 * b ≈ Ap*b
        @test A^2.5 * B ≈ Ap*B
    end
end