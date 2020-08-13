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
        @testset "Apply" begin
            @testset "Quasi * Quasi" begin
                A = QuasiArray(rand(3,3),(0:0.5:1,0:0.5:1))
                M = ApplyQuasiArray(*, A, A)
                @test M == A*A
                @test M[[0,0.5], [0.5,1]] ≈ (A*A)[[0,0.5], [0.5,1]]
            end
            @testset "Quasi * Array" begin
                A = QuasiArray(rand(3,3),(0:0.5:1,Base.OneTo(3)))
                B = randn(3,3)
                M = ApplyQuasiArray(*, A, B)
                @test M ≈ A*B
                @test M[[0,0.5], [1,3]] ≈ (A*B)[[0,0.5], [1,3]]
                # Number * MulQuasiArray reduces array
                @test 2M ≈ M*2 ≈ 2A*B
                @test (2M).args[2] == (M*2).args[2] == 2B
                @test M/2 ≈ 2\M ≈ A*B/2
                @test (2\M).args[2] == (M/2).args[2] == B/2

                M = ApplyQuasiArray(*, B', A')
                @test M ≈ B'A'
                @test M[[1,3], [0,0.5]] ≈ (B'A')[[1,3], [0,0.5]]
                # Number * MulQuasiArray reduces array
                @test 2M ≈ M*2 ≈ 2B'A'
                @test (2M).args[1] == (M*2).args[1] == 2B'
                @test M/2 ≈ 2\M ≈ B'A'/2
                @test (2\M).args[1] == (M/2).args[1] == B'/2
        end
        @testset "MyQuasi" begin
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