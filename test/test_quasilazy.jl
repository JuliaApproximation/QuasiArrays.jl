using QuasiArrays, LazyArrays, ArrayLayouts, Base64, Test
import QuasiArrays: QuasiLazyLayout, QuasiArrayApplyStyle, LazyQuasiMatrix, LazyQuasiArrayStyle
import LazyArrays: MulStyle, ApplyStyle, arguments

struct MyQuasiLazyMatrix <: LazyQuasiMatrix{Float64}
    A::QuasiArray
end

Base.axes(A::MyQuasiLazyMatrix) = axes(A.A)
Base.getindex(A::MyQuasiLazyMatrix, x::Float64, y::Float64) = A.A[x,y]

@testset "LazyQuasiArray" begin
    @testset "sub" begin
        A = MyQuasiLazyMatrix(QuasiArray(rand(3,3),(0:0.5:1,0:0.5:1)))
        @test Base.BroadcastStyle(typeof(A)) isa LazyQuasiArrayStyle{2}
        @test Base.BroadcastStyle(typeof(view(A,0.5,0.5))) isa Base.Broadcast.DefaultArrayStyle{0}
        @test Base.BroadcastStyle(typeof(view(A,0:0.5:0.5,0.5))) isa Base.Broadcast.DefaultArrayStyle{1}
        @test Base.BroadcastStyle(typeof(view(A,0.5,0:0.5:0.5))) isa Base.Broadcast.DefaultArrayStyle{1}
    end
    @testset "*" begin
        @testset "Apply" begin
            @testset "Quasi * Quasi" begin
                A = QuasiArray(rand(3,3),(0:0.5:1,0:0.5:1))
                M = ApplyQuasiArray(*, A, A)
                @test M == A*A
                @test M[[0,0.5], [0.5,1]] ≈ (A*A)[[0,0.5], [0.5,1]]
                @test 2M ≈ M*2 ≈ 2A*A
                @test 2\M ≈ M/2 ≈ A*A/2
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

                @test colsupport(M, 1) == axes(M,1)
                @test rowsupport(M, 1) == axes(M,2)

                M = ApplyQuasiArray(*, B', A')
                @test M ≈ B'A'
                @test M[[1,3], [0,0.5]] ≈ (B'A')[[1,3], [0,0.5]]
                # Number * MulQuasiArray reduces array
                @test 2M ≈ M*2 ≈ 2B'A'
                @test (2M).args[1] == (M*2).args[1] == 2B'
                @test M/2 ≈ 2\M ≈ B'A'/2
                @test (2\M).args[1] == (M/2).args[1] == B'/2
            end
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
        @testset "(x .* D) * y" begin
            A = QuasiArray(rand(3,3),(0:0.5:1,0:0.5:1))
            Ã = MyQuasiLazyMatrix(A)
            x = QuasiArray(rand(3), (axes(A,1),))
            y = QuasiArray(rand(3), (axes(A,1),))

            @test BroadcastQuasiArray(*, x, A) * y ≈ BroadcastQuasiArray(*, x, Ã) * y ≈ (x .* A) * y
            @test BroadcastQuasiArray(*, A, x) * y ≈ BroadcastQuasiArray(*, Ã, x) * y ≈ (x .* A) * y
            @test BroadcastQuasiArray(*, A, x)^2 ≈ (x .* A)^2

            @test BroadcastQuasiArray(*, x, ApplyQuasiArray(^, A, 2)) * y ≈ (x .* A^2) * y
        end

        @testset "summary" begin
            A = ApplyQuasiArray(*, ones(Inclusion([1,2,3]), Inclusion([4,5])), fill(2,Inclusion([4,5])))
            @test stringmime("text/plain", A) == "(ones(Inclusion([1, 2, 3]), Inclusion([4, 5]))) * (fill(2, Inclusion([4, 5])))"
        end

        @testset "sub *" begin
            A = QuasiArray(rand(3,3),(0:0.5:1,Base.OneTo(3)))
            B = randn(3,3)
            M = ApplyQuasiArray(*, A, B)
            @test arguments(view(M,0.5,:))[1] == B'
            @test arguments(view(M,0.5,:))[2] == A[0.5,:]
            @test arguments(view(M,:,2))[1] == A
            @test arguments(view(M,:,2))[2] == B[:,2]
        end
    end
    @testset "\\" begin
        A = QuasiArray(rand(3,3),(0:0.5:1,0:0.5:1))
        b = QuasiArray(rand(3), (axes(A,1),))
        l = MyQuasiLazyMatrix(A) \ b
        @test l isa ApplyQuasiArray
        @test l[0.] ≈ (A\b)[0.]
        L = MyQuasiLazyMatrix(A) \ A
        @test L[0.,0.] ≈ 1
        MyQuasiLazyMatrix(A) \ ApplyArray(*, A, A)
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
    @testset "Broadcast" begin
        A = MyQuasiLazyMatrix(QuasiArray(rand(3,3),(0:0.5:1,0:0.5:1)))
        @test exp.(A) isa BroadcastQuasiArray
        @test exp.(A) == exp.(A.A)
        @test exp.(exp.(A) .+ 1) isa BroadcastQuasiArray
        @test exp.(exp.(A) .+ 1) == exp.(exp.(A.A) .+ 1)
    end
end