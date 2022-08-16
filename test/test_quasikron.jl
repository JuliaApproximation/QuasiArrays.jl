using QuasiArrays, DomainSets, StaticArrays, Test
import QuasiArrays: ArrayQuasiVector

@testset "Kron" begin
    @testset "InclusionKron" begin
        a = Inclusion(0:0.5:2)
        b = Inclusion([1,3,4])
        c = quasikron(a,b)
        @test eltype(c) == SVector{2,Float64}
        @test c[SVector(0.5,3)] == SVector(0.5,3)
        @test_throws BoundsError c[SVector(0.6,3)]
        @test_throws BoundsError c[SVector(0.5,2)]
    end

    @testset "QuasiKron" begin
        @testset "vec" begin
            a = QuasiVector(randn(5), 0:0.5:2)
            b = QuasiVector([5,6,8], [1,3,4])
            K = quasikron(a, b)
            @test axes(K) == (quasikron(axes(a,1),axes(b,1)),)
            @test K[SVector(0.5,3)] == a[0.5]b[3]
            @test_throws BoundsError K[SVector(0.6,3)]
            @test_throws BoundsError K[SVector(0.5,2)]
        end
        @testset "mat" begin
            a = QuasiMatrix(randn(5,3), 0:0.5:2, Base.OneTo(3))
            b = QuasiMatrix(randn(3,2), [1,3,4], Base.OneTo(2))
            K = QuasiKron(a,b)
            @test SVector(0.5,3) in axes(K,1)
            @test SVector(3,2) in axes(K,2)
            @test K[SVector(0.5,3), SVector(3,2)] ≈ a[0.5,3]*b[3,2]
        end
    end

    @testset "ArrayQuasiVector" begin
        A = randn(2,3)
        a = ArrayQuasiVector(A)
        for k in axes(A,1), j in axes(A,2)
            @test a[(k,j)] == A[k,j]
        end
        @test_throws BoundsError a[(3,1)]
        @test_throws BoundsError a[(1,4)]

        A = randn(2,3,4)
        a = ArrayQuasiVector(A)
        for k in axes(A,1), j in axes(A,2), l in axes(A,3)
            @test a[(k,j,l)] == A[k,j,l]
        end
        @test_throws BoundsError a[(3,1,1)]
        @test_throws BoundsError a[(1,4,1)]
        @test_throws BoundsError a[(1,1,5)]

        @testset "mul" begin
            a = QuasiMatrix(randn(5,3), 0:0.5:2, Base.OneTo(3))
            b = QuasiMatrix(randn(3,2), [1,3,4], Base.OneTo(2))
            K = QuasiKron(a,b)

            C = randn(3,2)
            c = ArrayQuasiVector(C)
            f = K*c
            @test axes(f,1) == axes(K,1)
            @test f[SVector(0.5,3)] ≈ a[0.5,:]'*C*b[3,:]
        end
    end
end