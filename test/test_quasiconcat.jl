using QuasiArrays, DomainSets, Test
import QuasiArrays: ApplyQuasiMatrix, UnionVcat

@testset "Concatenation" begin
    @testset "Hcat" begin
        A = QuasiArray(randn(2,2), ([0,0.5], Base.OneTo(2)))
        B = [A A]
        @test B isa ApplyQuasiMatrix{Float64,typeof(hcat)}
        @test axes(B) == (axes(A,1), Base.OneTo(4))
        @test B[0.0,1] == B[0.0,3] == A[0.0,1]
        @test B == B
        @test B ≠ A
        @test B ≠ [A A A]
        @test_throws BoundsError B[0.1,1]
        @test_throws BoundsError B[0.0,5]
        @test QuasiArray(B) == B
    end

    @testset "Inclusion Union" begin
        a = Inclusion(0:0.5:2)
        b = Inclusion([1,3,4])
        c = union(a,b)
        @test 0.5 in c
        @test 3 in c
        @test !(2.5 in c)
        @test c[0.5] ≡ 0.5
        @test c[3] ≡ c[3.0] ≡ 3.0
        @test_throws BoundsError c[2.5]

        @test Inclusion{Float64}([1,2]) ∪ Inclusion{Float64}([3,4]) isa Inclusion{Float64}
    end

    @testset "UnionVcat" begin
        @testset "vector" begin
            a = QuasiVector(randn(5), 0:0.5:2)
            b = QuasiVector([5,6,8], [1,3,4])
            c = UnionVcat(a,b)
            @test axes(c,1) == Inclusion(union(0:0.5:2, [1,3,4]))
            @test c[0.5] ≡ a[0.5]
            @test c[3] ≡ c[3.0] ≡ 6.0
            @test_throws BoundsError c[2.5]
        end
        @testset "matrix" begin
            A = QuasiArray(randn(5,2), (0:0.5:2, [1,4]))
            B = QuasiArray(randn(3,2), (3:0.5:4, [1,4]))
            C = UnionVcat(A, B)
            @test C[0,1] == A[0,1]
            @test C[0.5,1] == A[0.5,1]
            @test C[0.5,4] == A[0.5,4]
            @test C[3,1] == B[3,1]
            @test C[3.5,1] == B[3.5,1]
            @test C[3.5,4] == B[3.5,4]
            @test_throws BoundsError C[2.5,1]
            @test_throws BoundsError C[3,2]

            @test_throws ArgumentError UnionVcat(A, QuasiArray(randn(3,1), (3:0.5:4, [1])))
        end
    end
end 