using QuasiArrays, Test
import QuasiArrays: ApplyQuasiMatrix, UnionVcat, InclusionUnion

@testset "Concatenation" begin
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

    @testset "InclusionUnion" begin
        a = Inclusion(0:0.5:2)
        b = Inclusion([1,3,4])
        c = union(a,b)
        @test 0.5 in c
        @test 3 in c
        @test !(2.5 in c)
        @test c[0.5] ≡ 0.5
        @test c[3] ≡ c[3.0] ≡ 3.0
        @test_throws BoundsError c[2.5]
    end

    @test "UnionVcat" begin
        @testset "vector" begin
            a = QuasiVector(randn(5), 0:0.5:2)
            b = QuasiVector([5,6,8], [1,3,4])
            c = UnionVcat(a,b)
            @test axes(c,1) == InclusionUnion(axes(a,1),axes(b,1))
            @test c[0.5] ≡ a[0.5]
            @test c[3] ≡ c[3.0] ≡ 6.0
            @test_throws BoundsError c[2.5]
        end
        @testset "matrix" begin
            
        end
    end
end 