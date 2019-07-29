using QuasiArrays, Test
import QuasiArrays: Inclusion

@testset "AbstractQuasiArray" begin
    A = QuasiArray(rand(5,4,3), (range(0;stop=1,length=5), Base.OneTo(4), [2,3,6]))

    @testset "QuasiArray basics" begin
        @test_throws ArgumentError QuasiArray(rand(5,4,3), Inclusion.((range(0;stop=1,length=6), Base.OneTo(4), [1,3,4])))
        @test axes(A) == Inclusion.((range(0;stop=1,length=5), Base.OneTo(4), [2,3,6]))
        @test A[0.25,2,6] == A.parent[2,2,3]
        @test_throws BoundsError A[0.1,2,6]
    end

    @testset "Bounds checking" begin
        @test QuasiArrays.checkbounds(Bool, A, 0, 1, 2) == QuasiArrays.checkbounds(Bool, A, 0.0, 1, 2) == true
        @test QuasiArrays.checkbounds(Bool, A, 1, 4, 3) == true
        @test QuasiArrays.checkbounds(Bool, A, -1, 1, 1) == false
        @test QuasiArrays.checkbounds(Bool, A, 1, -1, 1) == false
        @test QuasiArrays.checkbounds(Bool, A, 1, 1, -1) == false
        @test QuasiArrays.checkbounds(Bool, A, 6, 4, 3) == false
        @test QuasiArrays.checkbounds(Bool, A, 5, 5, 3) == false
        @test QuasiArrays.checkbounds(Bool, A, 5, 4, 4) == false
        @test QuasiArrays.checkbounds(Bool, A, 0, 2, 2, 1) == true  # extra indices
        @test QuasiArrays.checkbounds(Bool, A, 2, 2, 2, 2) == false
        @test QuasiArrays.checkbounds(Bool, A, 1, 1)  == false
        @test QuasiArrays.checkbounds(Bool, A, 1, 12) == false
        @test QuasiArrays.checkbounds(Bool, A, 5, 12) == false
        @test QuasiArrays.checkbounds(Bool, A, 1, 13) == false
        @test QuasiArrays.checkbounds(Bool, A, 6, 12) == false
    end

    @testset "vector indices" begin
        @test QuasiArrays.checkbounds(Bool, A, 0:0.25:1, 1:4, [2,3,6]) == true
        @test QuasiArrays.checkbounds(Bool, A, 0:5, 1:4, 1:3) == false
        @test QuasiArrays.checkbounds(Bool, A, 1:5, 0:4, 1:3) == false
        @test QuasiArrays.checkbounds(Bool, A, 1:5, 1:4, 0:3) == false
        @test QuasiArrays.checkbounds(Bool, A, 1:6, 1:4, 1:3) == false
        @test QuasiArrays.checkbounds(Bool, A, 1:5, 1:5, 1:3) == false
        @test QuasiArrays.checkbounds(Bool, A, 1:5, 1:4, 1:4) == false
        @test QuasiArrays.checkbounds(Bool, A, 1:60) == true
        @test QuasiArrays.checkbounds(Bool, A, 1:61) == false
        @test QuasiArrays.checkbounds(Bool, A, 0.25, 2, 2, 1:1) == true  # extra indices
        @test QuasiArrays.checkbounds(Bool, A, 2, 2, 2, 1:2) == false
        @test QuasiArrays.checkbounds(Bool, A, 1:5, 1:4) == false
        @test QuasiArrays.checkbounds(Bool, A, 1:5, 1:12) == false
        @test QuasiArrays.checkbounds(Bool, A, 1:5, 1:13) == false
        @test QuasiArrays.checkbounds(Bool, A, 1:6, 1:12) == false
    end

    @testset "inferred axes/size" begin
        @inferred axes(A)
        @test @inferred(size(A)) == (5,4,3)
        @test @inferred(size(A, 2)) == 4
    end

    @testset "isinteger and isreal" begin
        v = QuasiArray(1:5, (range(0;stop=1,length=5),))
        @test all(isinteger, QuasiDiagonal(v))
        @test isreal(QuasiDiagonal(v))
    end

    @testset "unary ops" begin
        A = QuasiDiagonal(QuasiArray(1:5, (range(0;stop=1,length=5),)))
        @test +(A) == A
        @test *(A) == A
    end

    @testset "ndims and friends" begin
        A = QuasiDiagonal(QuasiArray(1:5, (range(0;stop=1,length=5),)))
        @test ndims(A) == 2
        @test ndims(QuasiDiagonal{Float64}) == 2
    end

    @testset "empty" begin
        v = QuasiArray([1, 2, 3],(2:4,))
        v2 = empty(v)
        v3 = empty(v, Float64)
        @test !isempty(v)
    end
end