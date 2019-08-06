using QuasiArrays, Test

@testset "AbstractQuasiArray" begin
    A = QuasiArray(rand(5,4,3), (range(0;stop=1,length=5), Base.OneTo(4), [2,3,6]))

    @testset "Convert" begin
        @test convert(typeof(A),A) === convert(AbstractQuasiArray{Float64},A) ===
                convert(AbstractQuasiArray{Float64,3},A) === A
        @test Array(A) == Array{Float64}(A) == Array{Float64,3}(A) ==
                convert(Array,A) == convert(Array{Float64},A) ==
                convert(Array{Float64,3},A) ==
                convert(AbstractArray,A) == convert(AbstractArray{Float64},A) ==
                convert(AbstractArray{Float64,3},A) == parent(A)
        @test QuasiArray(A) == QuasiArray{Float64}(A) == QuasiArray{Float64,3}(A) ==
                QuasiArray{Float64,3,typeof(A.axes)}(A) == copy(A) == A                
        v = QuasiArray(1:5, (range(0;stop=1,length=5),))
        @test Vector(v) == convert(AbstractVector,v) == Vector{Int}(v) ==
                    convert(AbstractVector{Int},v) == parent(v)
        @test QuasiVector(v) == v                    
        M = QuasiArray(rand(5,4), (range(0;stop=1,length=5), Base.OneTo(4)))
        @test Matrix(M) == convert(AbstractMatrix, M) == Matrix{Float64}(M) ==
                convert(AbstractMatrix{Float64}, M) == parent(M)
        @test QuasiMatrix(M) == M
    end

    @testset "QuasiArray basics" begin
        @test_throws ArgumentError QuasiArray(rand(5,4,3), (range(0;stop=1,length=6), Base.OneTo(4), [1,3,4]))
        @test_throws MethodError QuasiArray(rand(5,4,3), Inclusion.((range(0;stop=1,length=6), Base.OneTo(4), [1,3,4])))
        @test axes(A) == (Inclusion(range(0;stop=1,length=5)), Base.OneTo(4), Inclusion([2,3,6]))
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

    @testset "QuasiArray indexing" begin
        v = QuasiArray([1, 2, 3],(0:0.5:1,))
        @test axes(v) == axes(v[:]) == axes(v[Inclusion(0:0.5:1)]) == (Inclusion(0:0.5:1),)
        @test v[0.5] == v[:][0.5] == v[Inclusion(0:0.5:1)][0.5] == 2
    end
end
