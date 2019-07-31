# This file is based on a part of Julia. License is MIT: https://julialang.org/license

using QuasiArrays, Test

@testset "arrayops" begin
    @testset "basics" begin
        @test length(QuasiVector([1, 2, 3],0:0.5:1)) == 3
        @test count(!iszero, QuasiVector([1, 2, 3],0:0.5:1)) == 3

        let a = QuasiVector(fill(1., 4),0:0.5:1.5), b = a+a, c = a-a, d = a+a+a
            @test b[0.0] === 2. && b[0.5] === 2. && b[1.0] === 2. && b[1.5] === 2.
            @test c[0.0] === 0. && c[0.5] === 0. && c[1.0] === 0. && c[1.5] === 0.
            @test d[0.0] === 3. && d[0.5] === 3. && d[1.0] === 3. && d[1.5] === 3.
        end

        v = QuasiVector([1,2,3], 0:0.5:1)
        @test isequal(1 .+ v, QuasiVector([2,3,4], 0:0.5:1))
        @test isequal(v .+ 1, QuasiVector([2,3,4], 0:0.5:1))
        @test isequal(1 .- v, QuasiVector([0,-1,-2], 0:0.5:1))
        @test isequal(v .- 1, QuasiVector([0,1,2], 0:0.5:1))

        @test isequal(5*v, QuasiVector([5,10,15], 0:0.5:1))
        @test isequal(v*5, QuasiVector([5,10,15], 0:0.5:1))
        @test isequal(1 ./ v, QuasiVector([1.0,0.5,1/3], 0:0.5:1))
        @test isequal(v/5, QuasiVector([0.2,0.4,0.6], 0:0.5:1))

        @test isequal(2 .% v, QuasiVector([0,0,2], 0:0.5:1))
        @test isequal(v .% 2, QuasiVector([1,0,1], 0:0.5:1))
        @test isequal(2 .÷ v, QuasiVector([2,1,0], 0:0.5:1))
        @test isequal(v .÷ 2, QuasiVector([0,1,1], 0:0.5:1))
        @test isequal(-2 .% v, QuasiVector([0,0,-2], 0:0.5:1))
        @test isequal(-2 .÷ v, QuasiVector([-2,-1,0], 0:0.5:1))

        a = QuasiArray(fill(1.,2,2),(0:0.5:0.5,1:0.5:1.5))
        b = similar(a)
        @test axes(b) === axes(a)
        @test b.axes === a.axes
        copyto!(b,a)
        @test b == a

        a[0.0,1.0] = 1
        a[0.0,1.5] = 2
        a[0.5,1.0] = 3
        a[0.5,1.5] = 4
        b = copy(a')
        @test a[0.0,1.0] == 1. && a[0.0,1.5] == 2. && a[0.5,1.0] == 3. && a[0.5,1.5] == 4.
        @test b[1.0,0.0] == 1. && b[1.5,0.0] == 2. && b[1.0,0.5] == 3. && b[1.5,0.5] == 4.
        @test a[[0.0,0.5], [1.0,1.5]] == [1 2; 3 4]
        V = view(a, [0.0,0.5], [1.0,1.5])
        V[1,1] = 2
        @test a[0.0,1.0] == V[1,1] == V[CartesianIndex(1,1)] == 2
        V .= 1
        @test a.parent == fill(1,2,2)
        view(a, [0.0,0.5], [1.0,1.5]) .= 2
        @test a.parent == fill(2,2,2)
        a[[0.0 0.5], [1.0 1.5]] .= 1
        @test a.parent == fill(1.,2,2)
        a[[0.0 0.5], 1] .= 0
        @test a[0.0,1.0] == 0. && a[0.0,1.5] == 1. && a[0.5,1.0] == 0. && a[0.5,1.5] == 1.
        a[:, [1.0 1.5]] .= 2
        @test parent(a) == fill(2.,2,2)
    end

    @testset "construction" begin
        @test typeof(QuasiVector{Int}(undef, (1:3,))) == QuasiVector{Int,Tuple{UnitRange{Int}}}
        @test typeof(QuasiVector(undef, 1:3)) == QuasiVector{Any,Tuple{UnitRange{Int}}}
        @test typeof(QuasiMatrix{Int}(undef, 1:2,1:3)) == QuasiMatrix{Int,NTuple{2,UnitRange{Int}}}
        @test typeof(QuasiMatrix(undef, 1:2,1:3)) == QuasiMatrix{Any,NTuple{2,UnitRange{Int}}}

        @test size(QuasiVector{Int}(undef, 1:3)) == (3,)
        @test size(QuasiVector(undef, 1:3)) == (3,)
        @test size(QuasiMatrix{Int}(undef, 1:2,1:3)) == (2,3)
        @test size(QuasiMatrix(undef, 1:2,1:3)) == (2,3)

        @test_throws MethodError QuasiMatrix()
        @test_throws MethodError QuasiMatrix{Int}()
        @test_throws MethodError QuasiArray{Int,3}()
    end

    @testset "end" begin
        X = QuasiArray([ i+2j for i=1:5, j=1:5 ],(0:0.5:2,0:0.5:2))
        @test X[end,end] == 15
        @test_broken X[end]     == 15  # linear index
        @test X[0.5,  end] == 12
        @test X[end,  0.5] == 9
        @test X[end-0.5,0.5] == 8
        Y = ([2, 1, 4, 3] .- 1) ./ 2
        @test X[Y[end],0.0] == 5
        @test X[end,Y[end]] == 11
    end
end