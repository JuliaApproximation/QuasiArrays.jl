using QuasiArrays, Test

@testset "QuasiSubArray" begin
    @testset "basics" begin
        A = QuasiArray(reshape(1:60, 5,4,3), (range(0;stop=1,length=5), Base.OneTo(4), [2,3,6]))
        sA = view(A, 0.25:0.25, 1:4, :)
        @test @inferred(size(sA)) == (1, 4, 3)
        @test parent(sA) == A
        @test parentindices(sA) == (0.25:0.25, 1:4, Inclusion([2,3,6]))
        @test size(sA) == (1, 4, 3)
        @test axes(sA) == (Base.OneTo(1), Base.OneTo(4), Inclusion([2,3,6]))
        @test sA[1, 2, [2,3,6]][:] == [7,27,47]
        sA = view(A, 0.0:0.25:1, 1:4, 6)
        sA[1:5,1:4] .= -2
        @test all(A[:,:,6] .== -2)
        fill!(sA, -3)
        @test all(A[:,:,6] .== -3)
        sA = view(A, 0.0:0.25:1, 3:3, 2:3)
        @test size(sA) == (5,1,2)
        @test axes(sA) === (Base.OneTo(5), Base.OneTo(1), Base.OneTo(2))
        @test convert(AbstractArray,sA) == convert(AbstractArray{Int},sA) ==
                Array(sA) == Array{Int}(sA) == A[0.0:0.25:1,3:3,2:3]

        @test Matrix(view(sA,1:1,1:1,1))[1] == sA[1,1,1] == 11
        # Test with mixed types
        @test sA[:, Int16(1), [big(1),2]] == [11:15 31:35]
        sA = view(A, 0.25:0.25, 1:4, [2 3; 2 6])
        @test ndims(sA) == 4
        @test axes(sA) === (Base.OneTo(1), Base.OneTo(4), Base.OneTo(2), Base.OneTo(2))
        @test_throws BoundsError view(A, 0:0.25:0.25, 3, [1 3; 4 2])
        sA = view(A, 0:0.25:0.25, 3, [2 3; 2 6])
        @test ndims(sA) == 3
        @test axes(sA) === (Base.OneTo(2), Base.OneTo(2), Base.OneTo(2))
    end

    # @testset "logical indexing #4763" begin
    #     A = view(QuasiVector(1:5,range(0,2;length=5)), 0.5:0.5:1.5)
    #     A[Inclusion(0.5:0.5:1.0)]
    #     @test A[A.<3] == view(A, A.<7) == [5, 6]
    #     @test Base.unsafe_getindex(A, A.<7) == [5, 6]
    #     B = reshape(1:16, 4, 4)
    #     sB = view(B, 2:3, 2:3)
    #     @test sB[sB.>8] == view(sB, sB.>8) == [10, 11]
    #     @test Base.unsafe_getindex(sB, sB.>8) == [10, 11]
    # end
    #
    # # Tests where dimensions are dropped
    # A = copy(reshape(1:120, 3, 5, 8))
    # sA = view(A, 2, :, 1:8)
    # @test parent(sA) == A
    # @test parentindices(sA) == (2, Base.Slice(1:5), 1:8)
    # @test Base.parentdims(sA) == [2:3;]
    # @test size(sA) == (5, 8)
    # @test axes(sA) === (Base.OneTo(5), Base.OneTo(8))
    # @test @inferred(strides(sA)) == (3,15)
    # @test sA[2, 1:8][:] == [5:15:120;]
    # @test sA[:,1] == [2:3:14;]
    # @test sA[2:5:end] == [5:15:110;]
    # sA[2:5:end] .= -1
    # @test all(sA[2:5:end] .== -1)
    # @test all(A[5:15:120] .== -1)
    # test_bounds(sA)
    # sA = view(A, 1:3, 1:5, 5)
    # @test Base.parentdims(sA) == [1:2;]
    # @test size(sA) == (3,5)
    # @test axes(sA) === (Base.OneTo(3),Base.OneTo(5))
    # @test @inferred(strides(sA)) == (1,3)
    # test_bounds(sA)
    # sA = view(A, 1:2:3, 3, 1:2:8)
    # @test Base.parentdims(sA) == [1,3]
    # @test size(sA) == (2,4)
    # @test axes(sA) === (Base.OneTo(2), Base.OneTo(4))
    # @test @inferred(strides(sA)) == (2,30)
    # @test sA[:] == A[sA.indices...][:]
    # test_bounds(sA)
    #
    # let a = [5:8;]
    #     @test parent(a) == a
    #     @test parentindices(a) == (1:4,)
    # end
    #
    # # issue #6218 - logical indexing
    # A = rand(2, 2, 3)
    # msk = fill(true, 2, 2)
    # msk[2,1] = false
    # sA = view(A, :, :, 1)
    # sA[msk] .= 1.0
    # @test sA[msk] == fill(1, count(msk))
    #
    # # bounds checking upon construction; see #4044, #10296
    # @test_throws BoundsError view(1:10, 8:11)
    # A = reshape(1:20, 5, 4)
    # sA = view(A, 1:2, 1:3)
    # @test_throws BoundsError view(sA, 1:3, 1:3)
    # @test_throws BoundsError view(sA, 1:2, 1:4)
    # view(sA, 1:2, 1:2)
    # @test_throws BoundsError view(A, 17:23)
    # view(A, 17:20)
    #
    # # Linear indexing by one multidimensional array:
    # A = reshape(1:120, 3, 5, 8)
    # sA = view(A, :, :, :)
    # @test sA[[72 17; 107 117]] == [72 17; 107 117]
    # @test sA[[99 38 119 14 76 81]] == [99 38 119 14 76 81]
    # @test sA[[fill(1, (2, 2, 2)); fill(2, (2, 2, 2))]] == [fill(1, (2, 2, 2)); fill(2, (2, 2, 2))]
    # sA = view(A, 1:2, 2:3, 3:4)
    # @test sA[(1:8)'] == [34 35 37 38 49 50 52 53]
    # @test sA[[1 2 4 4; 6 1 1 4]] == [34 35 38 38; 50 34 34 38]
    #
    # # issue #11871
    # let a = fill(1., (2,2)),
    #     b = view(a, 1:2, 1:2)
    #     b[2] = 2
    #     @test b[2] === 2.0
    # end
    #
    # # issue #15138
    # let a = [1,2,3],
    #     b = view(a, UInt(1):UInt(2))
    #     @test b == view(a, UInt(1):UInt(2)) == view(view(a, :), UInt(1):UInt(2)) == [1,2]
    # end
    #
    # let A = reshape(1:4, 2, 2),
    #     B = view(A, :, :)
    #     @test parent(B) === A
    #     @test parent(view(B, 0x1, :)) === parent(view(B, 0x1, :)) === A
    # end
    #
    # # issue #15168
    # let A = rand(10), sA = view(copy(A), :)
    #     @test sA[Int16(1)] === sA[Int32(1)] === sA[Int64(1)] === A[1]
    #     permute!(sA, Vector{Int16}(1:10))
    #     @test A == sA
    # end
    #
    # # the following segfaults with LLVM 3.8 on Windows, ref #15417
    # @test Array(view(view(reshape(1:13^3, 13, 13, 13), 3:7, 6:6, :), 1:2:5, :, 1:2:5)) ==
    #     cat([68,70,72],[406,408,410],[744,746,748]; dims=3)
    #
    # # tests @view (and replace_ref_end!)
    # X = reshape(1:24,2,3,4)
    # Y = 4:-1:1
    #
    # @test isa(@view(X[1:3]), SubArray)
    #
    # @test X[1:end] == @.(@view X[1:end]) # test compatibility of @. and @view
    # @test X[1:end-3] == @view X[1:end-3]
    # @test X[1:end,2,2] == @view X[1:end,2,2]
    # @test X[1,1:end-2,1] == @view X[1,1:end-2,1]
    # @test X[1,2,1:end-2] == @view X[1,2,1:end-2]
    # @test X[1,2,Y[2:end]] == @view X[1,2,Y[2:end]]
    # @test X[1:end,2,Y[2:end]] == @view X[1:end,2,Y[2:end]]
    #
    # u = (1,2:3)
    # @test X[u...,2:end] == @view X[u...,2:end]
    # @test X[(1,)...,(2,)...,2:end] == @view X[(1,)...,(2,)...,2:end]
    #
    # # test macro hygiene
    # let size=(x,y)-> error("should not happen"), Base=nothing
    #     @test X[1:end,2,2] == @view X[1:end,2,2]
    # end
    #
    # # test that side effects occur only once
    # let foo = [X]
    #     @test X[2:end-1] == @view (push!(foo,X)[1])[2:end-1]
    #     @test foo == [X, X]
    # end
    #
    # # test @views macro
    # @views let f!(x) = x[1:end-1] .+= x[2:end].^2
    #     x = [1,2,3,4]
    #     f!(x)
    #     @test x == [5,11,19,4]
    #     @test x[1:3] isa SubArray
    #     @test x[2] === 11
    #     @test Dict((1:3) => 4)[1:3] === 4
    #     x[1:2] .= 0
    #     @test x == [0,0,19,4]
    #     x[1:2] .= 5:6
    #     @test x == [5,6,19,4]
    #     f!(x[3:end])
    #     @test x == [5,6,35,4]
    #     x[Y[2:3]] .= 7:8
    #     @test x == [5,8,7,4]
    #     x[(3,)..., ()...] += 3
    #     @test x == [5,8,10,4]
    #     i = Int[]
    #     # test that lhs expressions in update operations are evaluated only once:
    #     x[push!(i,4)[1]] += 5
    #     @test x == [5,8,10,9] && i == [4]
    #     x[push!(i,3)[end]] += 2
    #     @test x == [5,8,12,9] && i == [4,3]
    #     @. x[3:end] = 0       # make sure @. works with end expressions in @views
    #     @test x == [5,8,0,0]
    # end
    # @views @test isa(X[1:3], SubArray)
    # @test X[1:end] == @views X[1:end]
    # @test X[1:end-3] == @views X[1:end-3]
    # @test X[1:end,2,2] == @views X[1:end,2,2]
    # @test X[1,2,1:end-2] == @views X[1,2,1:end-2]
    # @test X[1,2,Y[2:end]] == @views X[1,2,Y[2:end]]
    # @test X[1:end,2,Y[2:end]] == @views X[1:end,2,Y[2:end]]
    # @test X[u...,2:end] == @views X[u...,2:end]
    # @test X[(1,)...,(2,)...,2:end] == @views X[(1,)...,(2,)...,2:end]
    # # test macro hygiene
    # let size=(x,y)-> error("should not happen"), Base=nothing
    #     @test X[1:end,2,2] == @views X[1:end,2,2]
    # end
    #
    # # issue #18034
    # # ensure that it is possible to create an isbits, IndexLinear view of an immutable Array
    # let
    #     struct ImmutableTestArray{T, N} <: Base.DenseArray{T, N}
    #     end
    #     Base.size(::Union{ImmutableTestArray, Type{ImmutableTestArray}}) = (0, 0)
    #     Base.IndexStyle(::Union{ImmutableTestArray, Type{ImmutableTestArray}}) = Base.IndexLinear()
    #     a = ImmutableTestArray{Float64, 2}()
    #     @test Base.IndexStyle(view(a, :, :)) == Base.IndexLinear()
    #     @test isbits(view(a, :, :))
    # end
    #
    # # ref issue #17351
    # @test @inferred(reverse(view([1 2; 3 4], :, 1), dims=1)) == [3, 1]
    #
    # let
    #     s = view(reshape(1:6, 2, 3), 1:2, 1:2)
    #     @test @inferred(s[2,2,1]) === 4
    # end
    #
    # # issue #18581: slices with OneTo axes can be linear
    # let
    #     A18581 = rand(5, 5)
    #     B18581 = view(A18581, :, axes(A18581,2))
    #     @test IndexStyle(B18581) === IndexLinear()
    # end
    #
    # @test sizeof(view(zeros(UInt8, 10), 1:4)) == 4
    # @test sizeof(view(zeros(UInt8, 10), 1:3)) == 3
    # @test sizeof(view(zeros(Float64, 10, 10), 1:3, 2:6)) == 120
    #
    # # PR #25321
    # # checks that issue in type inference is resolved
    # A = rand(5,5,5,5)
    # V = view(A, 1:1 ,:, 1:3, :)
    # @test @inferred(strides(V)) == (1, 5, 25, 125)
    #
    # # Issue #26263 — ensure that unaliascopy properly trims the array
    # A = rand(5,5,5,5)
    # V = view(A, 2:5, :, 2:5, 1:2:5)
    # @test @inferred(Base.unaliascopy(V)) == V == A[2:5, :, 2:5, 1:2:5]
    # @test @inferred(sum(Base.unaliascopy(V))) ≈ sum(V) ≈ sum(A[2:5, :, 2:5, 1:2:5])
    #
    # # issue #27632
    # function _test_27632(A)
    #     for J in CartesianIndices(size(A)[2:end])
    #         A[1, J]
    #     end
    #     nothing
    # end
    # # check that this doesn't crash
    # _test_27632(view(ones(Int64, (1, 1, 1)), 1, 1, 1))
    #
    # # issue #29608 - views of single values can be considered contiguous
    # @test Base.iscontiguous(view(ones(1), 1))

    @testset "array subviews" begin
        A = QuasiArray(reshape(1:60, 5,4,3), (range(0;stop=1,length=5), Base.OneTo(4), [2,3,6]))   
        @test A[0.0,:,3] isa Array
        @test view(A,0.0,:,3) isa SubArray
        @test A[0.0,:,3] == view(A,0.0,:,3)
    end

    @testset "sub-of-sub" begin
        A = QuasiArray(randn(3,3),(0:0.5:1,1:0.5:2))
        V = view(A,0:0.5:0.5,:)
        V2 = view(V,:,1:0.5:1.5)
        @test V2 isa SubArray
        @test parent(V2) isa QuasiArray
        @test V2 == A[0:0.5:0.5,1:0.5:1.5]
    end
end