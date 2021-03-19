using QuasiArrays, FillArrays, LinearAlgebra, SparseArrays, StaticArrays, Random, Base64, Test
import QuasiArrays: AbstractQuasiFill

@testset "fill array constructors and convert" begin
    for (Typ, funcs, func) in ((:QuasiZeros, :zeros, :zero), (:QuasiOnes, :ones, :one))
        @eval begin
            @test $Typ((Base.OneTo(5),)) isa AbstractQuasiVector{Float64}
            @test $Typ((Base.OneTo(5),Base.OneTo(5))) isa AbstractQuasiMatrix{Float64}

            for T in (Int, Float64)
                ax = [1,3,4]
                Z = $Typ{T}(ax)
                qZ = QuasiArray($funcs(T,length(ax)),(ax,))
                @test eltype(Z) == T
                @test Z[1] ≡ $func(T)
                @test Z[3] ≡ $func(T)
                @test Z[4] ≡ $func(T)
                @test_throws BoundsError Z[2]

                @test QuasiArray(Z) == qZ
                @test QuasiArray{T}(Z) == qZ
                @test QuasiArray{T,1}(Z) == qZ

                @test convert(AbstractQuasiArray,Z) ≡ Z
                @test convert(AbstractQuasiArray{T},Z) ≡ AbstractQuasiArray{T}(Z) ≡ Z
                @test convert(AbstractQuasiVector{T},Z) ≡ AbstractQuasiVector{T}(Z) ≡ Z

                Z = $Typ{T}(ax, ax)
                qZ = QuasiArray($funcs(T,length(ax),length(ax)),(ax,ax))
                @test eltype(Z) == T
                @test QuasiArray(Z) == qZ
                @test QuasiArray{T}(Z) == qZ
                @test QuasiArray{T,2}(Z) == qZ

                @test convert(AbstractQuasiArray,Z) ≡ convert(AbstractQuasiFill,Z) ≡ Z
                @test convert(AbstractQuasiArray{T},Z) ≡ convert(AbstractQuasiFill{T},Z) ≡ AbstractQuasiArray{T}(Z) ≡ Z
                @test convert(AbstractQuasiMatrix{T},Z) ≡ convert(AbstractQuasiFill{T,2},Z) ≡ AbstractQuasiMatrix{T}(Z) ≡ Z

                @test AbstractQuasiArray{Float32}(Z) ≡ $Typ{Float32}(ax,ax)
                @test AbstractQuasiArray{Float32,2}(Z) ≡ $Typ{Float32}(ax,ax)
            end
        end
    end

    @test QuasiFill(1) ≡ QuasiFill{Int}(1) ≡ QuasiFill{Int,0}(1) ≡ QuasiFill{Int,0,Tuple{}}(1,())
    ax = [1,3,4]
    @test QuasiFill(1.0,ax) isa AbstractQuasiVector{Float64}
    @test QuasiFill(1.0,ax,ax) isa AbstractQuasiMatrix{Float64}
    @test QuasiFill(1,ax) ≡ QuasiFill(1,(Inclusion(ax),))
    @test QuasiFill(1,ax,ax) ≡ QuasiFill(1,(Inclusion(ax),Inclusion(ax)))
    @test eltype(QuasiFill(1.0,ax,ax)) == Float64

    for T in (Int, Float64)
        F = QuasiFill{T}(one(T), ax)
        qF = QuasiArray(fill(one(T),length(ax)), (ax,))

        @test eltype(F) == T
        @test QuasiArray(F) == qF
        @test QuasiArray{T}(F) == qF
        @test QuasiArray{T,1}(F) == qF

        F = QuasiFill{T}(one(T), ax, ax)
        qF = QuasiArray(fill(one(T),length(ax),length(ax)), (ax,ax))
        @test eltype(F) == T
        @test QuasiArray(F) == qF
        @test QuasiArray{T}(F) == qF
        @test QuasiArray{T,2}(F) == qF

        @test convert(AbstractQuasiArray,F) ≡ F
        @test convert(AbstractQuasiArray{T},F) ≡ AbstractQuasiArray{T}(F) ≡ F
        @test convert(AbstractQuasiMatrix{T},F) ≡ AbstractQuasiMatrix{T}(F) ≡ F

        @test convert(AbstractQuasiArray{Float32},F) ≡ AbstractQuasiArray{Float32}(F) ≡
                QuasiFill{Float32}(one(Float32),ax,ax)
        @test convert(AbstractQuasiMatrix{Float32},F) ≡ AbstractQuasiMatrix{Float32}(F) ≡
                QuasiFill{Float32}(one(Float32),ax,ax)

        @test QuasiFill{T}(F) ≡ QuasiFill{T,2}(F) ≡ typeof(F)(F) ≡ F
    end

    @test QuasiEye(ax) isa QuasiDiagonal{Float64}
    @test QuasiEye(ax) == QuasiEye{Float64}(ax)
    @test eltype(QuasiEye(ax)) == Float64

    @test QuasiEye((Inclusion(ax),)) ≡ QuasiEye(ax)

    for T in (Int, Float64)
        E = QuasiEye{T}(ax)
        M = QuasiMatrix{T}(I, ax, ax)

        @test eltype(E) == T
        @test QuasiArray(E) == M
        @test QuasiArray{T}(E) == M
        @test QuasiArray{T,2}(E) == M

        @test convert(AbstractQuasiArray,E) === E
        @test convert(AbstractQuasiArray{T},E) === E
        @test convert(AbstractQuasiMatrix{T},E) === E


        @test AbstractQuasiArray{Float32}(E) == QuasiEye{Float32}(ax)
        @test QuasiEye{T}((Inclusion(ax),)) ≡ QuasiEye{T}(ax)
    end

    @testset "Bool should change type" begin
        ax = [1,3,4]
        x = QuasiFill(true,ax)
        y = x + x
        @test y isa QuasiFill{Int,1}
        @test y[1] == 2

        x = QuasiOnes{Bool}(ax)
        y = x + x
        @test y isa QuasiFill{Int,1}
        @test y[1] == 2
        @test x + QuasiZeros{Bool}(ax) ≡ x
        @test x - QuasiZeros{Bool}(ax) ≡ x
        @test QuasiZeros{Bool}(ax) + x ≡ x
        @test -x ≡ QuasiFill(-1,ax)
    end

    @testset "copy should return Fill" begin
        ax = [1,3,4]
        x = QuasiFill(1.0,ax)
        @test copy(x) ≡ x
        x = QuasiZeros(10)
        @test copy(x) ≡ x
        x = QuasiFill([1.,2.],ax)
        @test copy(x) == x
        @test copy(x) === x   # because isbits(x)
        @test copy(x) isa QuasiFill
    end

    @testset "in" begin
        ax = [1,3,4]
        for T in [QuasiZeros, QuasiOnes, QuasiFill]
            A = T(4, ax)
            @test FillArrays.getindex_value(A) in A
            @test !(FillArrays.getindex_value(A) + 1 in A)
        end
    end
end

@testset "indexing" begin
    ax = [1,3,4]
    A = QuasiFill(3.0,ax)
    @test A[[1,3]] ≡ Fill(3.0,2)
    @test A[Inclusion([1,3])] isa QuasiFill
    @test_throws BoundsError A[1:3]
    @test_throws BoundsError A[1:26]
    A = QuasiFill(3.0, ax, ax)
    @test A[[1,3], [3,4]] ≡ Fill(3.0,2,2)
    @test A[[1,3], Inclusion([3,4])] == QuasiFill(3.0,(Base.OneTo(2),Inclusion([3,4])))

    A = QuasiOnes{Int}(ax,ax)
    @test A[[1,3],1] ≡ Ones{Int}(2)
    @test A[[1,3],[3,4]] ≡ Ones{Int}(2,2)
    @test_throws BoundsError A[1:26]
    
    A = QuasiZeros{Int}(ax,ax)
    @test A[[1,3],1] ≡ Zeros{Int}(2)
    @test A[[1,3],[3,4]] ≡ Zeros{Int}(2,2)
    @test_throws BoundsError A[1:26]

    @testset "colon" begin
        @test QuasiOnes(ax)[:] == QuasiOnes(ax)
        @test QuasiZeros(ax)[:] == QuasiZeros(ax)
        @test QuasiFill(3.0,ax)[:] == QuasiFill(3.0,ax)

        @test QuasiOnes(ax,ax)[:,:] ≡ QuasiOnes(ax,ax)
    end

    @testset "mixed integer / vector /colon" begin
        ax = [1,3,4]
        a = QuasiFill(2.0,ax)
        z = QuasiZeros(ax)
        @test a[Inclusion(ax)] ≡ a[:] ≡ a
        @test z[Inclusion(ax)] ≡ z[:] ≡ z

        bx = 2:5
        A = QuasiFill(2.0,ax,bx)
        Z = QuasiZeros(ax,bx)
        @test A[:,2] ≡ A[Inclusion(ax),2] ≡ QuasiFill(2.0,ax)
        @test A[ax,2] ≡ Fill(2.0,3)
        @test A[1,:] ≡ A[1,Base.IdentityUnitRange(bx)] ≡ QuasiFill(2.0,bx)

        @test A[:,:] == A[Inclusion(ax),Base.IdentityUnitRange(bx)] == A[Inclusion(ax),:] == A[:,Base.IdentityUnitRange(bx)] == A
        @test Z[:,2] ≡ Z[Inclusion(ax),2] ≡ QuasiZeros(ax)
        @test Z[1,:] ≡ Z[1,Base.IdentityUnitRange(bx)] ≡ QuasiZeros(bx)
        @test Z[:,:] == Z[Inclusion(ax),Base.IdentityUnitRange(bx)] == Z[Inclusion(ax),:] == Z[:,Base.IdentityUnitRange(bx)] == Z

        cx = 1:2
        A = QuasiFill(2.0,ax,bx,cx)
        Z = QuasiZeros(ax,bx,cx)
        @test A[:,2,1] ≡ A[Inclusion(ax),2,1] ≡ QuasiFill(2.0,ax)
        @test A[1,:,1] ≡ A[1,Base.IdentityUnitRange(bx),1] ≡ QuasiFill(2.0,bx)
        @test A[:,:,:] ≡ A[Inclusion(ax),Base.IdentityUnitRange(bx),Base.IdentityUnitRange(cx)] ≡ A[Inclusion(ax),:,Base.IdentityUnitRange(cx)] ≡ A[:,Base.IdentityUnitRange(bx),Base.IdentityUnitRange(cx)] ≡ A
    end
end


# Check that all pair-wise combinations of + / - elements of As and Bs yield the correct
# type, and produce numerically correct results.
function test_addition_and_subtraction(As, Bs, Tout::Type)
    for A in As, B in Bs
        @test A + B isa Tout{promote_type(eltype(A), eltype(B))}
        @test QuasiArray(A + B) == QuasiArray(A) + QuasiArray(B)

        @test A - B isa Tout{promote_type(eltype(A), eltype(B))}
        @test QuasiArray(A - B) == QuasiArray(A) - QuasiArray(B)

        @test B + A isa Tout{promote_type(eltype(B), eltype(A))}
        @test QuasiArray(B + A) == QuasiArray(B) + QuasiArray(A)

        @test B - A isa Tout{promote_type(eltype(B), eltype(A))}
        @test QuasiArray(B - A) == QuasiArray(B) - QuasiArray(A)
    end
end

# Check that all permutations of + / - throw a `DimensionMismatch` exception.
function test_addition_and_subtraction_dim_mismatch(a, b)
    @test_throws DimensionMismatch a + b
    @test_throws DimensionMismatch a - b
    @test_throws DimensionMismatch b + a
    @test_throws DimensionMismatch b - a
end

@testset "FillArray addition and subtraction" begin
    ax = [1,3,4]
    bx = 2:5
    test_addition_and_subtraction_dim_mismatch(QuasiZeros(ax), QuasiZeros(bx))
    test_addition_and_subtraction_dim_mismatch(QuasiZeros(ax), QuasiZeros{Int}(bx))
    test_addition_and_subtraction_dim_mismatch(QuasiZeros(ax), QuasiZeros(bx,bx))
    test_addition_and_subtraction_dim_mismatch(QuasiZeros(ax), QuasiZeros{Int}(bx,ax))

    # Construct FillArray for repeated use.
    rng = MersenneTwister(123456)
    A_fill, B_fill = QuasiFill(randn(rng, Float64), ax), QuasiFill(4, ax)

    # Unary +/- constructs a new FillArray.
    @test +A_fill === A_fill
    @test -A_fill === QuasiFill(-A_fill.value, ax)

    # FillArray +/- FillArray should construct a new FillArray.
    test_addition_and_subtraction([A_fill, B_fill], [A_fill, B_fill], QuasiFill)
    test_addition_and_subtraction_dim_mismatch(A_fill, QuasiFill(randn(rng), 5, 2))

    # FillArray + Array (etc) should construct a new Array using `getindex`.
    A_dense, B_dense = QuasiArray(randn(rng, 3),ax), QuasiArray([5, 4, 3],ax)
    test_addition_and_subtraction([A_fill, B_fill], [A_dense, B_dense], QuasiArray)
    test_addition_and_subtraction_dim_mismatch(A_fill, QuasiArray(randn(rng, 3, 4),ax,bx))
end

@testset "Other matrix types" begin
    ax = [1,3,4]
    bx = 2:5
    z = QuasiZeros(ax)
    @test QuasiDiagonal(z) == QuasiDiagonal(QuasiArray(z))

    @test QuasiDiagonal(QuasiZeros(ax,bx)) == QuasiDiagonal(z)
    @test convert(QuasiDiagonal, QuasiZeros(ax,ax)) == QuasiDiagonal(z)
    @test_throws BoundsError convert(QuasiDiagonal, QuasiZeros(ax,bx))

    @test convert(QuasiDiagonal{Int}, QuasiZeros(ax,ax)) == QuasiDiagonal(z)
    @test_throws BoundsError convert(QuasiDiagonal{Int}, QuasiZeros(bx,ax))


    @test QuasiDiagonal(QuasiEye(ax)) == QuasiEye(ax)
    @test convert(QuasiDiagonal, QuasiEye(ax)) ==  QuasiEye(ax)
    @test convert(QuasiDiagonal{Int}, QuasiEye(ax)) == QuasiEye(ax)
end


@testset "==" begin
    ax,bx = [1,3,4],2:5
    @test QuasiZeros(ax,bx) == QuasiFill(0,ax,bx)
    @test QuasiZeros(ax,bx) ≠ QuasiZeros(3)
    @test QuasiOnes(ax,bx) == QuasiFill(1,ax,bx)
end

@testset "Rank" begin
    ax,bx = [1,3,4],2:5
    @test rank(QuasiZeros(ax,bx)) == 0
    @test rank(QuasiOnes(ax,bx)) == 1
    @test rank(QuasiFill(2,ax,bx)) == 1
    @test rank(QuasiFill(0,ax,bx)) == 0
end


@testset "Identities" begin
    ax,bx,cx = [1,3,4],2:5,1:5
    A = QuasiArray(randn(4,5),bx,cx)
    B = QuasiArray(randn(3,4),ax,bx)
    @test 1.0 .* QuasiZeros(ax,bx) ≡ QuasiZeros(ax,bx) .* 1.0 ≡ QuasiZeros(ax,bx)
    @test QuasiZeros(ax,bx) * QuasiZeros(bx,cx) == QuasiZeros(ax,bx) * A == B * QuasiZeros(bx,cx) == QuasiZeros(ax,cx)
    @test QuasiZeros(ax,bx) * A isa QuasiZeros
    @test B * QuasiZeros(bx,cx) isa QuasiZeros
    @test_throws DimensionMismatch B * QuasiZeros(ax, ax)
    
    # Check multiplication by Adjoint vectors works as expected.
    @test QuasiArray(randn(3, 4),ax,bx)' * QuasiZeros(ax) === QuasiZeros(bx)
    @test QuasiArray(randn(4),bx)' * QuasiZeros(bx) === zero(Float64)
    @test QuasiArray([1, 2, 3],ax)' * QuasiZeros{Int}(ax) === zero(Int)
    @test_broken QuasiArray([SVector(1,2)', SVector(2,3)', SVector(3,4)'],ax)' * QuasiZeros{Int}(ax) === SVector(0,0)


    @test +(QuasiZeros{Float64}(ax, bx)) === QuasiZeros{Float64}(ax, bx)
    @test -(QuasiZeros{Float32}(bx, cx)) === QuasiZeros{Float32}(bx, cx)

    # `Zeros` are closed under addition and subtraction (both unary and binary).
    z1, z2 = QuasiZeros{Float64}(ax), QuasiZeros{Int}(ax)
    @test +(z1) === z1
    @test -(z1) === z1

    test_addition_and_subtraction([z1, z2], [z1, z2], QuasiZeros)
    test_addition_and_subtraction_dim_mismatch(z1, QuasiZeros{Float64}(ax, bx))

    # `Zeros` +/- `Fill`s should yield `Fills`.
    fill1, fill2 = QuasiFill(5.0, ax), QuasiFill(5, ax)
    test_addition_and_subtraction([z1, z2], [fill1, fill2], QuasiFill)
    test_addition_and_subtraction_dim_mismatch(z1, QuasiFill(5, bx))

    X = QuasiArray(randn(3, 4), ax, bx)
    for op in [+, -]

        # Addition / subtraction with same eltypes.
        @test op(QuasiZeros(ax, bx), QuasiZeros(ax, bx)) === QuasiZeros(ax, bx)
        @test_throws DimensionMismatch op(X, QuasiZeros(ax, cx))
        @test eltype(op(QuasiZeros(ax, bx), X)) == Float64

        # Different eltypes, the other way around.
        @test op(X, QuasiZeros{Float32}(ax,bx)) isa QuasiMatrix{Float64}
        @test !(op(X, QuasiZeros{Float32}(ax,bx)) === X)
        @test op(X, QuasiZeros{Float32}(ax,bx)) == X
        @test !(op(X, QuasiZeros{ComplexF64}(ax,bx)) === X)
        @test op(X, QuasiZeros{ComplexF64}(ax,bx)) == X

        # Addition / subtraction of Zeros.
        @test eltype(op(QuasiZeros{Float64}(ax,bx), QuasiZeros{Int}(ax,bx))) == Float64
        @test eltype(op(QuasiZeros{Int}(bx,ax), QuasiZeros{Float32}(bx,ax))) == Float32
        @test op(QuasiZeros{Float64}(ax,bx), QuasiZeros{Int}(ax,bx)) isa QuasiZeros{Float64}
        @test op(QuasiZeros{Float64}(ax,bx), QuasiZeros{Int}(ax,bx)) === QuasiZeros{Float64}(ax,bx)
    end

    # Zeros +/- dense where + / - have different results.
    @test +(QuasiZeros(ax,bx), X) == X && +(X, QuasiZeros(ax,bx)) == X
    @test !(QuasiZeros(ax,bx) + X === X) && !(X + QuasiZeros(ax,bx) === X)
    @test -(QuasiZeros(ax,bx), X) == -X

    # Addition with different eltypes.
    @test +(QuasiZeros{Float32}(ax,bx), X) isa QuasiMatrix{Float64}
    @test !(+(QuasiZeros{Float32}(ax,bx), X) === X)
    @test +(QuasiZeros{Float32}(ax,bx), X) == X
    @test !(+(QuasiZeros{ComplexF64}(ax,bx), X) === X)
    @test +(QuasiZeros{ComplexF64}(ax,bx), X) == X

    # Subtraction with different eltypes.
    @test -(QuasiZeros{Float32}(ax,bx), X) isa QuasiMatrix{Float64}
    @test -(QuasiZeros{Float32}(ax,bx), X) == -X
    @test -(QuasiZeros{ComplexF64}(ax,bx), X) == -X

    # Tests for ranges.
    X = QuasiArray(randn(3),ax)
    @test !(QuasiZeros(ax) + X === X)

    # test Base.zero
    @test zero(QuasiZeros(ax)) == QuasiZeros(ax)
    @test zero(QuasiOnes(ax,ax)) == QuasiZeros(ax,ax)
    @test zero(QuasiFill(0.5, ax, ax)) == QuasiZeros(ax,ax)
end

@testset "maximum/minimum/svd/sort" begin
    ax = [1,3,4]
    @test maximum(QuasiFill(1, ax)) == minimum(QuasiFill(1, ax)) == 1
    @test sort(QuasiOnes(ax)) == sort!(QuasiOnes(ax))
end

@testset "Cumsum and diff" begin
    ax = [1,3,4]
    @test sum(QuasiFill(3,ax)) ≡ 9
    @test sum(x -> x + 1, QuasiFill(3,ax)) ≡ 12
    @test cumsum(QuasiFill(3,ax)) == 3:3:9

    @test sum(QuasiOnes(ax)) ≡ 3.0
    @test sum(x -> x + 1, QuasiOnes(ax)) ≡ 6.0
    @test cumsum(QuasiOnes(ax)) == 1:3

    @test sum(QuasiOnes{Int}(ax)) ≡ 3
    @test sum(x -> x + 1, QuasiOnes{Int}(ax)) ≡ 6
    @test cumsum(QuasiOnes{Int}(ax)) == Base.OneTo(3)

    @test sum(Zeros(10)) ≡ 0.0
    @test sum(x -> x + 1, Zeros(10)) ≡ 10.0
    @test cumsum(Zeros(10)) ≡ Zeros(10)

    @test sum(Zeros{Int}(10)) ≡ 0
    @test sum(x -> x + 1, Zeros{Int}(10)) ≡ 10
    @test cumsum(Zeros{Int}(10)) ≡ Zeros{Int}(10)

    @test cumsum(Zeros{Bool}(10)) ≡ Zeros{Bool}(10)
    @test cumsum(Ones{Bool}(10)) ≡ Base.OneTo{Int}(10)
    @test cumsum(Fill(true,10)) ≡ 1:1:10

    @test diff(Fill(1,10)) ≡ Zeros{Int}(9)
    @test diff(Ones{Float64}(10)) ≡ Zeros{Float64}(9)
end

@testset "Broadcast" begin
    x = Fill(5,5)
    @test (.+)(x) ≡ x
    @test (.-)(x) ≡ -x
    @test exp.(x) ≡ Fill(exp(5),5)
    @test x .+ 1 ≡ Fill(6,5)
    @test 1 .+ x ≡ Fill(6,5)
    @test x .+ x ≡ Fill(10,5)
    @test x .+ Ones(5) ≡ Fill(6.0,5)
    f = (x,y) -> cos(x*y)
    @test f.(x, Ones(5)) ≡ Fill(f(5,1.0),5)

    y = Ones(5,5)
    @test (.+)(y) ≡ Ones(5,5)
    @test (.-)(y) ≡ Fill(-1.0,5,5)
    @test exp.(y) ≡ Fill(exp(1),5,5)
    @test y .+ 1 ≡ Fill(2.0,5,5)
    @test y .+ y ≡ Fill(2.0,5,5)
    @test y .* y ≡ y ./ y ≡ y .\ y ≡ y

    rng = MersenneTwister(123456)
    sizes = [(5, 4), (5, 1), (1, 4), (1, 1), (5,)]
    for sx in sizes, sy in sizes
        x, y = Fill(randn(rng), sx), Fill(randn(rng), sy)
        x_one, y_one = Ones(sx), Ones(sy)
        x_zero, y_zero = Zeros(sx), Zeros(sy)
        x_dense, y_dense = randn(rng, sx), randn(rng, sy)

        for x in [x, x_one, x_zero, x_dense], y in [y, y_one, y_zero, y_dense]
            @test x .+ y == collect(x) .+ collect(y)
        end
        @test x_zero .+ y_zero isa Zeros
        @test x_zero .+ y_one isa Ones
        @test x_one .+ y_zero isa Ones

        for x in [x, x_one, x_zero, x_dense], y in [y, y_one, y_zero, y_dense]
            @test x .* y == collect(x) .* collect(y)
        end
        for x in [x, x_one, x_zero, x_dense]
            @test x .* y_zero isa Zeros
        end
        for y in [y, y_one, y_zero, y_dense]
            @test x_zero .* y isa Zeros
        end
    end

    @test Zeros{Int}(5) .+ Zeros(5) isa Zeros{Float64}

    # Test for conj, real and imag with complex element types
    @test conj(Zeros{ComplexF64}(10)) isa Zeros{ComplexF64}
    @test conj(Zeros{ComplexF64}(10,10)) isa Zeros{ComplexF64}
    @test conj(Ones{ComplexF64}(10)) isa Ones{ComplexF64}
    @test conj(Ones{ComplexF64}(10,10)) isa Ones{ComplexF64}
    @test real(Zeros{Float64}(10)) isa Zeros{Float64}
    @test real(Zeros{Float64}(10,10)) isa Zeros{Float64}
    @test real(Zeros{ComplexF64}(10)) isa Zeros{Float64}
    @test real(Zeros{ComplexF64}(10,10)) isa Zeros{Float64}
    @test real(Ones{Float64}(10)) isa Ones{Float64}
    @test real(Ones{Float64}(10,10)) isa Ones{Float64}
    @test real(Ones{ComplexF64}(10)) isa Ones{Float64}
    @test real(Ones{ComplexF64}(10,10)) isa Ones{Float64}
    @test imag(Zeros{Float64}(10)) isa Zeros{Float64}
    @test imag(Zeros{Float64}(10,10)) isa Zeros{Float64}
    @test imag(Zeros{ComplexF64}(10)) isa Zeros{Float64}
    @test imag(Zeros{ComplexF64}(10,10)) isa Zeros{Float64}
    @test imag(Ones{Float64}(10)) isa Zeros{Float64}
    @test imag(Ones{Float64}(10,10)) isa Zeros{Float64}
    @test imag(Ones{ComplexF64}(10)) isa Zeros{Float64}
    @test imag(Ones{ComplexF64}(10,10)) isa Zeros{Float64}

    @testset "range broadcast" begin
        rnge = range(-5.0, step=1.0, length=10)
        @test broadcast(*, Fill(5.0, 10), rnge) == broadcast(*, 5.0, rnge)
        @test broadcast(*, Zeros(10, 10), rnge) ≡ Zeros{Float64}(10, 10)
        @test broadcast(*, rnge, Zeros(10, 10)) ≡ Zeros{Float64}(10, 10)
        @test broadcast(*, Ones{Int}(10), rnge) ≡ rnge
        @test broadcast(*, rnge, Ones{Int}(10)) ≡ rnge
        @test_throws DimensionMismatch broadcast(*, Fill(5.0, 11), rnge)
        @test broadcast(*, rnge, Fill(5.0, 10)) == broadcast(*, rnge, 5.0)
        @test_throws DimensionMismatch broadcast(*, rnge, Fill(5.0, 11))

        # following should pass using alternative implementation in code
        deg = 5:5
        @test_throws ArgumentError @inferred(broadcast(*, Fill(5.0, 10), deg)) == broadcast(*, fill(5.0,10), deg)
        @test_throws ArgumentError @inferred(broadcast(*, deg, Fill(5.0, 10))) == broadcast(*, deg, fill(5.0,10))

        @test rnge .+ Zeros(10) ≡ rnge .- Zeros(10) ≡ Zeros(10) .+ rnge ≡ rnge

        @test_throws DimensionMismatch rnge .+ Zeros(5)
        @test_throws DimensionMismatch rnge .- Zeros(5)
        @test_throws DimensionMismatch Zeros(5) .+ rnge
    end

    @testset "Special Zeros/Ones" begin
        @test broadcast(+,Zeros(5)) ≡ broadcast(-,Zeros(5)) ≡ Zeros(5)
        @test broadcast(+,Ones(5)) ≡ Ones(5)

        @test Zeros(5) .* Ones(5) ≡ Zeros(5) .* 1 ≡ Zeros(5)
        @test Zeros(5) .* Fill(5.0, 5) ≡ Zeros(5) .* 5.0 ≡ Zeros(5)
        @test Ones(5) .* Zeros(5) ≡ 1 .* Zeros(5) ≡ Zeros(5)
        @test Fill(5.0, 5) .* Zeros(5) ≡ 5.0 .* Zeros(5) ≡ Zeros(5)

        @test Zeros(5) ./ Ones(5) ≡ Zeros(5) ./ 1 ≡ Zeros(5)
        @test Zeros(5) ./ Fill(5.0, 5) ≡ Zeros(5) ./ 5.0 ≡ Zeros(5)
        @test Ones(5) .\ Zeros(5) ≡ 1 .\ Zeros(5) ≡ Zeros(5)
        @test Fill(5.0, 5) .\ Zeros(5) ≡ 5.0 .\ Zeros(5) ≡ Zeros(5)

        @test conj.(Zeros(5)) ≡ Zeros(5)
        @test conj.(Zeros{ComplexF64}(5)) ≡ Zeros{ComplexF64}(5)

        @test_throws DimensionMismatch broadcast(*, Ones(3), 1:6)
        @test_throws DimensionMismatch broadcast(*, 1:6, Ones(3))
        @test_throws DimensionMismatch broadcast(*, Fill(1,3), 1:6)
        @test_throws DimensionMismatch broadcast(*, 1:6, Fill(1,3))

        @testset "Number" begin
            @test broadcast(*, Zeros(5), 2) ≡ broadcast(*, 2, Zeros(5)) ≡ Zeros(5)
        end

        @testset "Nested" begin
            @test randn(5) .\ rand(5) .* Zeros(5) ≡ Zeros(5)
            @test broadcast(*, Zeros(5), Base.Broadcast.broadcasted(\, randn(5), rand(5))) ≡ Zeros(5)
        end

        @testset "array-valued" begin
            @test broadcast(*, Fill([1,2],3), 1:3) == broadcast(*, 1:3, Fill([1,2],3)) == broadcast(*, 1:3, fill([1,2],3))
            @test broadcast(*, Fill([1,2],3), Zeros(3)) == broadcast(*, Zeros(3), Fill([1,2],3)) == broadcast(*, zeros(3), fill([1,2],3))
            @test broadcast(*, Fill([1,2],3), Zeros(3)) isa Fill{Vector{Float64}}
            @test broadcast(*, [[1,2], [3,4,5]], Zeros(2)) == broadcast(*, Zeros(2), [[1,2], [3,4,5]]) == broadcast(*, zeros(2), [[1,2], [3,4,5]])
        end

        @testset "NaN" begin
            @test Zeros(5) ./ Zeros(5) ≡ Zeros(5) .\ Zeros(5) ≡ Fill(NaN,5)
            @test Zeros{Int}(5,6) ./ Zeros{Int}(5) ≡ Zeros{Int}(5) .\ Zeros{Int}(5,6) ≡ Fill(NaN,5,6)
        end

        @testset "Addition" begin
            @test Zeros{Int}(5) .+ (1:5) ≡ (1:5) .+ Zeros{Int}(5) ≡ (1:5) .- Zeros{Int}(5) ≡ 1:5
            @test Zeros{Int}(1) .+ (1:5) ≡ (1:5) .+ Zeros{Int}(1) ≡ (1:5) .- Zeros{Int}(1) ≡ 1:5
            @test Zeros(5) .+ (1:5) == (1:5) .+ Zeros(5) == (1:5) .- Zeros(5) == 1:5
            @test Zeros{Int}(5) .+ Fill(1,5) ≡ Fill(1,5) .+ Zeros{Int}(5) ≡ Fill(1,5) .- Zeros{Int}(5) ≡ Fill(1,5)
            @test_throws DimensionMismatch Zeros{Int}(2) .+ (1:5)
            @test_throws DimensionMismatch (1:5) .+ Zeros{Int}(2)
        end
    end

    @testset "support Ref" begin
        @test Fill(1,10) .- 1 ≡ Fill(1,10) .- Ref(1) ≡ Fill(1,10) .- Ref(1I)
        @test Fill([1 2; 3 4],10) .- Ref(1I) == Fill([0 2; 3 3],10)
        @test Ref(1I) .+ Fill([1 2; 3 4],10) == Fill([2 2; 3 5],10)
    end

    @testset "Special Ones" begin
        @test Ones{Int}(5) .* (1:5) ≡ (1:5) .* Ones{Int}(5) ≡ 1:5
        @test Ones(5) .* (1:5) ≡ (1:5) .* Ones(5) ≡ 1.0:5
        @test Ones{Int}(5) .* Ones{Int}(5) ≡ Ones{Int}(5)
        @test Ones{Int}(5,2) .* (1:5) == Array(Ones{Int}(5,2)) .* Array(1:5)
        @test (1:5) .* Ones{Int}(5,2)  == Array(1:5) .* Array(Ones{Int}(5,2))
        @test (1:0.5:5) .* Ones{Int}(9,2)  == Array(1:0.5:5) .* Array(Ones{Int}(9,2))
        @test Ones{Int}(9,2) .* (1:0.5:5)  == Array(Ones{Int}(9,2)) .* Array(1:0.5:5)
        @test_throws DimensionMismatch Ones{Int}(6) .* (1:5)
        @test_throws DimensionMismatch (1:5) .* Ones{Int}(6)
        @test_throws DimensionMismatch Ones{Int}(5) .* Ones{Int}(6)
    end

    @testset "Zeros -" begin
        @test Zeros(10) - Zeros(10) ≡ Zeros(10)
        @test Ones(10) - Zeros(10) ≡ Ones(10)
        @test Ones(10) - Ones(10) ≡ Zeros(10)
        @test Fill(1,10) - Zeros(10) ≡ Fill(1.0,10)

        @test Zeros(10) .- Zeros(10) ≡ Zeros(10)
        @test Ones(10) .- Zeros(10) ≡ Ones(10)
        @test Ones(10) .- Ones(10) ≡ Zeros(10)
        @test Fill(1,10) .- Zeros(10) ≡ Fill(1.0,10)

        @test Zeros(10) .- Zeros(1,9) ≡ Zeros(10,9)
        @test Ones(10) .- Zeros(1,9) ≡ Ones(10,9)
        @test Ones(10) .- Ones(1,9) ≡ Zeros(10,9)
    end

    @testset "Zero .*" begin
        @test Zeros{Int}(10) .* Zeros{Int}(10) ≡ Zeros{Int}(10)
        @test randn(10) .* Zeros(10) ≡ Zeros(10)
        @test Zeros(10) .* randn(10) ≡ Zeros(10)
        @test (1:10) .* Zeros(10) ≡ Zeros(10)
        @test Zeros(10) .* (1:10) ≡ Zeros(10)
        @test_throws DimensionMismatch (1:11) .* Zeros(10)
    end
end

@testset "map" begin
    x = Ones(5)
    @test map(exp,x) === Fill(exp(1.0),5)
    @test map(isone,x) === Fill(true,5)

    x = Zeros(5)
    @test map(exp,x) === exp.(x)

    x = Fill(2,5,3)
    @test map(exp,x) === Fill(exp(2),5,3)
end

@testset "Offset indexing" begin
    A = Fill(3, (Base.Slice(-1:1),))
    @test axes(A)  == (Base.Slice(-1:1),)
    @test A[0] == 3
    @test_throws BoundsError A[2]
    @test_throws BoundsError A[-2]

    A = Zeros((Base.Slice(-1:1),))
    @test axes(A)  == (Base.Slice(-1:1),)
    @test A[0] == 0
    @test_throws BoundsError A[2]
    @test_throws BoundsError A[-2]
end

@testset "0-dimensional" begin
    A = Fill{Int,0,Tuple{}}(3, ())

    @test A[] ≡ A[1] ≡ 3
    @test A ≡ Fill{Int,0}(3, ()) ≡ Fill(3, ()) ≡ Fill(3)
    @test size(A) == ()
    @test axes(A) == ()

    A = Ones{Int,0,Tuple{}}(())
    @test A[] ≡ A[1] ≡ 1
    @test A ≡ Ones{Int,0}(()) ≡ Ones{Int}(()) ≡ Ones{Int}()

    A = Zeros{Int,0,Tuple{}}(())
    @test A[] ≡ A[1] ≡ 0
    @test A ≡ Zeros{Int,0}(()) ≡ Zeros{Int}(()) ≡ Zeros{Int}()
end

@testset "unique" begin
    @test unique(Fill(12, 20)) == unique(fill(12, 20))
    @test unique(Fill(1, 0)) == []
    @test unique(Zeros(0)) isa Vector{Float64}
    @test !allunique(Fill("a", 2))
    @test allunique(Ones(0))
end

@testset "iterate" begin
    for d in (0, 1, 2, 100)
        for T in (Float64, Int)
            m = Eye(d)
            mcp = [x for x in m]
            @test mcp == m
            @test eltype(mcp) == eltype(m)
        end
    end
end

@testset "properties" begin
    for d in (0, 1, 2, 100)
        @test isone(Eye(d))
    end
end

@testset "any all iszero isone" begin
    for T in (Int, Float64, ComplexF64)
        for m in (Eye{T}(0), Eye{T}(0, 0), Eye{T}(0, 1), Eye{T}(1, 0))
            @test ! any(isone, m)
            @test ! any(iszero, m)
            @test ! all(iszero, m)
            @test ! all(isone, m)
        end
        for d in (1, )
            for m in (Eye{T}(d), Eye{T}(d, d))
                @test ! any(iszero, m)
                @test ! all(iszero, m)
                @test any(isone, m)
                @test all(isone, m)
            end

            for m in (Eye{T}(d, d + 1), Eye{T}(d + 1, d))
                @test any(iszero, m)
                @test ! all(iszero, m)
                @test any(isone, m)
                @test ! all(isone, m)
            end

            onem = Ones{T}(d, d)
            @test isone(onem)
            @test ! iszero(onem)

            zerom = Zeros{T}(d, d)
            @test ! isone(zerom)
            @test  iszero(zerom)

            fillm0 = Fill(T(0), d, d)
            @test ! isone(fillm0)
            @test   iszero(fillm0)

            fillm1 = Fill(T(1), d, d)
            @test isone(fillm1)
            @test ! iszero(fillm1)

            fillm2 = Fill(T(2), d, d)
            @test ! isone(fillm2)
            @test ! iszero(fillm2)
        end
        for d in (2, 3)
            for m in (Eye{T}(d), Eye{T}(d, d), Eye{T}(d, d + 2), Eye{T}(d + 2, d))
                @test any(iszero, m)
                @test ! all(iszero, m)
                @test any(isone, m)
                @test ! all(isone, m)
            end

            m1 = Ones{T}(d, d)
            @test ! isone(m1)
            @test ! iszero(m1)
            @test all(isone, m1)
            @test ! all(iszero, m1)

            m2 = Zeros{T}(d, d)
            @test ! isone(m2)
            @test iszero(m2)
            @test ! all(isone, m2)
            @test  all(iszero, m2)

            m3 = Fill(T(2), d, d)
            @test ! isone(m3)
            @test ! iszero(m3)
            @test ! all(isone, m3)
            @test ! all(iszero, m3)
            @test ! any(iszero, m3)

            m4 = Fill(T(1), d, d)
            @test ! isone(m4)
            @test ! iszero(m4)
        end
    end

    @testset "all/any" begin
        @test any(Ones{Bool}(10)) === all(Ones{Bool}(10)) === any(Fill(true,10)) === all(Fill(true,10)) === true
        @test any(Zeros{Bool}(10)) === all(Zeros{Bool}(10)) === any(Fill(false,10)) === all(Fill(false,10)) === false
        @test all(b -> ndims(b) ==  1, Fill([1,2],10))
        @test any(b -> ndims(b) ==  1, Fill([1,2],10))
    end

    @testset "Error" begin
        @test_throws TypeError any(exp, Fill(1,5))
        @test_throws TypeError all(exp, Fill(1,5))
        @test_throws TypeError any(exp, Eye(5))
        @test_throws TypeError all(exp, Eye(5))
        @test_throws TypeError any(Fill(1,5))
        @test_throws TypeError all(Fill(1,5))
        @test_throws TypeError any(Zeros(5))
        @test_throws TypeError all(Zeros(5))
        @test_throws TypeError any(Ones(5))
        @test_throws TypeError all(Ones(5))
        @test_throws TypeError any(Eye(5))
        @test_throws TypeError all(Eye(5))
    end
end

@testset "Eye identity ops" begin
    m = Eye(10)
    D = Diagonal(Fill(2,10))

    for op in (permutedims, inv)
        @test op(m) === m
    end
    @test permutedims(D) ≡ D
    @test inv(D) ≡ Diagonal(Fill(1/2,10))

    for m in (Eye(10), Eye(10, 10), Eye(10, 8), Eye(8, 10), D)
        for op in (tril, triu, tril!, triu!)
            @test op(m) === m
        end
    end

    @test copy(m) ≡ m
    @test copy(D) ≡ D
    @test LinearAlgebra.copy_oftype(m, Int) ≡ Eye{Int}(10)
    @test LinearAlgebra.copy_oftype(D, Float64) ≡ Diagonal(Fill(2.0,10))
end

@testset "Issue #31" begin
    @test convert(SparseMatrixCSC{Float64,Int64}, Zeros{Float64}(3, 3)) == spzeros(3, 3)
    @test sparse(Zeros(4, 2)) == spzeros(4, 2)
end

@testset "Adjoint/Transpose/permutedims" begin
    @test Ones{ComplexF64}(5,6)' ≡ transpose(Ones{ComplexF64}(5,6)) ≡ Ones{ComplexF64}(6,5)
    @test Zeros{ComplexF64}(5,6)' ≡ transpose(Zeros{ComplexF64}(5,6)) ≡ Zeros{ComplexF64}(6,5)
    @test Fill(1+im, 5, 6)' ≡ Fill(1-im, 6,5)
    @test transpose(Fill(1+im, 5, 6)) ≡ Fill(1+im, 6,5)
    @test Ones(5)' isa Adjoint # Vectors still need special dot product
    @test Fill([1+im 2; 3 4; 5 6], 2,3)' == Fill([1+im 2; 3 4; 5 6]', 3,2)
    @test transpose(Fill([1+im 2; 3 4; 5 6], 2,3)) == Fill(transpose([1+im 2; 3 4; 5 6]), 3,2)

    @test permutedims(Ones(10)) ≡ Ones(1,10)
    @test permutedims(Zeros(10)) ≡ Zeros(1,10)
    @test permutedims(Fill(2.0,10)) ≡ Fill(2.0,1,10)
    @test permutedims(Ones(10,3)) ≡ Ones(3,10)
    @test permutedims(Zeros(10,3)) ≡ Zeros(3,10)
    @test permutedims(Fill(2.0,10,3)) ≡ Fill(2.0,3,10)

    @test permutedims(Ones(2,4,5), [3,2,1]) == permutedims(Array(Ones(2,4,5)), [3,2,1])
    @test permutedims(Ones(2,4,5), [3,2,1]) ≡ Ones(5,4,2)
    @test permutedims(Zeros(2,4,5), [3,2,1]) ≡ Zeros(5,4,2)
    @test permutedims(Fill(2.0,2,4,5), [3,2,1]) ≡ Fill(2.0,5,4,2)
end

@testset "setindex!/fill!" begin
    F = Fill(1,10)
    @test (F[1] = 1) == 1
    @test_throws BoundsError (F[11] = 1)
    @test_throws ArgumentError (F[10] = 2)


    F = Fill(1,10,5)
    @test (F[1] = 1) == 1
    @test (F[3,3] = 1) == 1
    @test_throws BoundsError (F[51] = 1)
    @test_throws BoundsError (F[1,6] = 1)
    @test_throws ArgumentError (F[10] = 2)
    @test_throws ArgumentError (F[10,1] = 2)

    @test (F[:,1] .= 1) == fill(1,10)
    @test_throws ArgumentError (F[:,1] .= 2)

    @test fill!(F,1) == F
    @test_throws ArgumentError fill!(F,2)
end

@testset "mult" begin
    @test Fill(2,10)*Fill(3,1,12) == Vector(Fill(2,10))*Matrix(Fill(3,1,12))
    @test Fill(2,10)*Fill(3,1,12) ≡ Fill(6,10,12)
    @test Fill(2,3,10)*Fill(3,10,12) ≡ Fill(60,3,12)
    @test Fill(2,3,10)*Fill(3,10) ≡ Fill(60,3)
    @test_throws DimensionMismatch Fill(2,10)*Fill(3,2,12)
    @test_throws DimensionMismatch Fill(2,3,10)*Fill(3,2,12)

    @test Ones(10)*Fill(3,1,12) ≡ Fill(3.0,10,12)
    @test Ones(10,3)*Fill(3,3,12) ≡ Fill(9.0,10,12)
    @test Ones(10,3)*Fill(3,3) ≡ Fill(9.0,10)

    @test Fill(2,10)*Ones(1,12) ≡ Fill(2.0,10,12)
    @test Fill(2,3,10)*Ones(10,12) ≡ Fill(20.0,3,12)
    @test Fill(2,3,10)*Ones(10) ≡ Fill(20.0,3)

    @test Ones(10)*Ones(1,12) ≡ Ones(10,12)
    @test Ones(3,10)*Ones(10,12) ≡ Fill(10.0,3,12)
    @test Ones(3,10)*Ones(10) ≡ Fill(10.0,3)

    @test Zeros(10)*Fill(3,1,12) ≡   Zeros(10,12)
    @test Zeros(10,3)*Fill(3,3,12) ≡ Zeros(10,12)
    @test Zeros(10,3)*Fill(3,3) ≡    Zeros(10)

    @test Fill(2,10)*  Zeros(1,12) ≡  Zeros(10,12)
    @test Fill(2,3,10)*Zeros(10,12) ≡ Zeros(3,12)
    @test Fill(2,3,10)*Zeros(10) ≡    Zeros(3)

    @test Zeros(10)*Zeros(1,12) ≡ Zeros(10,12)
    @test Zeros(3,10)*Zeros(10,12) ≡ Zeros(3,12)
    @test Zeros(3,10)*Zeros(10) ≡ Zeros(3)

    a = randn(3)
    A = randn(1,4)

    @test Fill(2,3)*A ≈ Vector(Fill(2,3))*A
    @test Fill(2,3,1)*A ≈ Matrix(Fill(2,3,1))*A
    @test Fill(2,3,3)*a ≈ Matrix(Fill(2,3,3))*a
    @test Ones(3)*A ≈ Vector(Ones(3))*A
    @test Ones(3,1)*A ≈ Matrix(Ones(3,1))*A
    @test Ones(3,3)*a ≈ Matrix(Ones(3,3))*a
    @test Zeros(3)*A  ≡ Zeros(3,4)
    @test Zeros(3,1)*A == Zeros(3,4)
    @test Zeros(3,3)*a == Zeros(3)

    @test A*Fill(2,4) ≈ A*Vector(Fill(2,4))
    @test A*Fill(2,4,1) ≈ A*Matrix(Fill(2,4,1))
    @test a*Fill(2,1,3) ≈ a*Matrix(Fill(2,1,3))
    @test A*Ones(4) ≈ A*Vector(Ones(4))
    @test A*Ones(4,1) ≈ A*Matrix(Ones(4,1))
    @test a*Ones(1,3) ≈ a*Matrix(Ones(1,3))
    @test A*Zeros(4)  ≡ Zeros(1)
    @test A*Zeros(4,1) ≡ Zeros(1,1)
    @test a*Zeros(1,3) ≡ Zeros(3,3)

    D = Diagonal(randn(1))
    @test Zeros(1,1)*D ≡ Zeros(1,1)
    @test Zeros(1)*D ≡ Zeros(1,1)
    @test D*Zeros(1,1) ≡ Zeros(1,1)
    @test D*Zeros(1) ≡ Zeros(1)

    D = Diagonal(Fill(2,10))
    @test D * Ones(10) ≡ Fill(2.0,10)
    @test D * Ones(10,5) ≡ Fill(2.0,10,5)
    @test Ones(5,10) * D ≡ Fill(2.0,5,10)

    # following test is broken in Base as of Julia v1.5
    @test_skip @test_throws DimensionMismatch Diagonal(Fill(1,1)) * Ones(10)
    @test_throws DimensionMismatch Diagonal(Fill(1,1)) * Ones(10,5)
    @test_throws DimensionMismatch Ones(5,10) * Diagonal(Fill(1,1))

    E = Eye(5)
    @test E*(1:5) ≡ 1.0:5.0
    @test (1:5)'E == (1.0:5)'
    @test E*E ≡ E
end

@testset "count" begin
    @test count(Ones{Bool}(10)) == count(Fill(true,10)) == 10
    @test count(Zeros{Bool}(10)) == count(Fill(false,10)) == 0
    @test count(x -> 1 ≤ x < 2, Fill(1.3,10)) == 10
    @test count(x -> 1 ≤ x < 2, Fill(2.0,10)) == 0
end

@testset "norm" begin
    for a in (Zeros{Int}(5), Zeros(5,3), Zeros(2,3,3),
                Ones{Int}(5), Ones(5,3), Ones(2,3,3),
                Fill(2.3,5), Fill([2.3,4.2],5), Fill(4)),
        p in (-Inf, 0, 0.1, 1, 2, 3, Inf)
        @test norm(a,p) ≈ norm(Array(a),p)
    end
end

@testset "multiplication" begin
    for T in (Float64, ComplexF64)
        fv = T == Float64 ? Float64(1.6) : ComplexF64(1.6, 1.3)
        n  = 10
        k  = 12
        m  = 15
        fillvec = Fill(fv, k)
        fillmat = Fill(fv, k, m)
        A  = rand(ComplexF64, n, k)
        @test A*fillvec ≈ A*Array(fillvec)
        @test A*fillmat ≈ A*Array(fillmat)
        A  = rand(ComplexF64, k, n)
        @test transpose(A)*fillvec ≈ transpose(A)*Array(fillvec)
        @test transpose(A)*fillmat ≈ transpose(A)*Array(fillmat)
        @test adjoint(A)*fillvec ≈ adjoint(A)*Array(fillvec)
        @test adjoint(A)*fillmat ≈ adjoint(A)*Array(fillmat)
    end
end

@testset "dot products" begin
    n = 15
    o = Ones(1:n)
    z = Zeros(1:n)
    D = Diagonal(o)
    Z = Diagonal(z)

    Random.seed!(5)
    u = rand(n)
    v = rand(n)

    @test dot(u, D, v) == dot(u, v)
    @test dot(u, 2D, v) == 2dot(u, v)
    @test dot(u, Z, v) == 0

    @test_throws DimensionMismatch dot(u[1:end-1], D, v)
    @test_throws DimensionMismatch dot(u[1:end-1], D, v[1:end-1])

    @test_throws DimensionMismatch dot(u, 2D, v[1:end-1])
    @test_throws DimensionMismatch dot(u, 2D, v[1:end-1])

    @test_throws DimensionMismatch dot(u, Z, v[1:end-1])
    @test_throws DimensionMismatch dot(u, Z, v[1:end-1])
end

if VERSION ≥ v"1.5"
    @testset "print" begin
        @test stringmime("text/plain", Zeros(3)) == "3-element Zeros{Float64}"
        @test stringmime("text/plain", Ones(3)) == "3-element Ones{Float64}"
        @test stringmime("text/plain", Fill(7,2)) == "2-element Fill{$Int}: entries equal to 7"
        @test stringmime("text/plain", Zeros(3,2)) == "3×2 Zeros{Float64}"
        @test stringmime("text/plain", Ones(3,2)) == "3×2 Ones{Float64}"
        @test stringmime("text/plain", Fill(7,2,3)) == "2×3 Fill{$Int}: entries equal to 7"
        @test stringmime("text/plain", Eye(5)) == "5×5 Eye{Float64}"
    end
end

@testset "reshape" begin
    @test reshape(Fill(2,6),2,3) ≡ reshape(Fill(2,6),(2,3)) ≡ reshape(Fill(2,6),:,3) ≡ reshape(Fill(2,6),2,:) ≡ Fill(2,2,3)
    @test reshape(Fill(2,6),big(2),3) == reshape(Fill(2,6), (big(2),3)) == reshape(Fill(2,6), big(2),:) == Fill(2,big(2),3)
    @test_throws DimensionMismatch reshape(Fill(2,6),2,4)
    @test reshape(Ones(6),2,3) ≡ reshape(Ones(6),(2,3)) ≡ reshape(Ones(6),:,3) ≡ reshape(Ones(6),2,:) ≡ Ones(2,3)
    @test reshape(Zeros(6),2,3) ≡ Zeros(2,3)
    @test reshape(Zeros(6),big(2),3) == Zeros(big(2),3)
    @test reshape(Fill(2,2,3),Val(1)) ≡ Fill(2,6)
    @test reshape(Fill(2, 2), (2, )) ≡ Fill(2, 2)
end

@testset "lmul!/rmul!" begin
    z = Zeros(1_000_000_000_000)
    @test lmul!(2.0,z) === z
    @test rmul!(z,2.0) === z
    @test_throws ArgumentError lmul!(Inf,z)
    @test_throws ArgumentError rmul!(z,Inf)

    x = Fill([1,2],1_000_000_000_000)
    @test lmul!(1.0,x) === x
    @test rmul!(x,1.0) === x
    @test_throws ArgumentError lmul!(2.0,x)
    @test_throws ArgumentError rmul!(x,2.0)
end

@testset "Modified" begin
    @testset "Diagonal{<:Fill}" begin
        D = Diagonal(Fill(Fill(0.5,2,2),10))
        @test @inferred(D[1,1]) === Fill(0.5,2,2)
        @test @inferred(D[1,2]) === Fill(0.0,2,2)
        @test axes(D) == (Base.OneTo(10),Base.OneTo(10))
        D = Diagonal(Fill(Zeros(2,2),10))
        @test @inferred(D[1,1]) === Zeros(2,2)
        @test @inferred(D[1,2]) === Zeros(2,2)
        D = Diagonal([Zeros(1,1), Zeros(2,2)])
        @test @inferred(D[1,1]) === Zeros(1,1)
        @test @inferred(D[1,2]) === Zeros(1,2)

        @test_throws ArgumentError Diagonal(Fill(Ones(2,2),10))[1,2]
    end
    @testset "Triangular" begin
        U = UpperTriangular(Ones(3,3))
        @test U == UpperTriangular(ones(3,3))
        @test axes(U) == (Base.OneTo(3),Base.OneTo(3))
    end
end

@testset "Trues" begin
    @test Trues(2,3) == Trues((2,3)) == trues(2,3)
    @test Falses(2,3) == Falses((2,3)) == falses(2,3)
    dim = (4,5)
    mask = Trues(dim)
    x = randn(dim)
    @test x[mask] == vec(x) # getindex
    y = similar(x)
    y[mask] = x # setindex!
    @test y == x
    @test_throws BoundsError ones(3)[Trues(2)]
    @test_throws BoundsError setindex!(ones(3), zeros(3), Trues(2))
    @test_throws DimensionMismatch setindex!(ones(2), zeros(3), Trues(2))
end

@testset "FillArray interface" begin
    @testset "SubArray" begin
        a = Fill(2.0,5)
        v = SubArray(a,(1:2,))
        @test FillArrays.getindex_value(v) == FillArrays.unique_value(v) == 2.0
        @test convert(Fill, v) ≡ Fill(2.0,2)
    end

    @testset "views" begin
        a = Fill(2.0,5)
        v = view(a,1:2)
        @test v isa Fill
        @test FillArrays.getindex_value(v) == FillArrays.unique_value(v) == 2.0
        @test convert(Fill, v) ≡ Fill(2.0,2)
        @test view(a,1) ≡ Fill(2.0)
    end

    @testset "view with bool" begin
        a = Fill(2.0,5)
        @test a[[true,false,false,true,false]] ≡ view(a,[true,false,false,true,false])
        a = Fill(2.0,2,2)
        @test a[[true false; false true]] ≡ view(a, [true false; false true])
    end

    @testset "adjtrans" begin
        a = Fill(2.0,5)
        @test FillArrays.getindex_value(a') == FillArrays.unique_value(a') == 2.0
        @test convert(Fill, a') ≡ Fill(2.0,1,5)
        @test FillArrays.getindex_value(transpose(a)) == FillArrays.unique_value(transpose(a)) == 2.0
        @test convert(Fill, transpose(a)) ≡ Fill(2.0,1,5)
    end
end
