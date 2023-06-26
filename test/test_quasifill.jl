using QuasiArrays, FillArrays, LinearAlgebra, StaticArrays, Random, Base64, Test
import QuasiArrays: AbstractQuasiFill

@testset "QuasiFill" begin
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
            @test A[1,:] ≡ A[1,Base.IdentityUnitRange(bx)] ≡ Fill(2.0,(Base.IdentityUnitRange(bx),))

            @test A[:,:] == A[Inclusion(ax),Base.IdentityUnitRange(bx)] == A[Inclusion(ax),:] == A[:,Base.IdentityUnitRange(bx)] == A
            @test Z[:,2] ≡ Z[Inclusion(ax),2] ≡ QuasiZeros(ax)
            @test Z[1,:] ≡ Z[1,Base.IdentityUnitRange(bx)] ≡ Zeros((Base.IdentityUnitRange(bx),))
            @test Z[:,:] == Z[Inclusion(ax),Base.IdentityUnitRange(bx)] == Z[Inclusion(ax),:] == Z[:,Base.IdentityUnitRange(bx)] == Z

            cx = 1:2
            A = QuasiFill(2.0,ax,bx,cx)
            Z = QuasiZeros(ax,bx,cx)
            @test A[:,2,1] ≡ A[Inclusion(ax),2,1] ≡ QuasiFill(2.0,ax)
            @test A[1,:,1] ≡ A[1,Base.IdentityUnitRange(bx),1] ≡ Fill(2.0,(Base.IdentityUnitRange(bx),))
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
        @test QuasiArray(randn(3, 4),ax,bx)' * QuasiZeros(ax) === Zeros((axes(B,2),))
        @test QuasiArray(randn(4),bx)' * QuasiZeros(bx) == zero(Float64)
        @test QuasiArray([1, 2, 3],ax)' * QuasiZeros{Int}(ax) == zero(Int)
        @test_broken QuasiArray([SVector(1,2)', SVector(2,3)', SVector(3,4)'],ax)' * QuasiZeros{Int}(ax) == SVector(0,0)


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


    @testset "Broadcast" begin
        ax = [1,3,4]
        x = QuasiFill(5,ax)
        @test (.+)(x) ≡ x
        @test (.-)(x) ≡ -x
        @test exp.(x) ≡ QuasiFill(exp(5),ax)
        @test x .+ 1 ≡ QuasiFill(6,ax)
        @test 1 .+ x ≡ QuasiFill(6,ax)
        @test x .+ x ≡ QuasiFill(10,ax)
        @test x .+ QuasiOnes(ax) ≡ QuasiFill(6.0,ax)
        f = (x,y) -> cos(x*y)
        @test f.(x, QuasiOnes(ax)) ≡ QuasiFill(f(5,1.0),ax)

        y = QuasiOnes(ax,ax)
        @test (.+)(y) ≡ QuasiOnes(ax,ax)
        @test (.-)(y) ≡ QuasiFill(-1.0,ax,ax)
        @test exp.(y) ≡ QuasiFill(exp(1),ax,ax)
        @test y .+ 1 ≡ QuasiFill(2.0,ax,ax)
        @test y .+ y ≡ QuasiFill(2.0,ax,ax)
        @test y .* y ≡ y ./ y ≡ y .\ y ≡ y

        @test QuasiZeros{Int}(5) .+ QuasiZeros(5) isa QuasiZeros{Float64}

        # Test for conj, real and imag with complex element types
        @test conj(QuasiZeros{ComplexF64}(ax)) isa QuasiZeros{ComplexF64}
        @test conj(QuasiZeros{ComplexF64}(ax,ax)) isa QuasiZeros{ComplexF64}
        @test conj(QuasiOnes{ComplexF64}(ax)) isa QuasiOnes{ComplexF64}
        @test conj(QuasiOnes{ComplexF64}(ax,ax)) isa QuasiOnes{ComplexF64}
        @test real(QuasiZeros{Float64}(ax)) isa QuasiZeros{Float64}
        @test real(QuasiZeros{Float64}(ax,ax)) isa QuasiZeros{Float64}
        @test real(QuasiZeros{ComplexF64}(ax)) isa QuasiZeros{Float64}
        @test real(QuasiZeros{ComplexF64}(ax,ax)) isa QuasiZeros{Float64}
        @test real(QuasiOnes{Float64}(ax)) isa QuasiOnes{Float64}
        @test real(QuasiOnes{Float64}(ax,ax)) isa QuasiOnes{Float64}
        @test real(QuasiOnes{ComplexF64}(ax)) isa QuasiOnes{Float64}
        @test real(QuasiOnes{ComplexF64}(ax,ax)) isa QuasiOnes{Float64}
        @test imag(QuasiZeros{Float64}(ax)) isa QuasiZeros{Float64}
        @test imag(QuasiZeros{Float64}(ax,ax)) isa QuasiZeros{Float64}
        @test imag(QuasiZeros{ComplexF64}(ax)) isa QuasiZeros{Float64}
        @test imag(QuasiZeros{ComplexF64}(ax,ax)) isa QuasiZeros{Float64}
        @test imag(QuasiOnes{Float64}(ax)) isa QuasiZeros{Float64}
        @test imag(QuasiOnes{Float64}(ax,ax)) isa QuasiZeros{Float64}
        @test imag(QuasiOnes{ComplexF64}(ax)) isa QuasiZeros{Float64}
        @test imag(QuasiOnes{ComplexF64}(ax,ax)) isa QuasiZeros{Float64}


        @testset "Special Zeros/Ones" begin
            @test broadcast(+,QuasiZeros(ax)) ≡ broadcast(-,QuasiZeros(ax)) ≡ QuasiZeros(ax)
            @test broadcast(+,QuasiOnes(ax)) ≡ QuasiOnes(ax)

            @test QuasiZeros(ax) .* QuasiOnes(ax) ≡ QuasiZeros(ax) .* 1 ≡ QuasiZeros(ax)
            @test QuasiZeros(ax) .* QuasiFill(5.0, ax) ≡ QuasiZeros(ax) .* 5.0 ≡ QuasiZeros(ax)
            @test QuasiOnes(ax) .* QuasiZeros(ax) ≡ 1 .* QuasiZeros(ax) ≡ QuasiZeros(ax)
            @test QuasiFill(5.0, ax) .* QuasiZeros(ax) ≡ 5.0 .* QuasiZeros(ax) ≡ QuasiZeros(ax)

            @test QuasiZeros(ax) ./ QuasiOnes(ax) ≡ QuasiZeros(ax) ./ 1 ≡ QuasiZeros(ax)
            @test QuasiZeros(ax) ./ QuasiFill(5.0, ax) ≡ QuasiZeros(ax) ./ 5.0 ≡ QuasiZeros(ax)
            @test QuasiOnes(ax) .\ QuasiZeros(ax) ≡ 1 .\ QuasiZeros(ax) ≡ QuasiZeros(ax)
            @test QuasiFill(5.0, ax) .\ QuasiZeros(ax) ≡ 5.0 .\ QuasiZeros(ax) ≡ QuasiZeros(ax)

            @test conj.(QuasiZeros(5)) ≡ QuasiZeros(5)
            @test conj.(QuasiZeros{ComplexF64}(5)) ≡ QuasiZeros{ComplexF64}(5)

            @test_throws DimensionMismatch broadcast(*, QuasiOnes(ax), 1:6)
            @test_throws DimensionMismatch broadcast(*, 1:6, QuasiOnes(ax))
            @test_throws DimensionMismatch broadcast(*, QuasiFill(1,ax), 1:6)
            @test_throws DimensionMismatch broadcast(*, 1:6, QuasiFill(1,ax))

            @testset "Number" begin
                @test broadcast(*, QuasiZeros(ax), 2) ≡ broadcast(*, 2, QuasiZeros(ax)) ≡ QuasiZeros(ax)
            end

            @testset "Nested" begin
                a = QuasiArray(randn(3), ax)
                b = QuasiArray(randn(3), ax)
                @test a .\ b .* QuasiZeros(ax) ≡ QuasiZeros(ax)
                @test broadcast(*, QuasiZeros(ax), Base.Broadcast.broadcasted(\, a, b)) ≡ QuasiZeros(ax)
            end

            @testset "array-valued" begin
                a = QuasiArray(1:3, ax)
                @test broadcast(*, QuasiFill([1,2],ax), a) == broadcast(*, a, QuasiFill([1,2],ax)) == QuasiArray((1:3) .* Fill([1,2],3), ax)
                @test broadcast(*, QuasiFill([1,2],ax), QuasiZeros(ax)) == broadcast(*, QuasiZeros(ax), QuasiFill([1,2],ax))
                @test broadcast(*, QuasiFill([1,2],ax), QuasiZeros(ax)) isa QuasiFill{Vector{Float64}}
                @test broadcast(*, QuasiArray([[1,2], [3,4,5], [1,2]], ax), QuasiZeros(ax)) == broadcast(*, QuasiZeros(ax), QuasiArray([[1,2], [3,4,5], [1,2]], ax))
            end

            @testset "NaN" begin
                @test QuasiZeros(ax) ./ QuasiZeros(ax) ≡ QuasiZeros(ax) .\ QuasiZeros(ax) ≡ QuasiFill(NaN,ax)
                @test QuasiZeros{Int}(ax,ax) ./ QuasiZeros{Int}(ax) ≡ QuasiZeros{Int}(ax) .\ QuasiZeros{Int}(ax,ax) ≡ QuasiFill(NaN,ax,ax)
            end

            @testset "Addition" begin
                a = QuasiArray(1:3,ax)
                @test QuasiZeros{Int}(ax) .+ a == a .+ QuasiZeros{Int}(ax) == a .- QuasiZeros{Int}(ax) == a
                @test QuasiZeros(ax) .+ a == a .+ QuasiZeros(ax) == a .- QuasiZeros(ax) == a
                @test QuasiZeros{Int}(ax) .+ QuasiFill(1,ax) ≡ QuasiFill(1,ax) .+ QuasiZeros{Int}(ax) ≡ QuasiFill(1,ax) .- QuasiZeros{Int}(ax) ≡ QuasiFill(1,ax)
                @test_throws DimensionMismatch QuasiZeros{Int}([1,2]) .+ a
                @test_throws DimensionMismatch a .+ QuasiZeros{Int}([1,2])
            end
        end

        @testset "support Ref" begin
            @test QuasiFill(1,ax) .- 1 ≡ QuasiFill(1,ax) .- Ref(1) ≡ QuasiFill(1,ax) .- Ref(1I)
            @test QuasiFill([1 2; 3 4],ax) .- Ref(1I) == QuasiFill([0 2; 3 3],ax)
            @test Ref(1I) .+ QuasiFill([1 2; 3 4],ax) == QuasiFill([2 2; 3 5],ax)
        end

        @testset "Special Ones" begin
            a = QuasiArray(1:3,ax)
            @test QuasiOnes{Int}(ax) .* a == a .* QuasiOnes{Int}(ax) == a
            @test QuasiOnes(ax) .* a == a .* QuasiOnes(ax) == a
            @test QuasiOnes{Int}(5) .* QuasiOnes{Int}(5) == QuasiOnes{Int}(5)
            @test_throws DimensionMismatch QuasiOnes{Int}([1,2]) .* a
            @test_throws DimensionMismatch a .* QuasiOnes{Int}([1,2])
            @test_throws DimensionMismatch QuasiOnes{Int}(ax) .* QuasiOnes{Int}([1,2])
        end

        @testset "Zeros -" begin
            @test QuasiZeros(ax) - QuasiZeros(ax) ≡ QuasiZeros(ax)
            @test QuasiOnes(ax) - QuasiZeros(ax) ≡ QuasiOnes(ax)
            @test QuasiOnes(ax) - QuasiOnes(ax) isa QuasiZeros
            @test QuasiFill(1,ax) - QuasiZeros(ax) ≡ QuasiFill(1.0,ax)

            @test QuasiZeros(ax) .- QuasiZeros(ax) ≡ QuasiZeros(ax)
            @test QuasiOnes(ax) .- QuasiZeros(ax) ≡ QuasiOnes(ax)
            @test QuasiOnes(ax) .- QuasiOnes(ax) ≡ QuasiZeros(ax)
            @test QuasiFill(1,ax) .- QuasiZeros(ax) ≡ QuasiFill(1.0,ax)
        end

        @testset "Zero .*" begin
            ax = [1,3,4]
            a = QuasiArray(randn(3),ax)
            b = QuasiArray(1:3,ax)
            @test QuasiZeros{Int}(ax) .* QuasiZeros{Int}(ax) ≡ QuasiZeros{Int}(ax)
            @test a .* QuasiZeros(ax) isa QuasiZeros
            @test QuasiZeros(ax) .* a isa QuasiZeros
            @test b .* QuasiZeros(ax) isa QuasiZeros
            @test QuasiZeros(ax) .* b isa QuasiZeros
            @test_throws DimensionMismatch (1:11) .* QuasiZeros(ax)
        end
    end

    @testset "map" begin
        ax = [1,3,4]
        x = QuasiOnes(ax)
        @test map(exp,x) === QuasiFill(exp(1.0),ax)
        @test map(isone,x) === QuasiFill(true,ax)

        x = QuasiZeros(ax)
        @test map(exp,x) === exp.(x)

        x = QuasiFill(2,ax,ax)
        @test map(exp,x) === QuasiFill(exp(2),ax,ax)
    end


    @testset "0-dimensional" begin
        A = QuasiFill{Int,0,Tuple{}}(3, ())

        @test A[] ≡ 3
        @test A ≡ QuasiFill{Int,0}(3, ()) ≡ QuasiFill(3, ()) ≡ QuasiFill(3)
        @test size(A) == ()
        @test axes(A) == ()

        A = QuasiOnes{Int,0,Tuple{}}(())
        @test A[] ≡ 1
        @test A ≡ QuasiOnes{Int,0}(()) ≡ QuasiOnes{Int}(()) ≡ QuasiOnes{Int}()

        A = QuasiZeros{Int,0,Tuple{}}(())
        @test A[] ≡ 0
        @test A ≡ QuasiZeros{Int,0}(()) ≡ QuasiZeros{Int}(()) ≡ QuasiZeros{Int}()
    end


    @testset "Eye identity ops" begin
        ax = [1,3,4]
        m = QuasiEye(ax)
        D = QuasiDiagonal(QuasiFill(2,ax))

        for op in (permutedims, inv)
            @test op(m) === m
        end
        @test permutedims(D) ≡ D
        @test inv(D) ≡ QuasiDiagonal(QuasiFill(1/2,ax))

        @test copy(m) ≡ m
        @test copy(D) ≡ D
        @test LinearAlgebra.copy_oftype(m, Int) ≡ QuasiEye{Int}(ax)
        @test LinearAlgebra.copy_oftype(D, Float64) ≡ QuasiDiagonal(QuasiFill(2.0,ax))
    end

    @testset "Adjoint/Transpose/permutedims" begin
        ax = [1,3,4]
        bx = 2:5
        @test QuasiOnes{ComplexF64}(ax,bx)' ≡ transpose(QuasiOnes{ComplexF64}(ax,bx)) ≡ QuasiOnes{ComplexF64}(bx,ax)
        @test QuasiZeros{ComplexF64}(ax,bx)' ≡ transpose(QuasiZeros{ComplexF64}(ax,bx)) ≡ QuasiZeros{ComplexF64}(bx,ax)
        @test QuasiFill(1+im, ax, bx)' ≡ QuasiFill(1-im, bx,ax)
        @test transpose(QuasiFill(1+im, ax, bx)) ≡ QuasiFill(1+im, bx,ax)
        @test QuasiOnes(ax)' isa QuasiAdjoint # Vectors still need special dot product
        @test QuasiFill([1+im 2; 3 4; 5 6], ax, bx)' == QuasiFill([1+im 2; 3 4; 5 6]',bx, ax)
        @test transpose(QuasiFill([1+im 2; 3 4; 5 6], ax, bx)) == QuasiFill(transpose([1+im 2; 3 4; 5 6]),bx, ax)

        @test permutedims(QuasiOnes(ax)) ≡ QuasiOnes(1,ax)
        @test permutedims(QuasiZeros(ax)) ≡ QuasiZeros(1,ax)
        @test permutedims(QuasiFill(2.0,ax)) ≡ QuasiFill(2.0,1,ax)
        @test permutedims(QuasiOnes(ax,bx)) ≡ QuasiOnes(bx,ax)
        @test permutedims(QuasiZeros(ax,bx)) ≡ QuasiZeros(bx,ax)
        @test permutedims(QuasiFill(2.0,ax,bx)) ≡ QuasiFill(2.0,bx,ax)
    end

    @testset "setindex!/fill!" begin
        ax = [1,3,4]
        bx = 2:5

        F = QuasiFill(1,ax)
        @test (F[1] = 1) == 1
        @test_throws BoundsError (F[11] = 1)
        @test_throws ArgumentError (F[3] = 2)


        F = QuasiFill(1,ax,bx)
        @test (F[3,3] = 1) == 1
        @test_throws BoundsError (F[51] = 1)
        @test_throws BoundsError (F[1,6] = 1)
        @test_throws ArgumentError (F[1,2] = 2)

        @test (F[:,2] .= 1) == QuasiFill(1,ax)
        @test_throws ArgumentError (F[:,2] .= 2)

        @test fill!(F,1) == F
        @test_throws ArgumentError fill!(F,2)
    end

    @testset "mult" begin
        ax = [1,3,4]
        bx = 2:5
        cx = 1:3

        @test QuasiFill(2,ax)*QuasiFill(3,Base.OneTo(1),bx) ≡ QuasiFill(6,ax,bx)
        @test QuasiFill(2,ax,bx)*QuasiFill(3,bx,cx) ≡ QuasiFill(6*length(bx),ax,cx)
        @test QuasiFill(2,ax,bx)*QuasiFill(3,bx) ≡ QuasiFill(6*length(bx),ax)
        @test_throws DimensionMismatch QuasiFill(2,bx)*QuasiFill(3,2,cx)
        @test_throws DimensionMismatch QuasiFill(2,3,bx)*QuasiFill(3,2,cx)

        @test QuasiOnes(bx)*QuasiFill(3,Base.OneTo(1),cx) ≡ QuasiFill(3.0,bx,cx)
        @test QuasiOnes(ax,bx)*QuasiFill(3,bx,cx) == QuasiFill(3length(bx),ax,cx)
        @test QuasiOnes(ax,bx)*QuasiFill(3,bx) == QuasiFill(3length(bx),ax)

        @test QuasiFill(2,ax)*QuasiOnes(Base.OneTo(1),bx) ≡ QuasiFill(2.0,ax,bx)
        @test QuasiFill(2,ax,bx)*QuasiOnes(bx,cx) == QuasiFill(2length(bx),ax,cx)
        @test QuasiFill(2,ax,bx)*QuasiOnes(bx) == QuasiFill(2length(bx),ax)

        @test QuasiOnes(ax)*QuasiOnes(Base.OneTo(1),bx) == QuasiOnes(ax,bx)
        @test QuasiOnes(ax,bx)*QuasiOnes(bx,cx) == QuasiFill(length(bx),ax,cx)
        @test QuasiOnes(ax,bx)*QuasiOnes(bx) == QuasiFill(length(bx),ax)

        @test QuasiZeros(ax)*QuasiFill(3,Base.OneTo(1),bx) ≡   QuasiZeros(ax,bx)
        @test QuasiZeros(ax,bx)*QuasiFill(3,bx,cx) ≡ QuasiZeros(ax,cx)
        @test QuasiZeros(ax,bx)*QuasiFill(3,bx) ≡    QuasiZeros(ax)

        @test QuasiFill(2,ax)*  QuasiZeros(Base.OneTo(1),bx) ≡  QuasiZeros(ax,bx)
        @test QuasiFill(2,ax,bx)*QuasiZeros(bx,cx) ≡ QuasiZeros(ax,cx)
        @test QuasiFill(2,ax,bx)*QuasiZeros(bx) ≡    QuasiZeros(ax)

        @test QuasiZeros(ax)*QuasiZeros(Base.OneTo(1),bx) ≡ QuasiZeros(ax,bx)
        @test QuasiZeros(ax,bx)*QuasiZeros(bx,cx) ≡ QuasiZeros(ax,cx)
        @test QuasiZeros(ax,bx)*QuasiZeros(bx) ≡ QuasiZeros(ax)

        a = QuasiArray(randn(4),bx)
        A = QuasiArray(randn(1,4),Base.OneTo(1),bx)


        @test QuasiFill(2,ax,Base.OneTo(1))*A ≈ QuasiMatrix(QuasiFill(2,ax,Base.OneTo(1)))*A
        @test QuasiFill(2,ax,bx)*a ≈ QuasiMatrix(QuasiFill(2,ax,bx))*a
        @test QuasiOnes(ax,Base.OneTo(1))*A ≈ QuasiMatrix(QuasiOnes(ax,Base.OneTo(1)))*A
        @test QuasiOnes(ax,bx)*a ≈ QuasiMatrix(QuasiOnes(ax,bx))*a
        @test QuasiZeros(ax,Base.OneTo(1))*A == QuasiZeros(ax,bx)
        @test QuasiZeros(ax,bx)*a == QuasiZeros(ax)

        @test A*QuasiFill(2,bx) ≈ A*QuasiVector(QuasiFill(2,bx))
        @test A*QuasiFill(2,bx,cx) ≈ A*QuasiMatrix(QuasiFill(2,bx,cx))
        @test A*QuasiOnes(bx) ≈ A*QuasiVector(QuasiOnes(bx))
        @test A*QuasiOnes(bx,cx) ≈ A*QuasiMatrix(QuasiOnes(bx,cx))
        @test A*QuasiZeros(bx)  ≡ Zeros((Base.OneTo(1),))
        @test A*QuasiZeros(bx,cx) ≡ QuasiZeros(Base.OneTo(1),cx)

        D = QuasiDiagonal(a)
        @test QuasiZeros(ax,bx)*D ≡ QuasiZeros(ax,bx)
        @test D*QuasiZeros(bx,cx) ≡ QuasiZeros(bx,cx)
        @test D*QuasiZeros(bx) ≡ QuasiZeros(bx)

        D = QuasiDiagonal(QuasiFill(2,bx))
        @test D * QuasiOnes(bx) == QuasiFill(2.0,bx)
        @test D * QuasiOnes(bx,cx) == QuasiFill(2.0,bx,cx)
        @test QuasiOnes(ax,bx) * D == QuasiFill(2.0,ax,bx)
    end

    @testset "zeros/ones/fill" begin
        ax = Inclusion([1,3,4])
        @test zero(ax) ≡ QuasiZeros{Int}((ax,))
        @test one(ax) ≡ QuasiOnes{Int}((ax,))
        @test zeros(ax) ≡ zeros(Float64,ax) ≡ QuasiZeros((ax,))
        @test zeros(Base.OneTo(3), ax) ≡ zeros(Float64,Base.OneTo(3),ax) ≡ QuasiZeros((Base.OneTo(3),ax))
        @test ones(ax) ≡ ones(Float64,ax) ≡ QuasiOnes((ax,))
        @test ones(Base.OneTo(3), ax) ≡ ones(Float64,Base.OneTo(3),ax) ≡ QuasiOnes((Base.OneTo(3),ax))
        @test fill(2,ax) ≡ QuasiFill(2,(ax,))
        @test fill(2,Base.OneTo(3), ax) ≡ QuasiFill(2,(Base.OneTo(3),ax))
    end

    @testset "Static eval" begin
        o = QuasiOnes([1,3,4])
        @test view(o,SVector(1,3)) ≡ o[SVector(1,3)] ≡ Ones((SOneTo(2),))
    end

    @testset "UniformScaling" begin
        A = QuasiArray(randn(3,3), 0:0.5:1,0:0.5:1)
        @test A+2I ≈ 2I+A ≈ QuasiArray(A.parent+2I, 0:0.5:1,0:0.5:1)
        @test A-2I ≈ -(2I-A) ≈ QuasiArray(A.parent-2I, 0:0.5:1,0:0.5:1)
        @test A*(2I) ≈ (2I)*A ≈ 2A
        @test A/(2I) ≈ (2I)\A ≈ A/2

        B = QuasiArray(randn(3,3), 0:0.5:1,1:0.5:2)
        @test_throws DimensionMismatch B+I
        @test_throws DimensionMismatch I+B
        @test B*I ≈ I*B ≈ B
    end

    @testset "show" begin
        @test stringmime("text/plain",ones(Inclusion([1,2,3]))) == "ones(Inclusion([1, 2, 3]))"
        @test stringmime("text/plain",zeros(Inclusion([1,2,3]))) == "zeros(Inclusion([1, 2, 3]))"
        @test stringmime("text/plain",fill(2,Inclusion([1,2,3]))) == "fill(2, Inclusion([1, 2, 3]))"
    end

    @testset "Mul" begin
        A = QuasiArray(randn(3,3), 0:0.5:1, Base.OneTo(3))
        @test A * Zeros(axes(A,2)) ≡ QuasiZeros(axes(A,1))
        B = QuasiArray(randn(3,3), 0:0.5:1,1:0.5:2)
        @test B * QuasiZeros(axes(B,2)) ≡ QuasiZeros(axes(B,1))

        @test_throws DimensionMismatch A * Zeros(2)
        @test_throws DimensionMismatch B * QuasiZeros(0:0.5:1)
        @test_throws DimensionMismatch FillArrays.mult_zeros(B, QuasiZeros(0:0.5:1))
    end

    @testset "isone" begin
        @test isone(QuasiFill(1, 0:0.5:1))
        @test isone(QuasiOnes(0:0.5:1))
        @test !isone(QuasiZeros(0:0.5:1))
    end
end