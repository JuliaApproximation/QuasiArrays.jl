# This file is based on a part of Julia. License is MIT: https://julialang.org/license

using QuasiArrays, LazyArrays, Base64, StaticArrays, Test
import Base: OneTo, Slice
import Base.Broadcast: check_broadcast_axes, newindex, broadcasted, broadcastable, Broadcasted, DefaultArrayStyle, BroadcastStyle
import QuasiArrays: QuasiCartesianIndex, QuasiCartesianIndices, DefaultQuasiArrayStyle, SubQuasiArray

@testset "Broadcasting" begin
    Z = QuasiArray(zeros(3,4),(0:0.5:1,0:3))
    z = QuasiArray(zeros(3),(0:0.5:1,))
    ax = (Inclusion(0:0.5:1),Slice(0:3))
    @test @inferred(Broadcast.combine_axes(Z,Z)) == ax
    @test @inferred(Broadcast.combine_axes(Z,z)) == ax
    @test @inferred(Broadcast.combine_axes(z,Z)) == ax
    @test @inferred(Broadcast.combine_axes(z,Z, zeros(1))) == ax

    check_broadcast_axes(ax, Z)
    check_broadcast_axes(ax, z)
    check_broadcast_axes(ax, Z, z)
    check_broadcast_axes(ax, Z, 1)
    check_broadcast_axes(ax, 5, 2)
    @test_throws DimensionMismatch check_broadcast_axes(ax, zeros(2,5))
    @test_throws DimensionMismatch check_broadcast_axes(ax, zeros(3,4))
    @test_throws DimensionMismatch check_broadcast_axes(ax, zeros(3,4,2))
    @test_throws DimensionMismatch check_broadcast_axes(ax, zeros(3,5), zeros(2))

    ci(x) = QuasiCartesianIndex(x...)
    @test @inferred(newindex(ci((2,2)), (true, true), (-1,-1)))   == ci((2,2))
    @test @inferred(newindex(ci((2,2)), (true, false), (-1,-1)))  == ci((2,-1))
    @test @inferred(newindex(ci((2,2)), (false, true), (-1,-1)))  == ci((-1,2))
    @test @inferred(newindex(ci((2,2)), (false, false), (-1,-1))) == ci((-1,-1))
    @test @inferred(newindex(ci((2,2)), (true,), (-1,-1)))   == ci((2,))
    @test @inferred(newindex(ci((2,2)), (true,), (-1,)))   == ci((2,))
    @test @inferred(newindex(ci((2,2)), (false,), (-1,))) == ci((-1,))
    @test @inferred(newindex(ci((2,2)), (), ())) == ci(())

    @test eltype(QuasiCartesianIndices(Inclusion.((0:0.1:1,0:0.2:1)))) == QuasiCartesianIndex{2,NTuple{2,Float64}}

    A = QuasiArray([1 0; 0 1], (0:0.5:0.5, 1:0.5:1.5))
    b = QuasiArray([1,2], (0:0.5:0.5,))
    @test @inferred(collect(eachindex(A))) == [QuasiCartesianIndex(0.0,1.0) QuasiCartesianIndex(0.0,1.5);
                                QuasiCartesianIndex(0.5,1.0) QuasiCartesianIndex(0.5,1.5)]
    @test collect(eachindex(A)) isa Matrix{QuasiCartesianIndex{2,NTuple{2,Float64}}}
    @test broadcastable(A) === A
    @test Base.BroadcastStyle(typeof(A)) === DefaultQuasiArrayStyle{2}()
    bc = broadcasted(+, A)
    @test bc isa Broadcasted{DefaultQuasiArrayStyle{2}}
    @test broadcast(+, A) == A
    @test broadcast(+, A, A) == QuasiArray([2 0; 0 2],A.axes)
    @test Base.BroadcastStyle(Base.BroadcastStyle(typeof(A)), Base.BroadcastStyle(Int)) == DefaultQuasiArrayStyle{2}()
    @test broadcast(+, A, 1) == QuasiArray([2 1; 1 2],A.axes)
    @test broadcast(+, A, b) == QuasiArray([2 1; 2 3],A.axes)
    A = QuasiArray([1 0; 0 1], (0:0.5:0.5, 1:0.5:1.5)); @test broadcast!(+, A, A, b) == QuasiArray([2 1; 2 3],A.axes)
    A = QuasiArray([1 0], (0:0, 1:0.5:1.5)); @test_throws DimensionMismatch broadcast!(+, A, A, b)
    A = QuasiArray([1 2], (0:0, 1:0.5:1.5)); B = QuasiArray([3,4], (0:0.5:0.5,));
    @test A .* B == QuasiArray([ 3 6; 4 8], (0:0.5:0.5,1:0.5:1.5))

    @testset "f.(args...) syntax (#15032)" begin
        x = QuasiVector([1, 3.2, 4.7],0:0.5:1)
        y = QuasiVector([3.5, pi, 1e-4],0:0.5:1)
        α = 0.2342
        @test sin.(x) == broadcast(sin, x)
        @test atan.(x, y) == broadcast(atan, x, y)
        @test atan.(x, y') == broadcast(atan, x, y')
        @test atan.(x, α) == broadcast(atan, x, α)
        @test atan.(α, y') == broadcast(atan, α, y')
    end

    @testset "sin" begin
        a = sin.(QuasiVector([1, 2],0:0.5:0.5))
        @test isa(a, QuasiVector{Float64})
        @test a ≈ QuasiVector([0.8414709848078965, 0.9092974268256817],0:0.5:0.5)
    end

    @testset "loop fusion" begin
        v = QuasiVector(1:10, range(0; stop=1,length=10))
        @test (x->x+1).((x->x+2).((x->x+3).(v))) == QuasiVector(7:16,range(0; stop=1,length=10))
        let A = QuasiArray([sqrt(i)+j for i = 1:3, j=1:4], (0:0.5:1,0:0.5:1.5))
            @test atan.(log.(A), sum(A, dims=1)) == broadcast(atan, broadcast(log, A), sum(A, dims=1))
        end
        let x = sin.(v)
            @test atan.((x->x+1).(x), (x->x+2).(x)) == broadcast(atan, x.+1, x.+2)
            @test sin.(atan.([x.+1,x.+2]...)) == sin.(atan.(x.+1 ,x.+2)) == @. sin(atan(x+1,x+2))
            @test sin.(atan.(x, 3.7)) == broadcast(x -> sin(atan(x,3.7)), x)
            @test atan.(x, 3.7) == broadcast(x -> atan(x,3.7), x) == broadcast(atan, x, 3.7)
        end
        # fusion with splatted args:
        let x = sin.(v), a = [x]
            @test cos.(x) == cos.(a...)
            @test atan.(x,x) == atan.(a..., a...) == atan.([x, x]...)
            @test atan.(x, cos.(x)) == atan.(a..., cos.(x)) == broadcast(atan, x, cos.(a...)) == broadcast(atan, a..., cos.(a...))
            @test ((args...)->cos(args[1])).(x) == cos.(x) == ((y,args...)->cos(y)).(x)
        end
    end

    @testset "Fused in-place assignment" begin
        x = QuasiVector(1:4,0:0.5:1.5); y = x
        y .= QuasiVector(2:5,0:0.5:1.5)
        @test y === x == QuasiVector(2:5,0:0.5:1.5)
        y .= factorial.(x)
        @test y === x ==  QuasiVector([2,6,24,120],0:0.5:1.5)
        y .= 7
        @test y === x ==  QuasiVector([7,7,7,7],0:0.5:1.5)
        y .= factorial.(3)
        @test y === x ==  QuasiVector([6,6,6,6],0:0.5:1.5)
        f17510() = 9
        y .= f17510.()
        @test y === x ==  QuasiVector([9,9,9,9],0:0.5:1.5)
        y .-= 1
        @test y === x ==  QuasiVector([8,8,8,8],0:0.5:1.5)
        y .-= QuasiVector(1:4,0:0.5:1.5)   # @. should convert to .-=
        @test y === x ==  QuasiVector([7,6,5,4],0:0.5:1.5)
        view(x,0:0.5:0.5) .= 1
        @test y === x == QuasiVector([1,1,5,4],0:0.5:1.5)
        view(x,:) .= 0         # use .= to make sure @. works with dotted assignment
        @test y === x == QuasiVector([0,0,0,0],0:0.5:1.5)
    end

    @testset "BroadcastQuasiArray" begin
        a = QuasiVector(randn(6), 0:0.5:2.5)
        b = BroadcastQuasiArray(exp, a)
        @test b[0.5] == exp(a[0.5])
        @test b[0.5:0.5:1.0] == b[[0.5,1.0]] == exp.(a[0.5:0.5:1.0])
        @test b == BroadcastQuasiVector(exp, a) == BroadcastQuasiVector{Float64}(exp, a) == BroadcastQuasiArray{Float64}(exp, a)

        A = QuasiArray(randn(6,6), (0:0.5:2.5,0:0.5:2.5))
        B = BroadcastQuasiArray(exp, A)
        @test B[0.5,1.0] == exp(A[0.5,1.0])
        @test B == BroadcastQuasiMatrix(exp, A) == BroadcastQuasiMatrix{Float64}(exp, A) == BroadcastQuasiArray{Float64}(exp, A)

        @test axes(exp.(A)) == axes(B)
        @test QuasiMatrix(B) == exp.(A)
        @test copy(B) ≡ B

        C = BroadcastQuasiArray(+, A, 2)
        @test C == A .+ 2
        D = BroadcastQuasiArray(+, A, C)
        @test D == A + C

        @test sum(B) ≈ sum(exp, A)
        @test sum(C) ≈ sum(A .+ 2)

        @testset "index bugs (ContinuumArrays #53)" begin
            x = Inclusion([0.0,2.0])
            @test BroadcastQuasiArray(exp,x)[QuasiCartesianIndex(2.0)] == exp(2.0)
        end

        @testset "subview" begin
            v = view(b,0.0:0.5:1.0)
            c = BroadcastArray(v)
            @test c == b[0.0:0.5:1.0]

            w = view(b,Inclusion(0.0:0.5:1.0))
            d = BroadcastQuasiArray(w)
            @test d == b[Inclusion(0.0:0.5:1.0)]
        end

        @testset "view bug" begin
            b = BroadcastQuasiVector(exp, Inclusion(0:0.5:1))
            v = view(b, 0.5)

            @test v[] == b[0.5]
        end

        @testset "show" begin
            x = Inclusion(0:0.5:1)
            @test stringmime("text/plain", exp.(x)) == "exp.(Inclusion(0.0:0.5:1.0))"
            @test stringmime("text/plain", x.^2) == "(Inclusion(0.0:0.5:1.0)) .^ 2"
            @test stringmime("text/plain", x .- x) == "Inclusion(0.0:0.5:1.0) .- Inclusion(0.0:0.5:1.0)"
            @test stringmime("text/plain", x .+ x) == "Inclusion(0.0:0.5:1.0) .+ Inclusion(0.0:0.5:1.0)"
            @test stringmime("text/plain", (-).(x)) == "(-).(Inclusion(0.0:0.5:1.0))"
            @test stringmime("text/plain", x ./ x) == "Inclusion(0.0:0.5:1.0) ./ Inclusion(0.0:0.5:1.0)"
            @test stringmime("text/plain", x .\ x) == "Inclusion(0.0:0.5:1.0) .\\ Inclusion(0.0:0.5:1.0)"
        end

        @testset "x .* (1:10)'" begin
            x = Inclusion([0.1,0.2])
            B = x .* (1:10)'
            @test B[0.1,2] == 0.2
            @test B[0.1,1:3] == 0.1*(1:3)
            @test B[[0.1,0.2],2] == 2*[0.1,0.2]
            @test B[[0.1,0.2],2:3] == [0.1,0.2] * (2:3)'
        end
    end

    @testset "SubQuasi Broadcast" begin
        a = QuasiVector(randn(6), 0:0.5:2.5)
        v = view(a,0:0.5:1)
        @test v isa SubArray
        @test BroadcastStyle(typeof(v)) isa DefaultArrayStyle{1}
        @test exp.(v) == exp.(Array(v))
        w = view(a,Inclusion(0:0.5:1))
        @test w isa SubQuasiArray
        @test BroadcastStyle(typeof(w)) isa DefaultQuasiArrayStyle{1}
        @test exp.(w) == exp.(a)[Inclusion(0:0.5:1)]

        A = QuasiArray(randn(6,6), (0:0.5:2.5, 0:0.5:2.5))
        v = view(A,:,0:0.5:1)
        @test BroadcastStyle(typeof(v)) isa DefaultQuasiArrayStyle{2}
        v = view(A,0.5,0:0.5:1)
        @test BroadcastStyle(typeof(v)) isa DefaultArrayStyle{1}
        @test v .+ (1:3) == parent(A)[2,1:3] .+ (1:3)
        v = view(A,0:0.5:1,0.5)
        @test BroadcastStyle(typeof(v)) isa DefaultArrayStyle{1}
        @test v .+ (1:3) == parent(A)[1:3,2] .+ (1:3)
        v = view(A,0.5,0.5)
        @test BroadcastStyle(typeof(v)) isa DefaultArrayStyle{0}
    end

    @testset "flatten Broadcast" begin
        a = QuasiVector(randn(6), 0:0.5:2.5)
        B = QuasiMatrix(randn(6,6), (0:0.5:2.5, 0:0.5:2.5))
        aB = BroadcastQuasiArray(*, a, ApplyQuasiArray(*, B, B))
        @test QuasiArrays.flatten(aB).args == (a .* B, B)
    end

    @testset "Broadcast add" begin
        A = QuasiMatrix(randn(6,6), (0:0.5:2.5, 0:0.5:2.5))
        a = QuasiVector(randn(6), 0:0.5:2.5)
        ã = QuasiMatrix(randn(6,1), (axes(a,1),Base.OneTo(1)))
        b = QuasiVector(randn(6), 0:0.5:2.5)
        c = 2.3

        for op in (+, -)
            @testset "$op" begin
                @testset "vec .$op mat" begin
                    B = BroadcastQuasiArray(op, a, A)
                    @test (B * b) isa QuasiVector
                    @test (B * A) isa QuasiMatrix
                    @test (A * B) isa QuasiMatrix
                    @test B * B isa QuasiMatrix
                    @test B * b ≈ op.(a, A) * b
                    @test B * A ≈ op.(a, A) * A
                    @test A * B ≈ A * op.(a, A)
                    @test B * B ≈ op.(a, A) * op.(a, A)
                end
                @testset "mat .$op vec" begin
                    B = BroadcastQuasiArray(op, A, a)
                    @test B * b ≈ op.(A, a) * b
                    @test B * A ≈ op.(A, a) * A
                    @test A * B ≈ A * op.(A, a)
                end
                @testset "mat .$op mat" begin
                    B = BroadcastQuasiArray(op, A, 2A)
                    @test B * b ≈ op.(A, 2A) * b
                    @test B * A ≈ op.(A, 2A) * A
                    @test A * B ≈ A * op.(A, 2A)
                end
                @testset "vecmat .$op mat" begin
                    B = BroadcastQuasiArray(op, ã, A)
                    @test B * b ≈ op.(ã, A) * b
                    @test B * A ≈ op.(ã, A) * A
                    @test A * B ≈ A * op.(ã, A)
                end
                @testset "mat .$op vecmat" begin
                    B = BroadcastQuasiArray(op, A, ã)
                    @test B * b ≈ op.(A, ã) * b
                    @test B * A ≈ op.(A, ã) * A
                    @test A * B ≈ A * op.(A, ã)
                end
                @testset "rowvec .$op mat" begin
                    B = BroadcastQuasiArray(op, a', A)
                    @test B * b ≈ op.(a', A) * b
                    @test B * A ≈ op.(a', A) * A
                    @test A * B ≈ A * op.(a', A)
                end
                @testset "constvec .$op mat" begin
                    B = BroadcastQuasiArray(op, c, A)
                    @test B * b ≈ op.(c, A) * b
                    @test B * A ≈ op.(c, A) * A
                    @test A * B ≈ A * op.(c, A)
                end
                @testset "constvec .$op vec" begin
                    B = BroadcastQuasiArray(op, c, b)
                    @test B * permutedims(b) ≈ op.(c, b) * permutedims(b)
                end
            end
        end

        @testset "Mixed" begin
            B = BroadcastQuasiArray(+, A, 2A)
            C = BroadcastQuasiArray(-, A, 2A)
            @test B*C ≈ 3A * (-A)
            @test C*B ≈ (-A) * 3A
        end
    end

    @testset "Static Broadcasting" begin
        a = QuasiArray(randn(3,2), [0.1,0.2,0.3], [0.2,0.3])
        v = view(a, SVector(0.1,0.2),0.2)
        @test Base.BroadcastStyle(typeof(v)) isa DefaultArrayStyle{1}

        v = view(a, BroadcastArray(+, [0.1,0.2]), 0.2)
        @test Base.BroadcastStyle(typeof(v)) isa LazyArrays.LazyArrayStyle{1}
    end

    @testset "empty broadcast bug" begin
        A = ones(Int, 4)
        A[ones(Int)] .+= 1
        @test A == [2,1,1,1]
    end

    @testset "adjoint arguments" begin
        a = QuasiVector(randn(6), 0:0.5:2.5)
        @test LazyArrays.MemoryLayout(BroadcastQuasiArray(*, 2, a)') isa LazyArrays.BroadcastLayout
        @test LazyArrays.MemoryLayout(transpose(BroadcastQuasiArray(*, 2, a))) isa LazyArrays.BroadcastLayout
        @test LazyArrays.arguments(BroadcastQuasiArray(*, 2, a)') == (2, a')
        @test LazyArrays.arguments(transpose(BroadcastQuasiArray(*, 2, a))) == (2, transpose(a))
    end
end
