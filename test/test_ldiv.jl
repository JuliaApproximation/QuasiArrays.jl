using QuasiArrays, LinearAlgebra, LazyArrays, Test

@testset "ldiv" begin
    A = QuasiArray(randn(5,5), (0:0.5:2, 1:0.5:3))
    b = QuasiArray(randn(5), (0:0.5:2,))
    L = Ldiv(A, b)
    @test axes(L) == (axes(A,2),)
    @test similar(L) isa QuasiVector
    @test A\b == copy(L) == QuasiArray(parent(A)\parent(b), (1:0.5:3,))

    B = QuasiArray(randn(5,3), (0:0.5:2,Base.OneTo(3)))
    L = Ldiv(A, B)
    @test axes(L) == (axes(A,2),axes(B,2))
    @test similar(L) isa QuasiMatrix
    @test A\B == copy(L) == QuasiArray(parent(A)\parent(B), (1:0.5:3,Base.OneTo(3)))


    A = QuasiArray(randn(5,5), (Base.OneTo(5), 0:0.5:2))
    v = randn(5)
    @test A\v == QuasiArray(parent(A)\v, (0:0.5:2,))
    @test A' \ b == parent(A)' \ parent(b)
end 