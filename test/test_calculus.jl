using QuasiArrays, IntervalSets, Test

@testset "Calculus" begin
    @testset "sum" begin
        A = QuasiArray(randn(2,3), (0:0.5:0.5, 1:0.5:2))
        @test sum(A) ≈ sum(A.parent)
        @test sum(A; dims=1) ≈ QuasiArray(sum(A.parent; dims=1), (1:1, 1:0.5:2))
        @test sum(A; dims=2) ≈ QuasiArray(sum(A.parent; dims=2), (0:0.5:0.5, 1:1))
    end

    @testset "* sum" begin
        b = QuasiVector(randn(3), 1:0.5:2)
        A = QuasiArray(randn(2,3), (0:0.5:0.5, 1:0.5:2))
        B = QuasiArray(randn(3,2), (1:0.5:2,0:0.5:0.5))

        @test sum(ApplyQuasiArray(*, A, b)) ≈ sum(A*b)
        @test sum(ApplyQuasiArray(*, A, B)) ≈ sum(A*B)
        @test sum(ApplyQuasiArray(*, A, B); dims=1) ≈ sum(A*B; dims=1)
        @test sum(ApplyQuasiArray(*, A, B); dims=2) ≈ sum(A*B; dims=2)
        @test sum(ApplyQuasiArray(*, A, B)) ≈ sum(A*B)

        @test sum(b) ≈ last(cumsum(b)) ≈ cumsum(b)[2]
        @test cumsum(B)[2:2,:] ≈ cumsum(B; dims=1)[2:2,:] ≈ sum(B; dims=1)
        @test cumsum(B; dims=2)[:,0.5:0.5] ≈ sum(B; dims=2)

        @test cumsum(ApplyQuasiArray(*, A, b)) ≈ cumsum(A*b)
        @test cumsum(ApplyQuasiArray(*, A, B); dims=1) ≈ cumsum(A*B; dims=1)
        @test cumsum(ApplyQuasiArray(*, A, B); dims=2) ≈ cumsum(A*B; dims=2)
    end

    @testset "Diff" begin
        x = range(0, 1; length=10_000)
        @test diff(Inclusion(x)) == ones(Inclusion(x[1:end-1]))
        @test diff(ones(Inclusion(x))) == zeros(Inclusion(x[1:end-1]))

        @test diff(ones(Inclusion(x), Inclusion(x))) == zeros(Inclusion(x[1:end-1]), Inclusion(x))
        @test diff(ones(Inclusion(x), Inclusion(x)); dims=2) == zeros(Inclusion(x), Inclusion(x[1:end-1]))

        b = QuasiVector(exp.(x), x)

        @test diff(b) ≈ b[Inclusion(x[1:end-1])] atol=1E-2


        A = QuasiArray(randn(3,2), (1:0.5:2,0:0.5:0.5))
        @test diff(A; dims=1)[:,0] == diff(A[:,0])
        @test diff(A; dims=2)[1,:] == diff(A[1,:])

        @testset "* diff" begin
            b = QuasiVector(randn(3), 1:0.5:2)
            A = QuasiArray(randn(2,3), (0:0.5:0.5, 1:0.5:2))
            B = QuasiArray(randn(3,2), (1:0.5:2,0:0.5:0.5))
    
            @test @inferred(diff(ApplyQuasiArray(*, A, b))) ≈ diff(A*b)
            @test @inferred(diff(ApplyQuasiArray(*, A, B))) ≈ diff(A*B)
        end
    end

    @testset "Interval" begin
        @test diff(Inclusion(0.0..1)) ≡ ones(Inclusion(0.0..1))
        @test diff(ones(Inclusion(0.0..1))) ≡ zeros(Inclusion(0.0..1))
        @test diff(ones(Inclusion(0.0..1), Base.OneTo(3))) ≡ zeros(Inclusion(0.0..1), Base.OneTo(3))
        @test diff(ones(Inclusion(0.0..1), Base.OneTo(3)); dims=2) ≡ zeros(Inclusion(0.0..1), Base.OneTo(2))
        @test diff(ones(Base.OneTo(3), Inclusion(0.0..1))) ≡ zeros(Base.OneTo(2), Inclusion(0.0..1))
        @test diff(ones(Base.OneTo(3), Inclusion(0.0..1)); dims=2) ≡ zeros(Base.OneTo(3), Inclusion(0.0..1))
    end
end