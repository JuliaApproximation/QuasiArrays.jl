using QuasiArrays, IntervalSets, Test
import QuasiArrays: ApplyQuasiArray

@testset "Continuous" begin
    @testset "Inclusion" begin
        # note behaviour changes with ContinuumArrays
        x = Inclusion(0..1)
        @test eltype(x) == Int
        @test axes(x) ≡ (x,)
        @test_throws InexactError x[0.1]
        @test x[1] ≡ 1
        @test_throws BoundsError x[2]
        @test Base.unsafe_getindex(x,2) ≡ 2

        for x in (Inclusion{Float64}(0..1),Inclusion(0.0..1))
            @test Inclusion(x) ≡ Inclusion{Float64}(x) ≡ convert(Inclusion,x) ≡ 
                    convert(Inclusion{Float64},x) ≡ convert(AbstractQuasiArray,x) ≡
                    convert(AbstractQuasiArray{Float64},x) ≡ convert(AbstractQuasiVector{Float64},x) ≡ 
                    convert(AbstractQuasiVector,x) ≡ x
            @test axes(x) ≡ (x,)
            @test x[0.1] ≡ 0.1
            @test x[1] ≡ 1.0
            @test_throws BoundsError x[2]
            @test Base.unsafe_getindex(x,2) ≡ 2.0
            @test x[BigFloat(π)/4] == π/4
        end

        @test Inclusion{Float64}(0..1) == Inclusion(0.0..1)
        @test Inclusion{Float64}(0..1) !== Inclusion(0.0..1)

        @test findfirst(isequal(0.1), Inclusion(0.0..1)) ≡ findlast(isequal(0.1), Inclusion(0..1)) ≡ 0.1
        @test findfirst(isequal(2.3), Inclusion(0.0..1)) ≡ findlast(isequal(2.3), Inclusion(0..1)) ≡ nothing
        @test findall(isequal(0.1), Inclusion(0.0..1)) == [0.1]
        @test findall(isequal(2.3), Inclusion(0.0..1)) == Float64[]
    end

    @testset "QuasiDiagonal" begin
        x = Inclusion(0.0..1)
        D = QuasiDiagonal(x)
        @test D[0.1,0.2] ≡ 0.0
        @test D[0.1,0.1] ≡ 0.1
        @test D[1,1] ≡ 1.0
        @test_throws BoundsError D[1.1,1.2]
    end
end