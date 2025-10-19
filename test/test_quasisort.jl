using QuasiArrays, Test

@testset "sort/find" begin
    v = QuasiVector([1,0,2,3,0], 0:0.5:2)
    @test @inferred(findall(iszero, v)) == [0.5,2]
    @test findall(isone, v) == [0]
    @test isempty(findall(==(4), v))
    @test findfirst(iszero, v) == 0.5
    @test isnothing(findfirst(==(4), v))
    @test findlast(iszero, v) == 2
    @test isnothing(findlast(==(4), v))

    w = QuasiVector([1,2,2,3,4], 0:0.5:2)
    @test @inferred(searchsorted(w, 2)) ≡ 0.5:0.5:1.0
    @test searchsorted(w, 3) ≡ 1.5:0.5:1.5
    @test isempty(searchsorted(w,0))
    @test isempty(searchsorted(w,2.5))
    @test isempty(searchsorted(w,5))
    @test searchsortedfirst(w,2) == 0.5
    @test searchsortedlast(w,2) == 1
    @test searchsortedfirst(w,0) == 0.0
    @test_broken searchsortedlast(w,0) ≤ 0
    @test_broken searchsortedfirst(w, 5) ≥ 2
    @test searchsortedlast(w, 5) == 2
end

@testset "minimum/maximum/extrema" begin
    v = QuasiVector([1,0,2,3,0], 0:0.5:2)
    @test minimum(v) == 0
    @test maximum(v) == 3
    @test extrema(v) == (0,3)
end