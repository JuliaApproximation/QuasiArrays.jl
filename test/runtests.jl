using QuasiArrays, Test
import QuasiArrays: Inclusion


A = QuasiArray(rand(5,4,3), (range(0;stop=1,length=5), Base.OneTo(4), [2,3,6]))

@testset "QuasiArray basics" begin
    @test_throws ArgumentError QuasiArray(rand(5,4,3), Inclusion.((range(0;stop=1,length=6), Base.OneTo(4), [1,3,4])))
    @test axes(A) == Inclusion.((range(0;stop=1,length=5), Base.OneTo(4), [2,3,6]))
    @test A[0.25,2,6] == A.parent[2,2,3]
    @test_throws BoundsError A[0.1,2,6]
end
