using Plots

function struct_isequal(a,b)
    fa = fieldnames(typeof(a))
    fb = fieldnames(typeof(b))
    fa != fb && return false
    for f in fa
        !struct_isequal(getfield(a,f),getfield(b,f)) && return false
    end
    return true
end

@testset "plots" begin
    v = QuasiVector(rand(5), rand(5))
    a = plot(v)
    b = plot(axes(v,1),parent(v))
    @test struct_isequal(a,b)
end
