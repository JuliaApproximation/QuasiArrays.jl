module QuasiArraysStatsBaseExt
using QuasiArrays, StatsBase
import StatsBase: sample, AbstractRNG
import QuasiArrays: sample_layout, MemoryLayout

"""
    sample([rng], w::AbstractQuasiArray)

Sample a single random element from `axes(w,1)` weighted according to `w`.
"""
sample(w::AbstractQuasiArray) = sample_layout(MemoryLayout(w), w)
sample(rng::AbstractRNG, w::AbstractQuasiArray) = sample_layout(MemoryLayout(w), rng, w)

"""
    sample([rng], w::AbstractQuasiArray, n::Integer)

Sample a n random elements from `axes(w,1)` weighted according to `w`.
"""
sample(w::AbstractQuasiArray, n::Integer) = sample_layout(MemoryLayout(w), w, n)
sample(rng::AbstractRNG, w::AbstractQuasiArray, n::Integer) = sample_layout(MemoryLayout(w), rng, w, n)

function sample_layout(_, f::AbstractQuasiVector)
    g = cumsum(f)
    searchsortedfirst(g/last(g), rand())
end

function sample_layout(_, rng::AbstractRNG, f::AbstractQuasiVector)
    g = cumsum(f)
    searchsortedfirst(g/last(g), rand())
end

function sample_layout(_, f::AbstractQuasiVector, n::Integer)
    g = cumsum(f)
    searchsortedfirst.(Ref(g/last(g)), rand(n))
end

function sample_layout(_, rng::AbstractRNG, f::AbstractQuasiVector, n::Integer)
    g = cumsum(f)
    searchsortedfirst.(Ref(g/last(g)), rand(rng, n))
end

function sample_layout(_, f::AbstractQuasiMatrix, n...)
    @assert size(f,2) == 1 # TODO generalise 
    sample(f[:,1], n...)
end

function sample_layout(_, rng::AbstractRNG, f::AbstractQuasiMatrix, n...)
    @assert size(f,2) == 1 # TODO generalise 
    sample(rng, f[:,1], n...)
end

end