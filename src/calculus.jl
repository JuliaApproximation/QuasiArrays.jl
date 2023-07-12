###
# sum/cumsum
###

# support overloading sum by MemoryLayout
_sum(V::AbstractQuasiArray, dims) = sum_layout(MemoryLayout(V), V, dims)
_sum(V::AbstractQuasiArray, ::Colon) = sum_layout(MemoryLayout(V), V, :)

_cumsum(A, dims) = cumsum_layout(MemoryLayout(A), A, dims)
cumsum(A::AbstractQuasiArray; dims::Integer) = _cumsum(A, dims)
cumsum(x::AbstractQuasiVector) = cumsum(x, dims=1)

# sum is equivalent to hitting by ones(n) on the left or right

cumsum_layout(::QuasiArrayLayout, A, ::Colon) = QuasiArray(cumsum(parent(A)), axes(A))
cumsum_layout(::QuasiArrayLayout, A, d::Int) = QuasiArray(cumsum(parent(A),dims=d), axes(A))

for Sum in (:sum, :cumsum)
    Sum_Lay = Symbol(Sum, "_layout")
    @eval function $Sum_Lay(LAY::ApplyLayout{typeof(*)}, V::AbstractQuasiMatrix, d::Int)
        a = arguments(LAY, V)
        if d == 1
            *($Sum(first(a); dims=1), tail(a)...)
        else
            @assert d == 2
            *(Base.front(a)..., $Sum(last(a); dims=2))
        end
    end
end

function cumsum_layout(LAY::ApplyLayout{typeof(*)}, V::AbstractQuasiVector, dims)
    a = arguments(LAY, V)
    apply(*, cumsum(a[1]; dims=dims), tail(a)...)
end

function sum_layout(LAY::ApplyLayout{typeof(*)}, V::AbstractQuasiVector, ::Colon)
    a = arguments(LAY, V)
    first(apply(*, sum(a[1]; dims=1), tail(a)...))
end

sum_layout(::MemoryLayout, A, dims) = sum_size(size(A), A, dims)
sum_size(::NTuple{N,Int}, A, dims) where N = _sum(identity, A, dims)
cumsum_layout(::MemoryLayout, A, dims) = cumsum_size(size(A), A, dims)
cumsum_size(::NTuple{N,Int}, A, dims) where N = error("Not implemented")


####
# diff
####

@inline diff(a::AbstractQuasiArray; dims::Integer=1) = diff_layout(MemoryLayout(a), a, dims)

function diff_layout(::SubBasisLayout, Vm, dims::Integer)
    dims == 1 || error("not implemented")
    diff(parent(Vm); dims=dims)[:,parentindices(Vm)[2]]
end

function diff_layout(::MappedBasisLayouts, V, dims)
    kr = basismap(V)
    @assert kr isa AbstractAffineQuasiVector
    view(diff(demap(V); dims=dims)*kr.A, kr, :)
end

diff_layout(::ExpansionLayout, A, dims...) = diff_layout(ApplyLayout{typeof(*)}(), A, dims...)
function diff_layout(LAY::ApplyLayout{typeof(*)}, V::AbstractQuasiVector, dims...)
    a = arguments(LAY, V)
    *(diff(a[1]), tail(a)...)
end

diff_layout(::MemoryLayout, A, dims...) = diff_size(size(A), A, dims...)
diff_size(sz, a, dims...) = error("diff not implemented for $(typeof(a))")

diff(x::Inclusion; dims::Integer=1) = ones(eltype(x), x)
diff(c::AbstractQuasiFill{<:Any,1}; dims::Integer=1) =  zeros(eltype(c), axes(c,1))
