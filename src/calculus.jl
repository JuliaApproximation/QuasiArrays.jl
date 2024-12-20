###
# sum/cumsum
###

# support overloading sum by MemoryLayout
_sum(V::AbstractQuasiArray, dims) = sum_layout(MemoryLayout(V), V, dims)
_sum(V::AbstractQuasiArray, ::Colon) = sum_layout(MemoryLayout(V), V, :)

_cumsum(A, dims) = cumsum_layout(MemoryLayout(A), A, dims)
cumsum(A::AbstractQuasiArray; dims::Integer=1) = _cumsum(A, dims)

# sum is equivalent to hitting by ones(n) on the left or right

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
    only(*(sum(a[1]; dims=1), tail(a)...))
end

function sum_layout(LAY::ApplyLayout{typeof(*)}, V::AbstractQuasiMatrix, ::Colon)
    a = arguments(LAY, V)
    only(*(sum(a[1]; dims=1), front(tail(a))..., sum(a[end]; dims=2)))
end

sum_layout(::MemoryLayout, A, dims) = sum_size(size(A), A, dims)
sum_size(::NTuple{N,Integer}, A, dims) where N = _sum(identity, A, dims)
cumsum_layout(::MemoryLayout, A, dims) = cumsum_size(size(A), A, dims)
cumsum_size(::NTuple{N,Integer}, A, dims) where N = error("Not implemented")


####
# diff
####

@inline diff(a::AbstractQuasiArray; dims::Integer=1) = diff_layout(MemoryLayout(a), a, dims)
function diff_layout(LAY::ApplyLayout{typeof(*)}, V::AbstractQuasiVector, dims...)
    a = arguments(LAY, V)
    *(diff(a[1]), tail(a)...)
end

function diff_layout(LAY::ApplyLayout{typeof(*)}, V::AbstractQuasiMatrix, dims=1)
    a = arguments(LAY, V)
    @assert dims == 1 #for type stability, for now
    # if dims == 1
        *(diff(a[1]), tail(a)...)
    # else
    #     *(front(a)..., diff(a[end]; dims=dims))
    # end
end

diff_layout(::MemoryLayout, A, dims...) = diff_size(size(A), A, dims...)
diff_size(sz, a, dims...) = error("diff not implemented for $(typeof(a))")

diff(x::Inclusion; dims::Integer=1) = ones(eltype(x), diffaxes(x))
diff(c::AbstractQuasiFill{<:Any,1}; dims::Integer=1) =  zeros(eltype(c), diffaxes(axes(c,1)))
function diff(c::AbstractQuasiFill{<:Any,2}; dims::Integer=1)
    a,b = axes(c)
    if dims == 1
        zeros(eltype(c), diffaxes(a), b)
    else
        zeros(eltype(c), a, diffaxes(b))
    end
end


diffaxes(a::Inclusion{<:Any,<:AbstractVector}) = Inclusion(a.domain[1:end-1])
diffaxes(a::OneTo) = oneto(length(a)-1)
diffaxes(a) = a # default is differentiation does not change axes

diff(b::QuasiVector; dims::Integer=1) = QuasiVector(diff(b.parent) ./ diff(b.axes[1]), (diffaxes(axes(b,1)),))
function diff(A::QuasiMatrix; dims::Integer=1)
    D = diff(A.parent; dims=dims)
    a,b = axes(A)
    if dims ==  1
        QuasiMatrix(D ./ diff(a.domain), (diffaxes(a), b))
    else
        QuasiMatrix(D ./ permutedims(diff(b.domain)), (a, diffaxes(b)))
    end
end