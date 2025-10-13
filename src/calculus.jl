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

_colon2one(::Colon) = 1
_colon2one(dims::Int) = dims

function cumsum_layout(LAY::ApplyLayout{typeof(*)}, V::AbstractQuasiVector, dims)
    a = arguments(LAY, V)
    apply(*, cumsum(a[1]; dims=_colon2one(dims)), tail(a)...)
end

function sum_layout(LAY::ApplyLayout{typeof(*)}, V::AbstractQuasiVector, dims)
    a = arguments(LAY, V)
    only(*(sum(a[1]; dims=_colon2one(dims)), tail(a)...))
end

function sum_layout(LAY::ApplyLayout{typeof(*)}, V::AbstractQuasiMatrix, dims)
    a = arguments(LAY, V)
    only(*(sum(a[1]; dims=_colon2one(dims)), front(tail(a))..., sum(a[end]; dims=2)))
end

sum_layout(::MemoryLayout, A, dims) = sum_size(size(A), A, dims)
sum_size(::NTuple{N,Integer}, A, dims) where N = _sum(identity, A, dims)
cumsum_layout(::MemoryLayout, A, dims) = cumsum_size(size(A), A, dims)
cumsum_size(::NTuple{N,Integer}, A, dims) where N = error("Not implemented")


####
# diff
####

@inline diff(a::AbstractQuasiArray, order...; dims::Integer=1) = diff_layout(MemoryLayout(a), a, order...; dims)
function diff_layout(LAY::ApplyLayout{typeof(*)}, V::AbstractQuasiVecOrMat, order...; dims=1)
    a = arguments(LAY, V)
    dims == 1 || throw(ArgumentError("cannot differentiate a vector along dimension $dims"))
    *(diff(a[1], order...), tail(a)...)
end

diff_layout(::MemoryLayout, A, order...; dims...) = diff_size(size(A), A, order...; dims...)
diff_size(sz, a; dims...) = error("diff not implemented for $(typeof(a))")
function diff_size(sz, a, order; dims...)
    order < 0 && throw(ArgumentError("order must be non-negative"))
    order == 0 && return a
    isone(order) ? diff(a) : diff(diff(a), order-1)
end

diff(x::Inclusion; dims::Integer=1) = ones(eltype(x), diffaxes(x))
diff(x::Inclusion, order::Int; dims::Integer=1) = fill(ifelse(isone(order), one(eltype(x)), zero(eltype(x))), diffaxes(x,order))
diff(c::AbstractQuasiFill{<:Any,1}, order...; dims::Integer=1) =  zeros(eltype(c), diffaxes(axes(c,1),order...))
function diff(c::AbstractQuasiFill{<:Any,2}, order...; dims::Integer=1)
    a,b = axes(c)
    if dims == 1
        zeros(eltype(c), diffaxes(a, order...), b)
    else
        zeros(eltype(c), a, diffaxes(b, order...))
    end
end


diffaxes(a::Inclusion{<:Any,<:AbstractVector}, order=1) = Inclusion(a.domain[1:end-order])
diffaxes(a::OneTo, order=1) = oneto(length(a)-order)
diffaxes(a, order...) = a # default is differentiation does not change axes

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



#####
# norm
#####

if VERSION ≥ v"1.12-"
    # avoid iterate call in v1.12
    LinearAlgebra.norm_recursive_check(::AbstractQuasiArray) = nothing
end