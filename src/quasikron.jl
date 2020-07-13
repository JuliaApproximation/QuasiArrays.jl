"""
    InclusionKron(A, B, C...)

is used to represent Kronecker products of axes
"""

struct InclusionKron{T, Args<:Tuple} <: AbstractQuasiVector{T}
    args::Args
end

InclusionKron{T}(a...) where T = InclusionKron{T,typeof(a)}(a)
InclusionKron(a...) = InclusionKron{Tuple{map(eltype,a)...}}(a...)

copy(d::InclusionKron) = d

==(A::InclusionKron, B::InclusionKron) = A.args == B.args
axes(S::InclusionKron) = (S,)
unsafe_indices(S::InclusionKron) = (S,)
axes1(S::InclusionKron) = S

first(S::InclusionKron) = map(first,S.args)
last(S::InclusionKron) = map(last,S.args)
size(S::InclusionKron) = (prod(size.(S.args,1)),)
length(S::InclusionKron) = prod(map(length,S.args))
cardinality(S::InclusionKron) = prod(map(cardinality,S.args))
getindex(S::InclusionKron{T}, i::T) where T =
    (@_inline_meta; @boundscheck checkbounds(S, i); convert(T,i))
getindex(S::InclusionKron{T}, i::AbstractVector{T}) where T =
    (@_inline_meta; @boundscheck checkbounds(S, i); convert(AbstractVector{T},i))
getindex(S::InclusionKron, i::InclusionKron) =
    (@_inline_meta; @boundscheck checkbounds(S, i); copy(S))
getindex(S::InclusionKron, ::Colon) = copy(S)
show(io::IO, r::InclusionKron) = print(io, "InclusionKron(", r.args, ")")
# iterate(S::InclusionKron, s...) = iterate(S.domain, s...)

in(x, S::InclusionKron) = all(in.(x, S.args))

checkindex(::Type{Bool}, inds::InclusionKron, i) = i ∈ inds
checkindex(::Type{Bool}, inds::InclusionKron, ::Colon) = true
checkindex(::Type{Bool}, inds::InclusionKron, ::InclusionKron) = true

checkindex(::Type{Bool}, inds::InclusionKron{T}, I::AbstractArray{T}) where T = 
    __checkindex(Bool, inds, I)
checkindex(::Type{Bool}, inds::InclusionKron{T}, I::AbstractArray{T}) where T<:AbstractArray = 
    __checkindex(Bool, inds, I)
checkindex(::Type{Bool}, inds::InclusionKron{T}, I::AbstractArray{<:AbstractArray}) where T<:AbstractArray = 
    __checkindex(Bool, inds, convert(AbstractArray{T}, I))


"""
    QuasiKron(A, B, C...)

is a lazy representation of a Kronecker product of the quasi-arrays A, B, C.... Unlike 
standard kronecker product, we also kronecker the axes, using tuples.
"""

struct QuasiKron{T, N, Args<:Tuple} <: AbstractQuasiArray{T, N}
    args::Args
end

QuasiKron{T,N}(a...) where {T,N} = QuasiKron{T,N,typeof(a)}(a)
QuasiKron{T}(a::AbstractQuasiOrVector...) where T = QuasiKron{mapreduce(eltype,promote_type,a),1}(a...)
QuasiKron{T}(a::AbstractQuasiOrMatrix...) where T = QuasiKron{mapreduce(eltype,promote_type,a),2}(a...)
QuasiKron(a...) = QuasiKron{mapreduce(eltype,promote_type,a)}(a...)


__kronaxes(a::Inclusion...) = InclusionKron(a...)
_kronaxes(::Tuple{}...) = ()
_kronaxes(a::Tuple...) = (__kronaxes(map(first,a)...), _kronaxes(map(tail,a)...)...)
axes(K::QuasiKron) = _kronaxes(map(axes,K.args)...)

function getindex(K::QuasiKron{<:Any,1}, i::Tuple)
    @boundscheck checkbounds(K, i)
    prod(getindex.(K.args,i))
end