# This file is a part of Julia. License is MIT: https://julialang.org/license


IndexStyle(A::AbstractQuasiArray) = IndexStyle(typeof(A))
IndexStyle(::Type{<:AbstractQuasiArray}) = IndexCartesian()

IndexStyle(A::AbstractQuasiArray, B::AbstractQuasiArray) = IndexStyle(IndexStyle(A), IndexStyle(B))
IndexStyle(A::AbstractArray, B::AbstractQuasiArray) = IndexStyle(IndexStyle(A), IndexStyle(B))
IndexStyle(A::AbstractQuasiArray, B::AbstractArray) = IndexStyle(IndexStyle(A), IndexStyle(B))
IndexStyle(A::AbstractQuasiArray, B::AbstractQuasiArray...) = IndexStyle(IndexStyle(A), IndexStyle(B...))
IndexStyle(A::AbstractQuasiArray, B::AbstractArray...) = IndexStyle(IndexStyle(A), IndexStyle(B...))


function promote_shape(a::AbstractQuasiArray, b::AbstractQuasiArray)
    promote_shape(axes(a), axes(b))
end

const QuasiIndices{N} = NTuple{N,Union{AbstractQuasiVector{<:Number},AbstractVector{<:Number}}}
function promote_shape(a::QuasiIndices, b::QuasiIndices)
    if length(a) < length(b)
        return promote_shape(b, a)
    end
    for i=1:length(b)
        if a[i] != b[i]
            throw(DimensionMismatch("dimensions must match"))
        end
    end
    for i=length(b)+1:length(a)
        if a[i] != 1:1
            throw(DimensionMismatch("dimensions must match"))
        end
    end
    return a
end

# check for valid sizes in A[I...] = X where X <: AbstractQuasiArray
# we want to allow dimensions that are equal up to permutation, but only
# for permutations that leave array elements in the same linear order.
# those are the permutations that preserve the order of the non-singleton
# dimensions.
function setindex_shape_check(X::AbstractQuasiArray, I::Integer...)
    li = ndims(X)
    lj = length(I)
    i = j = 1
    while true
        ii = length(axes(X,i))
        jj = I[j]
        if i == li || j == lj
            while i < li
                i += 1
                ii *= length(axes(X,i))
            end
            while j < lj
                j += 1
                jj *= I[j]
            end
            if ii != jj
                throw_setindex_mismatch(X, I)
            end
            return
        end
        if ii == jj
            i += 1
            j += 1
        elseif ii == 1
            i += 1
        elseif jj == 1
            j += 1
        else
            throw_setindex_mismatch(X, I)
        end
    end
end

setindex_shape_check(X::AbstractQuasiArray) =
    (length(X)==1 || throw_setindex_mismatch(X,()))

setindex_shape_check(X::AbstractQuasiArray, i::Integer) =
    (length(X)==i || throw_setindex_mismatch(X, (i,)))

setindex_shape_check(X::AbstractQuasiArray{<:Any,1}, i::Integer) =
    (length(X)==i || throw_setindex_mismatch(X, (i,)))

setindex_shape_check(X::AbstractQuasiArray{<:Any,1}, i::Integer, j::Integer) =
    (length(X)==i*j || throw_setindex_mismatch(X, (i,j)))

function setindex_shape_check(X::AbstractQuasiArray{<:Any,2}, i::Integer, j::Integer)
    if length(X) != i*j
        throw_setindex_mismatch(X, (i,j))
    end
    sx1 = length(axes(X,1))
    if !(i == 1 || i == sx1 || sx1 == 1)
        throw_setindex_mismatch(X, (i,j))
    end
end


to_index(I::AbstractQuasiArray{Bool}) = LogicalIndex(I)
to_index(I::AbstractQuasiArray) = I
to_index(I::AbstractQuasiArray{<:Union{AbstractArray, Colon}}) =
    throw(ArgumentError("invalid index: $I of type $(typeof(I))"))

to_quasi_index(i::Number) = i
to_quasi_index(i) = Base.to_index(i)
to_index(A::AbstractQuasiArray, i) = to_quasi_index(i)

LinearIndices(A::AbstractQuasiArray) = LinearIndices(axes(A))



"""
   Inclusion(domain)

Represents the inclusion operator of a domain (that is, a type that overrides in)
as an AbstractQuasiVector. That is, if `v = Inclusion(domain)`, then
`v[x] == x` if `x in domain`, otherwise it throws a `DomainError`.

Inclusions are useful for turning domains into axes. They also serve the same
role as `Slice` does for offset arrays.
"""
struct Inclusion{T,AX} <: AbstractQuasiVector{T}
    domain::AX
end
Inclusion{T}(domain) where T = Inclusion{T,typeof(domain)}(domain)
Inclusion{T}(S::Inclusion) where T = Inclusion{T}(S.domain)
Inclusion{T}(S::Slice) where T = Inclusion{T}(S.indices)
Inclusion(domain) = Inclusion{eltype(domain)}(domain)
Inclusion(S::Inclusion) = S
Inclusion(S::Slice) = Inclusion(S.indices)

convert(::Type{Inclusion}, d::Inclusion) = d
convert(::Type{Inclusion{T}}, d::Inclusion) where T = Inclusion{T}(d)
convert(::Type{AbstractVector}, d::Inclusion{<:Any,<:AbstractVector}) =
    convert(AbstractVector, d.domain)
convert(::Type{AbstractArray}, d::Inclusion{<:Any,<:AbstractVector}) =
    convert(AbstractArray, d.domain)
Vector(d::Inclusion{<:Any,<:AbstractVector}) = Vector(d.domain)
Array(d::Inclusion{<:Any,<:AbstractVector}) = Array(d.domain)

copy(d::Inclusion) = d

==(A::Inclusion, B::Inclusion) = A.domain == B.domain
domain(A::Inclusion) = A.domain
domain(A::AbstractUnitRange) = A
axes(S::Inclusion) = (S,)
unsafe_indices(S::Inclusion) = (S,)
axes1(S::Inclusion) = S
axes(S::Inclusion{<:Any,<:OneTo}) = (S.domain,)
unsafe_indices(S::Inclusion{<:Any,<:OneTo}) = (S.domain,)
axes1(S::Inclusion{<:Any,<:OneTo}) = S.domain

first(S::Inclusion) = first(S.domain)
last(S::Inclusion) = last(S.domain)
size(S::Inclusion) = (cardinality(S.domain),)
length(S::Inclusion) = cardinality(S.domain)
unsafe_length(S::Inclusion) = cardinality(S.domain)
cardinality(S::Inclusion) = cardinality(S.domain)
getindex(S::Inclusion{T}, i::Number) where T =
    (@_inline_meta; @boundscheck checkbounds(S, i); convert(T,i))
getindex(S::Inclusion{T}, i::AbstractVector{<:Number}) where T =
    (@_inline_meta; @boundscheck checkbounds(S, i); convert(AbstractVector{T},i))
getindex(S::Inclusion, i::Inclusion) =
    (@_inline_meta; @boundscheck checkbounds(S, i); copy(S))
getindex(S::Inclusion, ::Colon) = copy(S)
show(io::IO, r::Inclusion) = print(io, "Inclusion(", r.domain, ")")
iterate(S::Inclusion, s...) = iterate(S.domain, s...)

in(x, S::Inclusion) = x in S.domain

checkindex(::Type{Bool}, inds::Inclusion, i::Number) = i ∈ inds.domain
checkindex(::Type{Bool}, inds::Inclusion, ::Colon) = true
checkindex(::Type{Bool}, inds::Inclusion, ::Inclusion) = true
function checkindex(::Type{Bool}, inds::Inclusion, I::AbstractArray)
    @_inline_meta
    b = true
    for i in I
        b &= checkindex(Bool, inds, i)
    end
    b
end

function checkindex(::Type{Bool}, inds::Inclusion, r::AbstractRange)
    @_propagate_inbounds_meta
    isempty(r) | (checkindex(Bool, inds, first(r)) & checkindex(Bool, inds, last(r)))
end
checkindex(::Type{Bool}, indx::Inclusion, I::AbstractVector{Bool}) = indx == axes1(I)
checkindex(::Type{Bool}, indx::Inclusion, I::AbstractArray{Bool}) = false
