# This file is a part of Julia. License is MIT: https://julialang.org/license


IndexStyle(A::AbstractQuasiArray) = IndexStyle(typeof(A))
IndexStyle(::Type{<:AbstractQuasiArray}) = IndexCartesian()

IndexStyle(A::AbstractQuasiArray, B::AbstractQuasiArray) = IndexStyle(IndexStyle(A), IndexStyle(B))
IndexStyle(A::AbstractArray, B::AbstractQuasiArray) = IndexStyle(IndexStyle(A), IndexStyle(B))
IndexStyle(A::AbstractQuasiArray, B::AbstractArray) = IndexStyle(IndexStyle(A), IndexStyle(B))
IndexStyle(A::AbstractQuasiArray, B::AbstractQuasiArray...) = IndexStyle(IndexStyle(A), IndexStyle(B...))
IndexStyle(A::AbstractQuasiArray, B::AbstractArray...) = IndexStyle(IndexStyle(A), IndexStyle(B...))


promote_shape(a::AbstractQuasiArray, b::AbstractQuasiArray) = promote_shape(axes(a), axes(b))
function promote_shape(a::Tuple, b::Tuple)
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

to_quasi_index(::Type{IND}, i) where IND = convert(IND, i)
to_quasi_index(::Type{IND}, I::AbstractArray) where IND<:AbstractArray = convert(IND, I)
to_quasi_index(::Type{IND}, I::AbstractQuasiArray) where IND<:AbstractQuasiArray = convert(IND, I)    
to_quasi_index(::Type{IND}, I::AbstractArray) where IND = convert(AbstractArray{IND}, I)
to_quasi_index(::Type{IND}, I::AbstractQuasiArray) where IND = convert(AbstractQuasiArray{IND}, I)    
to_quasi_index(::Type{IND}, I::AbstractArray{<:AbstractArray}) where IND<:AbstractArray = convert(AbstractArray{IND}, I)
to_quasi_index(::Type{IND}, I::AbstractQuasiArray{<:AbstractArray}) where IND<:AbstractArray = convert(AbstractQuasiArray{IND}, I)

to_quasi_index(A, IND, i) = to_quasi_index(IND,i)

to_indices(A::AbstractQuasiArray, inds, ::Tuple{}) = ()
# if inds is empty then we are indexing past indices, so array-like
to_indices(A::AbstractQuasiArray, ::Tuple{}, I::Tuple{Any,Vararg{Any}}) = 
    (@_inline_meta; (to_index(A, I[1]), to_indices(A, (), tail(I))...))
to_indices(A::AbstractQuasiArray, inds, I::Tuple{Any,Vararg{Any}}) =
    (@_inline_meta; (to_quasi_index(A, eltype(inds[1]), I[1]), to_indices(A, _maybetail(inds), tail(I))...))
@inline to_indices(A::AbstractQuasiArray, inds, I::Tuple{CartesianIndex, Vararg{Any}}) =
    to_indices(A, inds, (I[1].I..., tail(I)...))
@inline to_indices(A::AbstractQuasiArray, inds, I::Tuple{Colon, Vararg{Any}}) =
    (uncolon(inds, I), to_indices(A, _maybetail(inds), tail(I))...)



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

struct PolynomialLayout <: MemoryLayout end

MemoryLayout(::Type{<:Inclusion}) = PolynomialLayout()

convert(::Type{Inclusion}, d::Inclusion) = d
convert(::Type{Inclusion{T}}, d::Inclusion) where T = Inclusion{T}(d)
convert(::Type{AbstractVector}, d::Inclusion{<:Any,<:AbstractVector}) =
    convert(AbstractVector, d.domain)
convert(::Type{AbstractArray}, d::Inclusion{<:Any,<:AbstractVector}) =
    convert(AbstractArray, d.domain)


Vector(d::Inclusion{<:Any,<:AbstractVector}) = Vector(d.domain)
Array(d::Inclusion{<:Any,<:AbstractVector}) = Array(d.domain)
Vector{T}(d::Inclusion{<:Any,<:AbstractVector}) where T = Vector{T}(d.domain)
Array{T}(d::Inclusion{<:Any,<:AbstractVector}) where T = Array{T}(d.domain)
AbstractVector(d::Inclusion{<:Any,<:AbstractVector}) = d.domain
AbstractArray(d::Inclusion{<:Any,<:AbstractVector}) = d.domain
AbstractVector{T}(d::Inclusion{<:Any,<:AbstractVector}) where T = convert(AbstractVector{T},d.domain)
AbstractArray{T}(d::Inclusion{<:Any,<:AbstractVector}) where T =  convert(AbstractArray{T},d.domain)


copy(d::Inclusion) = d

==(A::Inclusion, B::Inclusion) = A.domain == B.domain
domain(A::Inclusion) = A.domain
domain(A::AbstractUnitRange) = A
domain(A::IdentityUnitRange) = A.indices
axes(S::Inclusion) = (S,)
axes1(S::Inclusion) = S
axes(S::Inclusion{<:Any,<:OneTo}) = (S.domain,)
axes1(S::Inclusion{<:Any,<:OneTo}) = S.domain

first(S::Inclusion) = first(S.domain)
last(S::Inclusion) = last(S.domain)
size(S::Inclusion) = (cardinality(S.domain),)
length(S::Inclusion) = cardinality(S.domain)
if VERSION < v"1.7-"
    Base.unsafe_length(S::Inclusion) = length(S)
end
cardinality(S::Inclusion) = cardinality(S.domain)
measure(x) = cardinality(x) # TODO: Inclusion(0:0.5:1) should have 
getindex(S::Inclusion{T}, i::T) where T = (@_inline_meta; @boundscheck checkbounds(S, i); convert(T,i))
getindex(S::Inclusion{T}, i::AbstractArray{T}) where T = (@_inline_meta; @boundscheck checkbounds(S, i); convert(AbstractArray{T},i))
getindex(S::Inclusion, i::Inclusion) = (@_inline_meta; @boundscheck checkbounds(S, i); copy(S))
getindex(S::Inclusion, ::Colon) = copy(S)
Base.unsafe_getindex(S::Inclusion{T}, x) where T = convert(T, x)::T
summary(io::IO, r::Inclusion) = print(io, "Inclusion(", r.domain, ")")
iterate(S::Inclusion, s...) = iterate(S.domain, s...)

in(x, S::Inclusion) = x in S.domain
Base.issubset(S::Inclusion, d) = S.domain ⊆ d
Base.issubset(S::Inclusion, d::Inclusion) = S.domain ⊆ d.domain

intersect(x::Inclusion...) = Inclusion{mapreduce(eltype,promote_type,x)}(intersect(map(domain,x)...))
# use UnionDomain to support intervals
_union(a::ClosedInterval...) = UnionDomain(a...)
_union(a...) = union(a...)
union(x::Inclusion...) = Inclusion{mapreduce(eltype,promote_type,x)}(_union(map(domain,x)...))

checkindex(::Type{Bool}, inds::Inclusion{T}, i::T) where T = i ∈ inds
checkindex(::Type{Bool}, inds::Inclusion, i) = i ⊆ inds
checkindex(::Type{Bool}, inds::Inclusion, ::Colon) = true
checkindex(::Type{Bool}, inds::Inclusion, ::Inclusion) = true

function __checkindex(::Type{Bool}, inds::Inclusion, I::AbstractArray)
    @_inline_meta
    b = true
    for i in I
        b &= checkindex(Bool, inds, i)
    end
    b
end

checkindex(::Type{Bool}, inds::Inclusion{T}, I::AbstractArray{T}) where T = 
    __checkindex(Bool, inds, I)
checkindex(::Type{Bool}, inds::Inclusion{T}, I::AbstractArray{T}) where T<:AbstractArray = 
    __checkindex(Bool, inds, I)
checkindex(::Type{Bool}, inds::Inclusion{T}, I::AbstractArray{<:AbstractArray}) where T<:AbstractArray = 
    __checkindex(Bool, inds, convert(AbstractArray{T}, I))


checkindex(::Type{Bool}, indx::Inclusion, I::AbstractVector{Bool}) = indx == axes1(I)
checkindex(::Type{Bool}, indx::Inclusion, I::AbstractArray{Bool}) = false

for find in (:(Base.findfirst), :(Base.findlast))
    @eval $find(f::Base.Fix2{typeof(isequal)}, d::Inclusion) = f.x in d.domain ? f.x : nothing
end

function Base.findall(f::Base.Fix2{typeof(isequal)}, d::Inclusion)
    r = findfirst(f,d)
    r === nothing ? eltype(d)[] : [r]
end

