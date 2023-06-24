# This file is a part of Julia. License is MIT: https://julialang.org/license


# L is true if the view itself supports fast linear indexing
struct SubQuasiArray{T,N,P,I,L} <: AbstractQuasiArray{T,N}
    parent::P
    indices::I
    offset1::Int       # for linear indexing and pointer, only valid when L==true
    stride1::Int       # used only for linear indexing
    function SubQuasiArray{T,N,P,I,L}(parent, indices, offset1, stride1) where {T,N,P,I,L}
        @_inline_meta
        check_parent_index_match(parent, indices)
        new(parent, indices, offset1, stride1)
    end
end
# Compute the linear indexability of the indices, and combine it with the linear indexing of the parent
function SubQuasiArray(parent::AbstractQuasiArray, indices::Tuple)
    @_inline_meta
    SubQuasiArray(IndexStyle(viewindexing(indices), IndexStyle(parent)), parent, ensure_indexable(indices), index_dimsum(indices...))
end
function SubQuasiArray(::IndexCartesian, parent::P, indices::I, ::NTuple{N,Any}) where {P,I,N}
    @_inline_meta
    SubQuasiArray{eltype(P), N, P, I, false}(parent, indices, 0, 0)
end
function SubQuasiArray(::IndexLinear, parent::P, indices::I, ::NTuple{N,Any}) where {P,I,N}
    @_inline_meta
    # Compute the stride and offset
    stride1 = compute_stride1(parent, indices)
    SubQuasiArray{eltype(P), N, P, I, true}(parent, indices, compute_offset1(parent, stride1, indices), stride1)
end

check_parent_index_match(parent::AbstractQuasiArray{T,N}, ::NTuple{N, Bool}) where {T,N} = nothing
viewindexing(I::Tuple{AbstractQuasiArray, Vararg{Any}}) = IndexCartesian()

# Simple utilities
size(V::SubQuasiArray) = (@_inline_meta; map(n->cardinality(n), axes(V)))

similar(V::SubQuasiArray, T::Type, dims::Dims) = similar(V.parent, T, dims)

sizeof(V::SubQuasiArray) = length(V) * sizeof(eltype(V))

parent(V::SubQuasiArray) = V.parent
parentindices(V::SubQuasiArray) = V.indices
parentindices(a::AbstractQuasiArray) = map(OneTo, size(a))

## Aliasing detection
dataids(A::SubQuasiArray) = (dataids(A.parent)..., _splatmap(dataids, A.indices)...)
unaliascopy(A::SubQuasiArray) = typeof(A)(unaliascopy(A.parent), map(unaliascopy, A.indices), A.offset1, A.stride1)


# Transform indices to be "dense"
_trimmedindex(i::AbstractQuasiArray) = oftype(i, reshape(eachindex(IndexLinear(), i), axes(i)))

## SubArray creation
# We always assume that the dimensionality of the parent matches the number of
# indices that end up getting passed to it, so we store the parent as a
# ReshapedArray view if necessary. The trouble is that arrays of `CartesianIndex`
# can make the number of effective indices not equal to length(I).
_maybe_reshape_parent(A::AbstractQuasiArray, ::NTuple{1, Bool}) = reshape(A, Val(1))
_maybe_reshape_parent(A::AbstractQuasiArray{<:Any,1}, ::NTuple{1, Bool}) = reshape(A, Val(1))
_maybe_reshape_parent(A::AbstractQuasiArray{<:Any,N}, ::NTuple{N, Bool}) where {N} = A
_maybe_reshape_parent(A::AbstractQuasiArray, ::NTuple{N, Bool}) where {N} = reshape(A, Val(N))


function view(A::AbstractQuasiArray, I::Vararg{Any,N}) where {N}
    @_inline_meta
    J = map(i->unalias(A,i), to_indices(A, I))
    @boundscheck checkbounds(A, J...)
    unsafe_view(_maybe_reshape_parent(A, index_ndims(J...)), J...)
end

const ViewIndex{IND} = Union{IND,AbstractArray{<:IND}}
const QViewIndex{IND} = Union{IND,AbstractArray{<:IND},AbstractQuasiArray{<:IND}}

# This computes the linear indexing compatibility for a given tuple of indices
quasi_viewindexing(::Tuple{}, I::Tuple{}) = IndexLinear()
# Leading scalar indices simply increase the stride
quasi_viewindexing(axs::Tuple{AbstractQuasiVector{IND}, Vararg{Any}}, I::Tuple{IND, Vararg{Any}}) where IND =
    (@_inline_meta; quasi_viewindexing(tail(axs), tail(I)))
quasi_viewindexing(axs::Tuple{AbstractVector{IND}, Vararg{Any}}, I::Tuple{IND, Vararg{Any}}) where IND =
    (@_inline_meta; quasi_viewindexing(tail(axs), tail(I)))

# Slices may begin a section which may be followed by any number of Slices
# quasi_viewindexing(axs, I::Tuple{Slice, Slice, Vararg{Any}}) = (@_inline_meta; quasi_viewindexing(tail(I)))
# # A UnitRange can follow Slices, but only if all other indices are scalar
# quasi_viewindexing(I::Tuple{Slice, AbstractUnitRange, Vararg{ScalarIndex}}) = IndexLinear()
# quasi_viewindexing(I::Tuple{Slice, Slice, Vararg{ScalarIndex}}) = IndexLinear() # disambiguate
# # In general, ranges are only fast if all other indices are scalar
# quasi_viewindexing(I::Tuple{AbstractRange, Vararg{ScalarIndex}}) = IndexLinear()
# # All other index combinations are slow
# quasi_viewindexing(I::Tuple{Vararg{Any}}) = IndexCartesian()
# # Of course, all other array types are slow
quasi_viewindexing(axs::Tuple{AbstractQuasiVector{IND}, Vararg{Any}}, I::Tuple{AbstractArray{IND}, Vararg{Any}}) where IND = IndexCartesian()
quasi_viewindexing(axs::Tuple{AbstractVector{IND}, Vararg{Any}}, I::Tuple{AbstractArray{IND}, Vararg{Any}}) where IND = IndexCartesian()

# combined dimensionality of all indices
# rather than returning N, it returns an NTuple{N,Bool} so the result is inferrable
@inline quasi_index_dimsum(axs::Tuple{AbstractQuasiVector{IND},Vararg{Any}}, inds::Tuple{IND,Vararg{Any}}) where IND =
    (quasi_index_dimsum(tail(axs), tail(inds))...,)
@inline quasi_index_dimsum(axs::Tuple{AbstractVector{IND},Vararg{Any}}, inds::Tuple{IND,Vararg{Any}}) where IND =
    (quasi_index_dimsum(tail(axs), tail(inds))...,)
@inline quasi_index_dimsum(axs, inds::Tuple{Colon,Vararg{Any}}) = (true, quasi_index_dimsum(tail(axs), tail(inds))...)
@inline quasi_index_dimsum(axs::Tuple{AbstractQuasiVector{IND},Vararg{Any}}, inds::Tuple{AbstractArray{IND,N},Vararg{Any}}) where {N,IND} =
    (ntuple(x->true, Val(N))..., quasi_index_dimsum(tail(axs), tail(inds))...)
@inline quasi_index_dimsum(axs::Tuple{AbstractVector{IND},Vararg{Any}}, inds::Tuple{AbstractArray{IND,N},Vararg{Any}}) where {N,IND} =
    (ntuple(x->true, Val(N))..., quasi_index_dimsum(tail(axs), tail(inds))...)
quasi_index_dimsum(::Tuple{}, ::Tuple{}) = ()

function SubArray(parent::AbstractQuasiArray, indices::Tuple)
    @_inline_meta
    SubArray(IndexStyle(quasi_viewindexing(axes(parent), indices), IndexStyle(parent)), parent, ensure_indexable(indices), quasi_index_dimsum(axes(parent), indices))
end


unsafe_view(A::AbstractQuasiArray, I...) = _unsafe_view(indextype(A), A, Base.to_indices(A, I))
unsafe_view(V::SubQuasiArray, I...) = (@_inline_meta; _maybe_reindex(V, Base.to_indices(V, I)))
_unsafe_view(::Type{INDS}, A::AbstractQuasiArray, I::Tuple) where {INDS} = _unsafe_view_type(INDS, I)(A, I)
# If Array-based indexing, use SubArray, otherwise SubQuasiArray
_unsafe_view_type(::Type{Tuple{}}, ::Tuple) = SubArray
_unsafe_view_type(inds::Type{<:Tuple{IND,Vararg{Any}}}, I::Tuple{ViewIndex{IND},Vararg{Any}}) where IND = _unsafe_view_type(Base.tuple_type_tail(inds), tail(I))
_unsafe_view_type(inds::Type{<:Tuple{IND,Vararg{Any}}}, I::Tuple{AbstractQuasiArray{IND},Vararg{Any}}) where IND = SubQuasiArray
# _unsafe_view_type(inds::Type{<:Tuple{IND,Vararg{Any}}}, I::Tuple{Any,Vararg{Any}}) where IND = _unsafe_view_type(inds, tuple(convert(IND, I[1]), tail(I)...))



# When we take the view of a view, it's often possible to "reindex" the parent
# view's indices such that we can "pop" the parent view and keep just one layer
# of indirection. But we can't always do this because arrays of `CartesianIndex`
# might span multiple parent indices, making the reindex calculation very hard.
# So we use _maybe_reindex to figure out if there are any arrays of
# `CartesianIndex`, and if so, we punt and keep two layers of indirection.
_unsafe_view(::Type{IND}, V::SubQuasiArray, I::Vararg{ViewIndex{IND},N}) where {N,IND} =
    (@_inline_meta; _maybe_reindex(V, I))
_unsafe_view(::Type{IND}, V::SubQuasiArray, I::Vararg{QViewIndex{IND},N}) where {N,IND} =
    (@_inline_meta; _maybe_reindex(V, I))

_maybe_reindex(V, I) = (@_inline_meta; _maybe_reindex(V, I, I))
# _maybe_reindex(V, I, ::Tuple{AbstractArray{<:AbstractCartesianIndex}, Vararg{Any}}) =
#     (@_inline_meta; SubArray(V, I))
# # But allow arrays of CartesianIndex{1}; they behave just like arrays of Ints
# _maybe_reindex(V, I, A::Tuple{AbstractArray{<:AbstractCartesianIndex{1}}, Vararg{Any}}) =
#     (@_inline_meta; _maybe_reindex(V, I, tail(A)))
_maybe_reindex(V, I, A::Tuple{Any, Vararg{Any}}) = (@_inline_meta; _maybe_reindex(V, I, tail(A)))

_subarray(::Type{IND}, A::AbstractArray, idxs) where IND = SubArray(A, idxs)
_subarray(::Type{IND}, A::AbstractQuasiArray, idxs) where IND = _unsafe_view_type(IND, idxs)(A, idxs)

function _maybe_reindex(V, I, ::Tuple{})
    @_inline_meta
    @inbounds idxs = to_indices(V.parent, quasi_reindex(axes(parent(V)), V.indices, I))
    _subarray(indextype(V.parent), V.parent, idxs)
end

Base._maybe_reindex(V::SubArray{<:Any,N,<:AbstractQuasiArray}, I, ::Tuple{}) where N = _maybe_reindex(V, I, ())

## Re-indexing is the heart of a view, transforming A[i, j][x, y] to A[i[x], j[y]]
#
# Recursively look through the heads of the parent- and sub-indices, considering
# the following cases:
# * Parent index is array  -> re-index that with one or more sub-indices (one per dimension)
# * Parent index is Colon  -> just use the sub-index as provided
# * Parent index is scalar -> that dimension was dropped, so skip the sub-index and use the index as is

AbstractZeroDimQuasiArray{T} = AbstractQuasiArray{T, 0}

# Re-index into parent vectors with one subindex
reindex(idxs::Tuple{AbstractQuasiVector, Vararg{Any}}, subidxs::Tuple{Any, Vararg{Any}}) =
    (@_propagate_inbounds_meta; (idxs[1][subidxs[1]], reindex(tail(idxs), tail(subidxs))...))

# Parent matrices are re-indexed with two sub-indices
reindex(idxs::Tuple{AbstractQuasiMatrix, Vararg{Any}}, subidxs::Tuple{Any, Any, Vararg{Any}}) =
    (@_propagate_inbounds_meta; (idxs[1][subidxs[1], subidxs[2]], reindex(tail(idxs), tail(tail(subidxs)))...))

# In general, we index N-dimensional parent arrays with N indices
@generated function reindex(idxs::Tuple{AbstractQuasiArray{T,N}, Vararg{Any}}, subidxs::Tuple{Vararg{Any}}) where {T,N}
    if length(subidxs.parameters) >= N
        subs = [:(subidxs[$d]) for d in 1:N]
        tail = [:(subidxs[$d]) for d in N+1:length(subidxs.parameters)]
        :(@_propagate_inbounds_meta; (idxs[1][$(subs...)], reindex(tail(idxs), ($(tail...),))...))
    else
        :(throw(ArgumentError("cannot re-index SubArray with fewer indices than dimensions\nThis should not occur; please submit a bug report.")))
    end
end

# indices are taken from the range/vector
# Since bounds-checking is performance-critical and uses
# indices, it's worth optimizing these implementations thoroughly
axes(S::SubArray{T,N,<:AbstractQuasiArray}) where {T,N} =
    _quasi_indices_sub(axes(parent(S)), S.indices)
_quasi_indices_sub(axs::Tuple{AbstractQuasiVector{IND},Vararg{Any}}, inds::Tuple{IND,Vararg{Any}}) where IND =
    (@_inline_meta; _quasi_indices_sub(tail(axs), tail(inds)))
_quasi_indices_sub(axs::Tuple{AbstractVector{IND},Vararg{Any}}, inds::Tuple{IND,Vararg{Any}}) where IND =
    (@_inline_meta; _quasi_indices_sub(tail(axs), tail(inds)))
_quasi_indices_sub(::Tuple{}, ::Tuple{}) = ()
function _quasi_indices_sub(axs::Tuple{AbstractQuasiVector{IND},Vararg{Any}}, inds::Tuple{AbstractArray{IND},Vararg{Any}}) where IND
    @_inline_meta
    (axes(inds[1])..., _quasi_indices_sub(tail(axs), tail(inds))...)
end
function _quasi_indices_sub(axs::Tuple{AbstractVector{IND},Vararg{Any}}, inds::Tuple{AbstractArray{IND},Vararg{Any}}) where IND
    @_inline_meta
    (axes(inds[1])..., _quasi_indices_sub(tail(axs), tail(inds))...)
end

quasi_reindex(axs::Tuple{AbstractQuasiOrVector{IND}, Vararg{Any}}, idxs::Tuple{IND, Vararg{Any}}, subidxs::Tuple{Vararg{Any}}) where IND =
    (@_propagate_inbounds_meta; (idxs[1], quasi_reindex(tail(axs), tail(idxs), subidxs)...))
quasi_reindex(axs::Tuple{AbstractQuasiOrVector{IND}, Vararg{Any}}, idxs::Tuple{AbstractQuasiOrVector{IND}, Vararg{Any}}, subidxs::Tuple{Any, Vararg{Any}}) where IND =
    (@_propagate_inbounds_meta; (idxs[1][subidxs[1]], quasi_reindex(tail(axs), tail(idxs), tail(subidxs))...))

# Parent matrices are re-indexed with two sub-indices
quasi_reindex(axs::Tuple{AbstractQuasiOrVector{IND}, Vararg{Any}},idxs::Tuple{AbstractQuasiOrMatrix{IND}, Vararg{Any}}, subidxs::Tuple{Any, Any, Vararg{Any}}) where IND =
    (@_propagate_inbounds_meta; (idxs[1][subidxs[1], subidxs[2]], quasi_reindex(tail(axs), tail(idxs), tail(tail(subidxs)))...))


quasi_reindex(::Tuple{}, ::Tuple{}, ::Tuple{}) = ()

function getindex(V::SubArray{T,N,<:AbstractQuasiArray}, I::Vararg{Int,N}) where {T,N}
    @_inline_meta
    @boundscheck checkbounds(V, I...)
    @inbounds r = V.parent[quasi_reindex(axes(parent(V)), V.indices, I)...]
    r
end



# In general, we simply re-index the parent indices by the provided ones
SlowSubQuasiArray{T,N,P,I} = SubQuasiArray{T,N,P,I,false}
function _getindex(::Type{IND}, V::SlowSubQuasiArray, I::IND) where IND
    @_inline_meta
    @boundscheck checkbounds(V, I...)
    @inbounds r = V.parent[quasi_reindex(axes(parent(V)), V.indices, I)...]
    r
end

function _setindex!(::Type{IND}, V::SlowSubQuasiArray, x, I::IND) where IND
    @_inline_meta
    @boundscheck checkbounds(V, I...)
    @inbounds V.parent[quasi_reindex(axes(parent(V)), V.indices, I)...] = x
    V
end

IndexStyle(::Type{<:SubQuasiArray}) = IndexCartesian()

# Strides are the distance in memory between adjacent elements in a given dimension
# which we determine from the strides of the parent
strides(V::SubQuasiArray) = substrides(V.parent, V.indices)
substrides(parent, strds, I::Tuple{Any, Vararg{Any}}) = throw(ArgumentError("strides is invalid for SubQuasiArrays with indices of type $(typeof(I[1]))"))

stride(V::SubQuasiArray, d::Integer) = d <= ndims(V) ? strides(V)[d] : strides(V)[end] * size(V)[end]

compute_stride1(parent::AbstractQuasiArray, I::NTuple{N,Any}) where {N} =
    1

elsize(::Type{<:SubQuasiArray{<:Any,<:Any,P}}) where {P} = elsize(P)

function first_index(V::SubQuasiArray)
    P, I = parent(V), V.indices
    s1 = compute_stride1(P, I)
    s1 + compute_offset1(P, s1, I)
end

# Computing the first index simply steps through the indices, accumulating the
# sum of index each multiplied by the parent's stride.
# The running sum is `f`; the cumulative stride product is `s`.
# If the parent is a vector, then we offset the parent's own indices with parameters of I
compute_offset1(parent::AbstractQuasiVector, stride1::Integer, I::Tuple{AbstractRange}) = 0
# If the result is one-dimensional and it's a Colon, then linear
# indexing uses the indices along the given dimension. Otherwise
# linear indexing always starts with 1.

function compute_linindex(parent, I::NTuple{N,Any}) where N
    @_inline_meta
    IP = fill_to_length(axes(parent), OneTo(1), Val(N))
    compute_linindex(1, 1, IP, I)
end
function compute_linindex(f, s, IP::Tuple, I::Tuple{ScalarIndex, Vararg{Any}})
    @_inline_meta
    Δi = I[1]-first(IP[1])
    compute_linindex(f + Δi*s, s*length(IP[1]), tail(IP), tail(I))
end
function compute_linindex(f, s, IP::Tuple, I::Tuple{Any, Vararg{Any}})
    @_inline_meta
    Δi = first(I[1])-first(IP[1])
    compute_linindex(f + Δi*s, s*length(IP[1]), tail(IP), tail(I))
end
compute_linindex(f, s, IP::Tuple, I::Tuple{}) = f

find_extended_dims(dim, ::ScalarIndex, I...) = (@_inline_meta; find_extended_dims(dim + 1, I...))
find_extended_dims(dim, i1, I...) = (@_inline_meta; (dim, find_extended_dims(dim + 1, I...)...))
find_extended_dims(dim) = ()
find_extended_inds(::ScalarIndex, I...) = (@_inline_meta; find_extended_inds(I...))
find_extended_inds(i1, I...) = (@_inline_meta; (i1, find_extended_inds(I...)...))
find_extended_inds() = ()

unsafe_convert(::Type{Ptr{T}}, V::SubQuasiArray{T,N,P,<:Tuple{Vararg{RangeIndex}}}) where {T,N,P} =
    unsafe_convert(Ptr{T}, V.parent) + (first_index(V)-1)*sizeof(T)

pointer(V::SubQuasiArray, i::Int) = _pointer(V, i)
_pointer(V::SubQuasiArray{<:Any,1}, i::Int) = pointer(V, (i,))
_pointer(V::SubQuasiArray, i::Int) = pointer(V, Base._ind2sub(axes(V), i))


# indices are taken from the range/vector
# Since bounds-checking is performance-critical and uses
# indices, it's worth optimizing these implementations thoroughly
axes(S::SubQuasiArray) = (@_inline_meta; _indices_sub(S, S.indices...))
_indices_sub(S::SubQuasiArray) = ()
_indices_sub(S::SubQuasiArray, ::Number, I...) = (@_inline_meta; _indices_sub(S, I...))
function _indices_sub(S::SubQuasiArray, i1::Union{AbstractQuasiArray,AbstractArray}, I...)
    @_inline_meta
    (axes(i1)..., _indices_sub(S, I...)...)
end

@propagate_inbounds maybeview(A::AbstractQuasiArray, args...) = view(A, args...)
@propagate_inbounds maybeview(A::AbstractQuasiArray, args::Number...) = getindex(A, args...)
@propagate_inbounds maybeview(A::AbstractQuasiArray) = getindex(A)


##
# MemoryLayout
##

@inline MemoryLayout(A::Type{<:SubQuasiArray{T,N,P,I}}) where {T,N,P,I} =
    sublayout(MemoryLayout(P), I)



@inline sub_materialize(_, V::AbstractQuasiArray, _) = QuasiArray(V)
@inline sub_materialize(V::SubQuasiArray) = sub_materialize(MemoryLayout(typeof(V)), V)
@inline sub_materialize(A::AbstractQuasiArray) = A # already materialized


###
# show
###

show(io::IO, v::SubQuasiArray) = summary(io, v)

function summary(io::IO, v::SubQuasiArray)
    print(io, "view(")
    summary(io, parent(v))
    Base.showindices(io, v.indices...)
    print(io, ')')
    print(io, " with eltype ", eltype(v))
end

function (==)(A::SubQuasiArray, B::AbstractQuasiArray)
    parentindices(A) == axes(parent(A)) && return parent(A) == B
    Base.invoke(==, NTuple{2,AbstractQuasiArray}, A, B)
end

function (==)(A::AbstractQuasiArray, B::SubQuasiArray)
    parentindices(B) == axes(parent(B)) && return A == parent(B)
    Base.invoke(==, NTuple{2,AbstractQuasiArray}, A, B)
end

function (==)(A::SubQuasiArray, B::SubQuasiArray)
    parentindices(A) == axes(parent(A)) && return parent(A) == B
    parentindices(B) == axes(parent(B)) && return A == parent(B)
    Base.invoke(==, NTuple{2,AbstractQuasiArray}, A, B)
end