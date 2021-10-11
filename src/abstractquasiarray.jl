

# This file is a part of Julia. License is MIT: https://julialang.org/license

## Basic functions ##

"""
    AbstractQuasiArray{T,N}

Supertype for `N`-dimensional arrays (or array-like types) with elements of type `T`.
[`Array`](@ref) and other types are subtypes of this. See the manual section on the
[`AbstractQuasiArray` interface](@ref man-interface-array).
"""
AbstractQuasiArray

convert(::Type{T}, a::T) where {T<:AbstractQuasiArray} = a
convert(::Type{AbstractQuasiArray{T}}, a::AbstractQuasiArray) where {T} = AbstractQuasiArray{T}(a)
convert(::Type{AbstractQuasiArray{T,N}}, a::AbstractQuasiArray{<:Any,N}) where {T,N} = AbstractQuasiArray{T,N}(a)

Array{T}(a::AbstractQuasiArray) where T = T[a[k] for k in eachindex(a)]
Array{T,N}(a::AbstractQuasiArray{<:Any,N}) where {T,N} = Array{T}(a)
Array(a::AbstractQuasiArray{T}) where T = Array{T}(a)
Matrix(a::AbstractQuasiMatrix{T}) where T = Array{T}(a)
Vector(a::AbstractQuasiVector{T}) where T = Array{T}(a)
AbstractArray(a::AbstractQuasiArray) = Array(a)
AbstractMatrix(a::AbstractQuasiMatrix) = Matrix(a)
AbstractVector(a::AbstractQuasiVector) = Vector(a)
AbstractArray{T}(a::AbstractQuasiArray) where T = Array{T}(a)
AbstractMatrix{T}(a::AbstractQuasiMatrix) where T = Matrix{T}(a)
AbstractVector{T}(a::AbstractQuasiVector) where T = Vector{T}(a)


convert(::Type{Array{T}}, a::AbstractQuasiArray) where T = Array{T}(a)
convert(::Type{Array{T,N}}, a::AbstractQuasiArray{<:Any,N}) where {T,N} = Array{T}(a)
convert(::Type{Array}, a::AbstractQuasiArray{T}) where T = Array{T}(a)
convert(::Type{Matrix}, a::AbstractQuasiMatrix{T}) where T = Array{T}(a)
convert(::Type{Vector}, a::AbstractQuasiVector{T}) where T = Array{T}(a)

convert(::Type{AbstractArray{T}}, a::AbstractQuasiArray) where T = Array{T}(a)
convert(::Type{AbstractArray}, a::AbstractQuasiArray{T}) where T = convert(AbstractArray{T}, a)
convert(::Type{AbstractArray{T,N}}, a::AbstractQuasiArray{<:Any,N}) where {T,N} = convert(AbstractArray{T}, a)
convert(::Type{AbstractMatrix}, a::AbstractQuasiMatrix) = convert(AbstractArray, a)
convert(::Type{AbstractVector}, a::AbstractQuasiVector) = convert(AbstractArray, a)


"""
    size(A::AbstractQuasiArray, [dim])

Return a tuple containing the dimensions of `A`. Optionally you can specify a
dimension to just get the length of that dimension.

Note that `size` may not be defined for arrays with non-standard indices, in which case [`axes`](@ref)
may be useful. See the manual chapter on [arrays with custom indices](@ref man-custom-indices).

# Examples
```jldoctest
julia> A = fill(1, (2,3,4));

julia> size(A)
(2, 3, 4)

julia> size(A, 2)
3
```
"""
size(t::AbstractQuasiArray{T,N}, d) where {T,N} = d <= N ? size(t)[d] : 1

"""
    axes(A, d)

Return the valid range of indices for array `A` along dimension `d`.

See also [`size`](@ref), and the manual chapter on [arrays with custom indices](@ref man-custom-indices).

# Examples
```jldoctest
julia> A = fill(1, (5,6,7));

julia> axes(A, 2)
Base.OneTo(6)
```
"""
function axes(A::AbstractQuasiArray{T,N}, d) where {T,N}
    @_inline_meta
    d <= N ? axes(A)[d] : OneTo(1)
end


# Performance optimization: get rid of a branch on `d` in `axes(A, d)`
# for d=1. 1d arrays are heavily used, and the first dimension comes up
# in other applications.
axes1(A::AbstractQuasiArray{<:Any,0}) = OneTo(1)
axes1(A::AbstractQuasiArray) = (@_inline_meta; axes(A)[1])

keys(a::AbstractQuasiArray) = CartesianIndices(axes(a))
keys(a::AbstractQuasiVector) = LinearIndices(a)


eltype(::Type{<:AbstractQuasiArray{E}}) where {E} = @isdefined(E) ? E : Any
elsize(A::AbstractQuasiArray) = elsize(typeof(A))

"""
    ndims(A::AbstractQuasiArray) -> Integer

Return the number of dimensions of `A`.

# Examples
```jldoctest
julia> A = fill(1, (3,4,5));

julia> ndims(A)
3
```
"""
ndims(::AbstractQuasiArray{T,N}) where {T,N} = N
ndims(::Type{<:AbstractQuasiArray{T,N}}) where {T,N} = N

"""
    length(collection) -> Integer

Return the number of elements in the collection.

Use [`lastindex`](@ref) to get the last valid index of an indexable collection.

# Examples
```jldoctest
julia> length(1:5)
5

julia> length([1, 2, 3, 4])
4

julia> length([1 2; 3 4])
4
```
"""
length

"""
    length(A::AbstractQuasiArray)

Return the number of elements in the array, defaults to `prod(size(A))`.

# Examples
```jldoctest
julia> length([1, 2, 3, 4])
4

julia> length([1 2; 3 4])
4
```
"""
length(t::AbstractQuasiArray) = (@_inline_meta; prod(size(t)))

# eachindex iterates over all indices. IndexCartesian definitions are later.
eachindex(A::AbstractQuasiVector) = (@_inline_meta(); axes1(A))

"""
    eachindex(A...)

Create an iterable object for visiting each index of an `AbstractQuasiArray` `A` in an efficient
manner. For array types that have opted into fast linear indexing (like `Array`), this is
simply the range `1:length(A)`. For other array types, return a specialized Cartesian
range to efficiently index into the array with indices specified for every dimension. For
other iterables, including strings and dictionaries, return an iterator object
supporting arbitrary index types (e.g. unevenly spaced or non-integer indices).

If you supply more than one `AbstractQuasiArray` argument, `eachindex` will create an
iterable object that is fast for all arguments (a [`UnitRange`](@ref)
if all inputs have fast linear indexing, a [`CartesianIndices`](@ref)
otherwise).
If the arrays have different sizes and/or dimensionalities, `eachindex` will return an
iterable that spans the largest range along each dimension.

# Examples
```jldoctest
julia> A = [1 2; 3 4];

julia> for i in eachindex(A) # linear indexing
           println(i)
       end
1
2
3
4

julia> for i in eachindex(view(A, 1:2, 1:1)) # Cartesian indexing
           println(i)
       end
CartesianIndex(1, 1)
CartesianIndex(2, 1)
```
"""
eachindex(A::AbstractQuasiArray) = (@_inline_meta(); eachindex(IndexStyle(A), A))

function eachindex(A::AbstractQuasiArray, B::AbstractQuasiArray)
    @_inline_meta
    eachindex(IndexStyle(A,B), A, B)
end
function eachindex(A::AbstractQuasiArray, B::AbstractQuasiArray...)
    @_inline_meta
    eachindex(IndexStyle(A,B...), A, B...)
end
eachindex(::IndexLinear, A::AbstractQuasiArray) = (@_inline_meta; OneTo(length(A)))
eachindex(::IndexLinear, A::AbstractQuasiVector) = (@_inline_meta; axes1(A))
function eachindex(::IndexLinear, A::AbstractQuasiArray, B::AbstractQuasiArray...)
    @_inline_meta
    indsA = eachindex(IndexLinear(), A)
    _all_match_first(X->eachindex(IndexLinear(), X), indsA, B...) ||
        throw_eachindex_mismatch(IndexLinear(), A, B...)
    indsA
end


# keys with an IndexStyle
keys(s::IndexStyle, A::AbstractQuasiArray, B::AbstractQuasiArray...) = eachindex(s, A, B...)

"""
    lastindex(collection) -> index
    lastindex(collection, d) -> index

Return the last index of `collection`. If `d` is given, return the last index of `collection` along dimension `d`.

The syntaxes `A[end]` and `A[end, end]` lower to `A[lastindex(A)]` and
`A[lastindex(A, 1), lastindex(A, 2)]`, respectively.

# Examples
```jldoctest
julia> lastindex([1,2,4])
3

julia> lastindex(rand(3,4,5), 2)
4
```
"""
lastindex(a::AbstractQuasiArray) = (@_inline_meta; last(eachindex(IndexLinear(), a)))
lastindex(a::AbstractQuasiArray, d) = (@_inline_meta; last(axes(a, d)))

"""
    firstindex(collection) -> index
    firstindex(collection, d) -> index

Return the first index of `collection`. If `d` is given, return the first index of `collection` along dimension `d`.

# Examples
```jldoctest
julia> firstindex([1,2,4])
1

julia> firstindex(rand(3,4,5), 2)
1
```
"""
firstindex(a::AbstractQuasiArray) = (@_inline_meta; first(eachindex(IndexLinear(), a)))
firstindex(a::AbstractQuasiArray, d) = (@_inline_meta; first(axes(a, d)))

first(a::AbstractQuasiArray) = a[first(eachindex(a))]
stride(A::AbstractQuasiArray, k::Integer) = strides(A)[k]

function isassigned(a::AbstractQuasiArray, i...)
    try
        a[i...]
        true
    catch e
        if isa(e, BoundsError) || isa(e, UndefRefError)
            return false
        else
            rethrow(e)
        end
    end
end

function checkbounds(::Type{Bool}, A::AbstractQuasiArray, I...)
    @_inline_meta
    checkbounds_indices(Bool, axes(A), I)
end

# As a special extension, allow using logical arrays that match the source array exactly
function checkbounds(::Type{Bool}, A::AbstractQuasiArray{<:Any,N}, I::AbstractQuasiArray{Bool,N}) where N
    @_inline_meta
    axes(A) == axes(I)
end


function checkbounds(A::AbstractQuasiArray, I...)
    @_inline_meta
    checkbounds(Bool, A, to_indices(A,I)...) || throw_boundserror(A, I)
    nothing
end

const QuasiDimOrInd = Union{Number, AbstractQuasiOrVector{<:Number}}

similar(a::AbstractQuasiArray{T}) where {T}                             = similar(a, T)
similar(a::AbstractQuasiArray, ::Type{T}) where {T}                     = similar(a, T, axes(a))
similar(a::AbstractQuasiArray{T}, dims::Tuple) where {T}                = similar(a, T, dims)
similar(a::AbstractQuasiArray{T}, dims::QuasiDimOrInd...) where {T}          = similar(a, T, dims)
similar(a::AbstractQuasiArray, ::Type{T}, dims::QuasiDimOrInd...) where {T}  = similar(a, T, dims)
similar(::Type{<:AbstractQuasiArray{T}}, shape::NTuple{N,AbstractQuasiOrVector}) where {N,T} =
    QuasiArray{T,N}(undef, convert.(AbstractVector, shape))
similar(a::AbstractQuasiArray, ::Type{T}, dims::NTuple{N,AbstractQuasiOrVector}) where {T,N} =
    QuasiArray{T,N}(undef, convert.(AbstractVector, dims))
similar(a::AbstractQuasiArray, ::Type{T}, dims::Vararg{AbstractQuasiOrVector,N}) where {T,N} =
    QuasiArray{T,N}(undef, convert.(AbstractVector, dims))

similar(a::AbstractQuasiArray{T}, m::Int) where {T}              = Vector{T}(undef, m)
similar(a::AbstractQuasiArray, T::Type, dims::Dims{N}) where {N} = Array{T,N}(undef, dims)
similar(a::AbstractQuasiArray{T}, dims::Dims{N}) where {T,N}     = Array{T,N}(undef, dims)


similar(::Type{T}, dims::QuasiDimOrInd...) where {T<:AbstractQuasiArray} = similar(T, dims)
similar(::Type{T}, shape::Tuple{AbstractQuasiVector,Vararg{QuasiDimOrInd}}) where {N,T<:AbstractArray{<:Any,N}} = 
    QuasiArray{T,N}(undef, convert.(AbstractVector, shape))

empty(a::AbstractQuasiVector{T}, ::Type{U}=T) where {T,U} = Vector{U}()

# like empty, but should return a mutable collection, a Vector by default
emptymutable(a::AbstractQuasiVector{T}, ::Type{U}=T) where {T,U} = Vector{U}()


## copy between abstract arrays - generally more efficient
## since a single index variable can be used.

copyto!(dest::AbstractQuasiArray, src::AbstractQuasiArray) =
    copyto!(IndexStyle(dest), dest, IndexStyle(src), src)

function copyto!(::IndexStyle, dest::AbstractQuasiArray, ::IndexStyle, src::AbstractQuasiArray)
    destinds, srcinds = LinearIndices(dest), LinearIndices(src)
    isempty(srcinds) || (checkbounds(Bool, destinds, first(srcinds)) && checkbounds(Bool, destinds, last(srcinds))) ||
        throw(BoundsError(dest, srcinds))
    @inbounds for i in srcinds
        dest[i] = src[i]
    end
    return dest
end

function copyto!(::IndexStyle, dest::AbstractQuasiArray, ::IndexCartesian, src::AbstractQuasiArray)
    axes(dest) == axes(src) || throw(BoundsError(dest, axes(src)))
    i = 0
    @inbounds for i in eachindex(src)
        dest[i] = src[i]
    end
    return dest
end

function copy(a::AbstractQuasiArray)
    @_propagate_inbounds_meta
    copymutable(a)
end


function copymutable(a::AbstractQuasiArray)
    @_propagate_inbounds_meta
    copyto!(similar(a), a)
end

## iteration support for arrays by iterating over `eachindex` in the array ##
# Allows fast iteration by default for both IndexLinear and IndexCartesian arrays

# While the definitions for IndexLinear are all simple enough to inline on their
# own, IndexCartesian's CartesianIndices is more complicated and requires explicit
# inlining.
function iterate(A::AbstractQuasiArray, state=(eachindex(A),))
    y = iterate(state...)
    y === nothing && return nothing
    A[y[1]], (state[1], tail(y)...)
end

isempty(a::AbstractQuasiArray) = (length(a) == 0)

getindex(A::AbstractQuasiArray, I...) = _getindex(indextype(A), A, I)

function _getindex(::Type{IND}, A::AbstractQuasiArray, I) where IND
    @_propagate_inbounds_meta
    error_if_canonical_getindex(IndexStyle(A), A, I)
    _getindex(IND, IndexStyle(A), A, to_indices(A, I))
end
function unsafe_getindex(A::AbstractQuasiArray, I...)
    @_inline_meta
    @inbounds r = getindex(A, I...)
    r
end

## Internal definitions
_getindex(_, ::IndexStyle, A::AbstractQuasiArray, I) = layout_getindex(A, I...)


## IndexCartesian Scalar indexing: Canonical method is full dimensionality of indices
function _getindex(::Type{IND}, ::IndexCartesian, A::AbstractQuasiArray, I::IND) where {M,IND}
    @_inline_meta
    @boundscheck checkbounds(A, I...) # generally _to_subscript_indices requires bounds checking
    @inbounds r = getindex(A, _to_subscript_indices(A, I...)...)
    r
end

error_if_canonical_getindex(::IndexCartesian, A::AbstractQuasiArray{T,N}, I::Tuple) where {T,N} =
    _error_if_canonical_getindex(indextype(A), A, I)

error_if_canonical_getindex(::IndexLinear, A::AbstractQuasiArray{T,N}, I::Tuple) where {T,N} =
    _error_if_canonical_getindex(indextype(A), A, I)

_error_if_canonical_getindex(::Type{IND}, A::AbstractQuasiArray{T,N}, I::IND) where {T,N,IND} =
    error("getindex not defined for ", typeof(A))

_error_if_canonical_getindex(::Type, ::AbstractQuasiArray, ::Any...) = nothing    

_to_subscript_indices(A::AbstractQuasiArray, i) = (@_inline_meta; _unsafe_ind2sub(A, i))
_to_subscript_indices(A::AbstractQuasiArray{T,N}) where {T,N} = (@_inline_meta; fill_to_length((), 1, Val(N)))
_to_subscript_indices(A::AbstractQuasiArray{T,0}) where {T} = ()
_to_subscript_indices(A::AbstractQuasiArray{T,0}, i) where {T} = ()
_to_subscript_indices(A::AbstractQuasiArray{T,0}, I...) where {T} = ()
function _to_subscript_indices(A::AbstractQuasiArray{T,N}, I...) where {T,N}
    @_inline_meta
    J, Jrem = IteratorsMD.split(I, Val(N))
    _to_subscript_indices(A, J, Jrem)
end
_to_subscript_indices(A::AbstractQuasiArray, J::Tuple, Jrem::Tuple{}) =
    __to_subscript_indices(A, axes(A), J, Jrem)
function __to_subscript_indices(A::AbstractQuasiArray,
        ::Tuple{AbstractUnitRange,Vararg{AbstractUnitRange}}, J::Tuple, Jrem::Tuple{})
    @_inline_meta
    (J..., map(first, tail(_remaining_size(J, axes(A))))...)
end
_to_subscript_indices(A::AbstractQuasiArray{T,N}, I::Vararg{Any,N}) where {T,N} = I

## Setindex! is defined similarly. We first dispatch to an internal _setindex!
# function that allows dispatch on array storage

setindex!(A::AbstractQuasiArray, v, I...) = _setindex!(indextype(A), A, v, I)

function _setindex!(::Type{IND}, A::AbstractQuasiArray, v, I) where IND
    @_propagate_inbounds_meta
    error_if_canonical_setindex(IndexStyle(A), A, I)
    _setindex!(IND, IndexStyle(A), A, v, to_indices(A, I))
end

error_if_canonical_setindex(::IndexCartesian, A::AbstractQuasiArray{T,N}, I::Tuple) where {T,N} =
    _error_if_canonical_setindex(indextype(A), A, I)

error_if_canonical_setindex(::IndexLinear, A::AbstractQuasiArray{T,N}, I::Tuple) where {T,N} =
    _error_if_canonical_setindex(indextype(A), A, I)    

_error_if_canonical_setindex(::Type{IND}, A::AbstractQuasiArray{T,N}, I::IND) where {T,N,IND} =
    error("setindex! not defined for ", typeof(A))

_error_if_canonical_setindex(::Type, ::AbstractQuasiArray, ::Any...) = nothing    
## Internal definitions
_setindex!(::Type, ::IndexStyle, A::AbstractQuasiArray, v, I) =
    error("setindex! for $(typeof(A)) with types $(typeof(I)) is not supported")

# IndexCartesian Scalar indexing
function _setindex!(::Type{IND}, ::IndexCartesian, A::AbstractQuasiArray{T,N}, v, I::NTuple{N,Any}) where {T,N,IND}
    @_propagate_inbounds_meta
    setindex!(A, v, I...)
end
function _setindex!(::Type{IND}, ::IndexCartesian, A::AbstractQuasiArray, v, I::NTuple{M,Any}) where {M,IND}
    @_inline_meta
    @boundscheck checkbounds(A, I...)
    @inbounds r = setindex!(A, v, _to_subscript_indices(IND, A, I...)...)
    r
end


parent(a::AbstractQuasiArray) = a


unalias(dest, A::AbstractQuasiArray) = mightalias(dest, A) ? unaliascopy(A) : A
unaliascopy(A::AbstractQuasiArray)::typeof(A) = (@_noinline_meta; _unaliascopy(A, copy(A)))

"""
    Base.mightalias(A::AbstractQuasiArray, B::AbstractQuasiArray)

Perform a conservative test to check if quasi arrays `A` and `B` might share the same memory.

By default, this simply checks if either of the arrays reference the same memory
regions, as identified by their [`Base.dataids`](@ref).
"""
mightalias(A::AbstractQuasiArray, B::AbstractQuasiArray) = !_isdisjoint(dataids(A), dataids(B))

"""
    Base.dataids(A::AbstractQuasiArray)

Return a tuple of `UInt`s that represent the mutable data segments of an array.

Custom arrays that would like to opt-in to aliasing detection of their component
parts can specialize this method to return the concatenation of the `dataids` of
their component parts.  A typical definition for an array that wraps a parent is
`Base.dataids(C::CustomArray) = dataids(C.parent)`.
"""
dataids(A::AbstractQuasiArray) = (UInt(objectid(A)),)


## Reductions and accumulates ##

function isequal(A::AbstractQuasiArray, B::AbstractQuasiArray)
    if A === B return true end
    if axes(A) != axes(B)
        return false
    end
    for (a, b) in zip(A, B)
        if !isequal(a, b)
            return false
        end
    end
    return true
end

function _equals(_, _, A, B)
    if axes(A) != axes(B)
        return false
    end
    anymissing = false
    for (a, b) in zip(A, B)
        eq = (a == b)
        if ismissing(eq)
            anymissing = true
        elseif !eq
            return false
        end
    end
    return anymissing ? missing : true
end


(==)(A::AbstractQuasiArray, B::AbstractQuasiArray) = _equals(MemoryLayout(A), MemoryLayout(B), A, B)

##
# show
##

show(io::IO, A::AbstractQuasiArray) = summary(io, A)

struct QuasiArrayLayout <: MemoryLayout end
MemoryLayout(::Type{<:AbstractQuasiArray}) = QuasiArrayLayout()
