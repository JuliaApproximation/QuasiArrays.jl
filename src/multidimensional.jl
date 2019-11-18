### Multidimensional iterators
module QuasiIteratorsMD
    import Base: eltype, length, size, first, last, in, getindex,
                 setindex!, IndexStyle, min, max, zero, oneunit, isless, eachindex,
                 ndims, IteratorSize, convert, show, iterate, promote_rule

    import Base: +, -, *, (:)
    import Base: simd_outer_range, simd_inner_length, simd_index
    using Base: IndexLinear, IndexCartesian, AbstractCartesianIndex, fill_to_length, tail
    using Base.Iterators: Reverse
    import QuasiArrays: AbstractQuasiArray, AbstractQuasiVector, domain, AbstractQuasiOrVector

    export QuasiCartesianIndex, QuasiCartesianIndices

    """
        QuasiCartesianIndex(i, j, k...)   -> I
        QuasiCartesianIndex((i, j, k...)) -> I

    Create a multidimensional index `I`, which can be used for
    indexing a multidimensional array `A`.  In particular, `A[I]` is
    equivalent to `A[i,j,k...]`.  One can freely mix integer and
    `QuasiCartesianIndex` indices; for example, `A[Ipre, i, Ipost]` (where
    `Ipre` and `Ipost` are `QuasiCartesianIndex` indices and `i` is an
    `Int`) can be a useful expression when writing algorithms that
    work along a single dimension of an array of arbitrary
    dimensionality.

    A `QuasiCartesianIndex` is sometimes produced by [`eachindex`](@ref), and
    always when iterating with an explicit [`QuasiCartesianIndices`](@ref).

    # Examples
    ```jldoctest
    julia> A = reshape(Vector(1:16), (2, 2, 2, 2))
    2×2×2×2 Array{Int64,4}:
    [:, :, 1, 1] =
     1  3
     2  4

    [:, :, 2, 1] =
     5  7
     6  8

    [:, :, 1, 2] =
      9  11
     10  12

    [:, :, 2, 2] =
     13  15
     14  16

    julia> A[QuasiCartesianIndex((1, 1, 1, 1))]
    1

    julia> A[QuasiCartesianIndex((1, 1, 1, 2))]
    9

    julia> A[QuasiCartesianIndex((1, 1, 2, 1))]
    5
    ```
    """
    struct QuasiCartesianIndex{N,II<:Tuple} <: AbstractCartesianIndex{N}
        I::II
        QuasiCartesianIndex{N,II}(index::II) where {N,II} = new(index)
    end
    QuasiCartesianIndex{N}(index::NTuple{N,Number}) where N = QuasiCartesianIndex{N,typeof(index)}(index)
    QuasiCartesianIndex(index::NTuple{N,Number}) where {N} = QuasiCartesianIndex{N}(index)
    QuasiCartesianIndex(index::Number...) = QuasiCartesianIndex(index)
    QuasiCartesianIndex{N}(index::Vararg{Number,N}) where {N} = QuasiCartesianIndex{N}(index)
    # Allow passing tuples smaller than N
    QuasiCartesianIndex{N}(index::Tuple) where {N} = QuasiCartesianIndex{N}(fill_to_length(index, 1, Val(N)))
    QuasiCartesianIndex{N}(index::Number...) where {N} = QuasiCartesianIndex{N}(index)
    QuasiCartesianIndex{N}() where {N} = QuasiCartesianIndex{N}(())
    # Un-nest passed QuasiCartesianIndexes
    QuasiCartesianIndex(index::Union{Number, QuasiCartesianIndex}...) = QuasiCartesianIndex(flatten(index))
    flatten(I::Tuple{}) = I
    flatten(I::Tuple{Any}) = I
    flatten(I::Tuple{<:QuasiCartesianIndex}) = I[1].I
    @inline flatten(I) = _flatten(I...)
    @inline _flatten() = ()
    @inline _flatten(i, I...)                 = (i, _flatten(I...)...)
    @inline _flatten(i::QuasiCartesianIndex, I...) = (i.I..., _flatten(I...)...)
    QuasiCartesianIndex(index::Tuple{Vararg{Union{Number, QuasiCartesianIndex}}}) = QuasiCartesianIndex(index...)
    show(io::IO, i::QuasiCartesianIndex) = (print(io, "QuasiCartesianIndex"); show(io, i.I))

    # length
    length(::QuasiCartesianIndex{N}) where {N} = N
    length(::Type{QuasiCartesianIndex{N}}) where {N} = N

    # indexing
    getindex(index::QuasiCartesianIndex, i::Integer) = index.I[i]
    Base.get(A::AbstractQuasiArray, I::QuasiCartesianIndex, default) = get(A, I.I, default)
    eltype(::Type{T}) where {T<:QuasiCartesianIndex} = eltype(fieldtype(T, :I))

    # access to index tuple
    Tuple(index::QuasiCartesianIndex) = index.I

    # equality
    Base.:(==)(a::QuasiCartesianIndex{N}, b::QuasiCartesianIndex{N}) where N = a.I == b.I

    # zeros and ones
    zero(::QuasiCartesianIndex{N}) where {N} = zero(QuasiCartesianIndex{N})
    zero(::Type{QuasiCartesianIndex{N}}) where {N} = QuasiCartesianIndex(ntuple(x -> 0, Val(N)))
    oneunit(::QuasiCartesianIndex{N}) where {N} = oneunit(QuasiCartesianIndex{N})
    oneunit(::Type{QuasiCartesianIndex{N}}) where {N} = QuasiCartesianIndex(ntuple(x -> 1, Val(N)))

    # arithmetic, min/max
    @inline (-)(index::QuasiCartesianIndex{N}) where {N} =
        QuasiCartesianIndex{N}(map(-, index.I))
    @inline (+)(index1::QuasiCartesianIndex{N}, index2::QuasiCartesianIndex{N}) where {N} =
        QuasiCartesianIndex{N}(map(+, index1.I, index2.I))
    @inline (-)(index1::QuasiCartesianIndex{N}, index2::QuasiCartesianIndex{N}) where {N} =
        QuasiCartesianIndex{N}(map(-, index1.I, index2.I))
    @inline min(index1::QuasiCartesianIndex{N}, index2::QuasiCartesianIndex{N}) where {N} =
        QuasiCartesianIndex{N}(map(min, index1.I, index2.I))
    @inline max(index1::QuasiCartesianIndex{N}, index2::QuasiCartesianIndex{N}) where {N} =
        QuasiCartesianIndex{N}(map(max, index1.I, index2.I))

    @inline (*)(a::Number, index::QuasiCartesianIndex{N}) where {N} = QuasiCartesianIndex{N}(map(x->a*x, index.I))
    @inline (*)(index::QuasiCartesianIndex, a::Number) = *(a,index)

    # comparison
    @inline isless(I1::QuasiCartesianIndex{N}, I2::QuasiCartesianIndex{N}) where {N} = _isless(0, I1.I, I2.I)
    @inline function _isless(ret, I1::NTuple{N,Int}, I2::NTuple{N,Int}) where N
        newret = ifelse(ret==0, icmp(I1[N], I2[N]), ret)
        _isless(newret, Base.front(I1), Base.front(I2))
    end
    _isless(ret, ::Tuple{}, ::Tuple{}) = ifelse(ret==1, true, false)
    icmp(a, b) = ifelse(isless(a,b), 1, ifelse(a==b, 0, -1))

    # conversions
    convert(::Type{T}, index::QuasiCartesianIndex{1}) where {T<:Number} = convert(T, index[1])
    convert(::Type{T}, index::QuasiCartesianIndex) where {T<:Tuple} = convert(T, index.I)

    # hashing
    const cartindexhash_seed = UInt == UInt64 ? 0xd60ca92f8284b8b0 : 0xf2ea7c2e
    function Base.hash(ci::QuasiCartesianIndex, h::UInt)
        h += cartindexhash_seed
        for i in ci.I
            h = hash(i, h)
        end
        return h
    end

    # nextind and prevind with QuasiCartesianIndex
    function Base.nextind(a::AbstractQuasiArray{<:Any,N}, i::QuasiCartesianIndex{N}) where {N}
        iter = QuasiCartesianIndices(axes(a))
        # might overflow
        I = inc(i.I, first(iter).I, last(iter).I)
        return I
    end
    function Base.prevind(a::AbstractQuasiArray{<:Any,N}, i::QuasiCartesianIndex{N}) where {N}
        iter = QuasiCartesianIndices(axes(a))
        # might underflow
        I = dec(i.I, last(iter).I, first(iter).I)
        return I
    end

    # Iteration over the elements of QuasiCartesianIndex cannot be supported until its length can be inferred,
    # see #23719
    Base.iterate(::QuasiCartesianIndex) =
        error("iteration is deliberately unsupported for QuasiCartesianIndex. Use `I` rather than `I...`, or use `Tuple(I)...`")

    # Iteration
    """
        QuasiCartesianIndices(sz::Dims) -> R
        QuasiCartesianIndices((istart:istop, jstart:jstop, ...)) -> R

    Define a region `R` spanning a multidimensional rectangular range
    of integer indices. These are most commonly encountered in the
    context of iteration, where `for I in R ... end` will return
    [`QuasiCartesianIndex`](@ref) indices `I` equivalent to the nested loops

        for j = jstart:jstop
            for i = istart:istop
                ...
            end
        end

    Consequently these can be useful for writing algorithms that
    work in arbitrary dimensions.

        QuasiCartesianIndices(A::AbstractQuasiArray) -> R

    As a convenience, constructing a `QuasiCartesianIndices` from an array makes a
    range of its indices.

    # Examples
    ```jldoctest
    julia> foreach(println, QuasiCartesianIndices((2, 2, 2)))
    QuasiCartesianIndex(1, 1, 1)
    QuasiCartesianIndex(2, 1, 1)
    QuasiCartesianIndex(1, 2, 1)
    QuasiCartesianIndex(2, 2, 1)
    QuasiCartesianIndex(1, 1, 2)
    QuasiCartesianIndex(2, 1, 2)
    QuasiCartesianIndex(1, 2, 2)
    QuasiCartesianIndex(2, 2, 2)

    julia> QuasiCartesianIndices(fill(1, (2,3)))
    2×3 QuasiCartesianIndices{2,Tuple{Base.OneTo{Int64},Base.OneTo{Int64}}}:
     QuasiCartesianIndex(1, 1)  QuasiCartesianIndex(1, 2)  QuasiCartesianIndex(1, 3)
     QuasiCartesianIndex(2, 1)  QuasiCartesianIndex(2, 2)  QuasiCartesianIndex(2, 3)
    ```

    ## Conversion between linear and QuasiCartesian indices

    Linear index to QuasiCartesian index conversion exploits the fact that a
    `QuasiCartesianIndices` is an `AbstractQuasiArray` and can be indexed linearly:

    ```jldoctest
    julia> QuasiCartesian = QuasiCartesianIndices((1:3, 1:2))
    3×2 QuasiCartesianIndices{2,Tuple{UnitRange{Int64},UnitRange{Int64}}}:
     QuasiCartesianIndex(1, 1)  QuasiCartesianIndex(1, 2)
     QuasiCartesianIndex(2, 1)  QuasiCartesianIndex(2, 2)
     QuasiCartesianIndex(3, 1)  QuasiCartesianIndex(3, 2)

    julia> cartesian[4]
    QuasiCartesianIndex(1, 2)
    ```

    ## Broadcasting

    `QuasiCartesianIndices` support broadcasting arithmetic (+ and -) with a `QuasiCartesianIndex`.

    !!! compat "Julia 1.1"
        Broadcasting of QuasiCartesianIndices requires at least Julia 1.1.

    ```jldoctest
    julia> CIs = QuasiCartesianIndices((2:3, 5:6))
    2×2 QuasiCartesianIndices{2,Tuple{UnitRange{Int64},UnitRange{Int64}}}:
     QuasiCartesianIndex(2, 5)  QuasiCartesianIndex(2, 6)
     QuasiCartesianIndex(3, 5)  QuasiCartesianIndex(3, 6)

    julia> CI = QuasiCartesianIndex(3, 4)
    QuasiCartesianIndex(3, 4)

    julia> CIs .+ CI
    2×2 QuasiCartesianIndices{2,Tuple{UnitRange{Int64},UnitRange{Int64}}}:
     QuasiCartesianIndex(5, 9)  QuasiCartesianIndex(5, 10)
     QuasiCartesianIndex(6, 9)  QuasiCartesianIndex(6, 10)
    ```

    For cartesian to linear index conversion, see [`LinearIndices`](@ref).
    """

    struct QuasiCartesianIndices{N,R<:NTuple{N,AbstractQuasiOrVector{<:Number}},RR<:NTuple{N,Number}} <: AbstractArray{QuasiCartesianIndex{N,RR},N}
        indices::R
    end

    QuasiCartesianIndices(nd::NTuple{N,AbstractQuasiOrVector{<:Number}}) where N = 
        QuasiCartesianIndices{N,typeof(nd),Tuple{map(eltype,nd)...}}(nd)

    QuasiCartesianIndices(::Tuple{}) = QuasiCartesianIndices{0,typeof(())}(())

    QuasiCartesianIndices(index::QuasiCartesianIndex) = QuasiCartesianIndices(index.I)
    QuasiCartesianIndices(sz::NTuple{N,<:Number}) where {N} = QuasiCartesianIndices(map(Base.OneTo, sz))

    QuasiCartesianIndices(A::AbstractQuasiArray) = QuasiCartesianIndices(axes(A))

    """
        (:)(I::QuasiCartesianIndex, J::QuasiCartesianIndex)

    Construct [`QuasiCartesianIndices`](@ref) from two `QuasiCartesianIndex`.

    !!! compat "Julia 1.1"
        This method requires at least Julia 1.1.

    # Examples
    ```jldoctest
    julia> I = QuasiCartesianIndex(2,1);

    julia> J = QuasiCartesianIndex(3,3);

    julia> I:J
    2×3 QuasiCartesianIndices{2,Tuple{UnitRange{Int64},UnitRange{Int64}}}:
     QuasiCartesianIndex(2, 1)  QuasiCartesianIndex(2, 2)  QuasiCartesianIndex(2, 3)
     QuasiCartesianIndex(3, 1)  QuasiCartesianIndex(3, 2)  QuasiCartesianIndex(3, 3)
    ```
    """
    (:)(I::QuasiCartesianIndex{N}, J::QuasiCartesianIndex{N}) where N =
        QuasiCartesianIndices(map((i,j) -> i:j, Tuple(I), Tuple(J)))

    promote_rule(::Type{QuasiCartesianIndices{N,R1}}, ::Type{QuasiCartesianIndices{N,R2}}) where {N,R1,R2} =
        QuasiCartesianIndices{N,Base.indices_promote_type(R1,R2)}

    convert(::Type{Tuple{}}, R::QuasiCartesianIndices{0}) = ()
    convert(::Type{NTuple{N,AbstractUnitRange{Int}}}, R::QuasiCartesianIndices{N}) where {N} =
        R.indices
    convert(::Type{NTuple{N,AbstractUnitRange}}, R::QuasiCartesianIndices{N}) where {N} =
        convert(NTuple{N,AbstractUnitRange{Int}}, R)
    convert(::Type{NTuple{N,UnitRange{Int}}}, R::QuasiCartesianIndices{N}) where {N} =
        UnitRange{Int}.(convert(NTuple{N,AbstractUnitRange}, R))
    convert(::Type{NTuple{N,UnitRange}}, R::QuasiCartesianIndices{N}) where {N} =
        UnitRange.(convert(NTuple{N,AbstractUnitRange}, R))
    convert(::Type{Tuple{Vararg{AbstractUnitRange{Int}}}}, R::QuasiCartesianIndices{N}) where {N} =
        convert(NTuple{N,AbstractUnitRange{Int}}, R)
    convert(::Type{Tuple{Vararg{AbstractUnitRange}}}, R::QuasiCartesianIndices) =
        convert(Tuple{Vararg{AbstractUnitRange{Int}}}, R)
    convert(::Type{Tuple{Vararg{UnitRange{Int}}}}, R::QuasiCartesianIndices{N}) where {N} =
        convert(NTuple{N,UnitRange{Int}}, R)
    convert(::Type{Tuple{Vararg{UnitRange}}}, R::QuasiCartesianIndices) =
        convert(Tuple{Vararg{UnitRange{Int}}}, R)

    convert(::Type{QuasiCartesianIndices{N,R}}, inds::QuasiCartesianIndices{N}) where {N,R} =
        QuasiCartesianIndices(convert(R, inds.indices))

    # AbstractQuasiArray implementation
    Base.axes(iter::QuasiCartesianIndices{N,R}) where {N,R} = map(Base.axes1, iter.indices)
    Base.IndexStyle(::Type{QuasiCartesianIndices{N,R}}) where {N,R} = IndexCartesian()
    @inline function Base.getindex(iter::QuasiCartesianIndices{N,R}, I::Vararg{Int, N}) where {N,R}
        @boundscheck checkbounds(iter, I...)
        QuasiCartesianIndex(getindex.(domain.(iter.indices), I))
    end

    ndims(R::QuasiCartesianIndices) = ndims(typeof(R))
    ndims(::Type{QuasiCartesianIndices{N}}) where {N} = N
    ndims(::Type{QuasiCartesianIndices{N,TT}}) where {N,TT} = N

    eachindex(::IndexCartesian, A::AbstractQuasiArray) = QuasiCartesianIndices(axes(A))

    @inline function eachindex(::IndexCartesian, A::AbstractQuasiArray, B::AbstractQuasiArray...)
        axsA = axes(A)
        Base._all_match_first(axes, axsA, B...) || Base.throw_eachindex_mismatch(IndexCartesian(), A, B...)
        QuasiCartesianIndices(axsA)
    end

    eltype(::Type{QuasiCartesianIndices{N}}) where {N} = QuasiCartesianIndex{N}
    eltype(::Type{QuasiCartesianIndices{N,TT}}) where {N,TT} = QuasiCartesianIndex{N}
    eltype(::Type{QuasiCartesianIndices{N,TT,RR}}) where {N,TT,RR} = QuasiCartesianIndex{N,RR}
    IteratorSize(::Type{<:QuasiCartesianIndices{N}}) where {N} = Base.HasShape{N}()

    size(iter::QuasiCartesianIndices) = map(length, iter.indices)
    length(iter::QuasiCartesianIndices) = prod(size(iter))

    first(iter::QuasiCartesianIndices) = QuasiCartesianIndex(map(first, iter.indices))
    last(iter::QuasiCartesianIndices)  = QuasiCartesianIndex(map(last, iter.indices))

    @inline function split(I::QuasiCartesianIndex, V::Val)
        i, j = split(I.I, V)
        QuasiCartesianIndex(i), QuasiCartesianIndex(j)
    end
    function split(R::QuasiCartesianIndices, V::Val)
        i, j = split(R.indices, V)
        QuasiCartesianIndices(i), QuasiCartesianIndices(j)
    end

    Base.LinearIndices(inds::QuasiCartesianIndices{N,R}) where {N,R} = LinearIndices{N,R}(inds.indices)
end  # IteratorsMD


using .QuasiIteratorsMD

## Bounds-checking with QuasiCartesianIndex
# Disallow linear indexing with QuasiCartesianIndex
function checkbounds(::Type{Bool}, A::AbstractQuasiArray, i::Union{QuasiCartesianIndex, AbstractArray{<:QuasiCartesianIndex}})
    @_inline_meta
    checkbounds_indices(Bool, axes(A), (i,))
end

@inline checkbounds_indices(::Type{Bool}, ::Tuple{}, I::Tuple{QuasiCartesianIndex,Vararg{Any}}) =
    checkbounds_indices(Bool, (), (I[1].I..., tail(I)...))
@inline checkbounds_indices(::Type{Bool}, IA::Tuple{Any}, I::Tuple{QuasiCartesianIndex,Vararg{Any}}) =
    checkbounds_indices(Bool, IA, (I[1].I..., tail(I)...))
@inline checkbounds_indices(::Type{Bool}, IA::Tuple, I::Tuple{QuasiCartesianIndex,Vararg{Any}}) =
    checkbounds_indices(Bool, IA, (I[1].I..., tail(I)...))

# Indexing into Array with mixtures of Numbers and QuasiCartesianIndices is
# extremely performance-sensitive. While the abstract fallbacks support this,
# codegen has extra support for SIMDification that sub2ind doesn't (yet) support
@propagate_inbounds getindex(A::Array, i1::Union{Number, QuasiCartesianIndex}, I::Union{Number, QuasiCartesianIndex}...) =
    A[to_indices(A, (i1, I...))...]
@propagate_inbounds setindex!(A::Array, v, i1::Union{Number, QuasiCartesianIndex}, I::Union{Number, QuasiCartesianIndex}...) =
    (A[to_indices(A, (i1, I...))...] = v; A)

# Support indexing with an array of QuasiCartesianIndex{N}s
# Here we try to consume N of the indices (if there are that many available)
# The first two simply handle ambiguities
@inline function checkbounds_indices(::Type{Bool}, ::Tuple{},
        I::Tuple{AbstractArray{QuasiCartesianIndex{N}},Vararg{Any}}) where N
    checkindex(Bool, (), I[1]) & checkbounds_indices(Bool, (), tail(I))
end
@inline function checkbounds_indices(::Type{Bool}, IA::Tuple{Any},
        I::Tuple{AbstractArray{QuasiCartesianIndex{0}},Vararg{Any}})
    checkbounds_indices(Bool, IA, tail(I))
end
@inline function checkbounds_indices(::Type{Bool}, IA::Tuple{Any},
        I::Tuple{AbstractArray{QuasiCartesianIndex{N}},Vararg{Any}}) where N
    checkindex(Bool, IA, I[1]) & checkbounds_indices(Bool, (), tail(I))
end
@inline function checkbounds_indices(::Type{Bool}, IA::Tuple,
        I::Tuple{AbstractArray{QuasiCartesianIndex{N}},Vararg{Any}}) where N
    IA1, IArest = IteratorsMD.split(IA, Val(N))
    checkindex(Bool, IA1, I[1]) & checkbounds_indices(Bool, IArest, tail(I))
end

function checkindex(::Type{Bool}, inds::Tuple, I::AbstractArray{<:QuasiCartesianIndex})
    b = true
    for i in I
        b &= checkbounds_indices(Bool, inds, (i,))
    end
    b
end

# combined count of all indices, including QuasiCartesianIndex and
# AbstractArray{QuasiCartesianIndex}
# rather than returning N, it returns an NTuple{N,Bool} so the result is inferrable
@inline function index_ndims(i1::QuasiCartesianIndex, I...)
    (map(x->true, i1.I)..., index_ndims(I...)...)
end
@inline function index_ndims(i1::AbstractArray{QuasiCartesianIndex{N}}, I...) where N
    (ntuple(x->true, Val(N))..., index_ndims(I...)...)
end

# In simple cases, we know that we don't need to use axes(A). Optimize those
# until Julia gets smart enough to elide the call on its own:
@inline to_indices(A, I::Tuple{Vararg{Union{Number, QuasiCartesianIndex}}}) = to_indices(A, (), I)
# But some index types require more context spanning multiple indices
# QuasiCartesianIndexes are simple; they just splat out
@inline to_indices(A, inds, I::Tuple{QuasiCartesianIndex, Vararg{Any}}) =
    to_indices(A, inds, (I[1].I..., tail(I)...))
# But for arrays of QuasiCartesianIndex, we just skip the appropriate number of inds
@inline function to_indices(A, inds, I::Tuple{AbstractArray{QuasiCartesianIndex{N}}, Vararg{Any}}) where N
    _, indstail = IteratorsMD.split(inds, Val(N))
    (to_index(A, I[1]), to_indices(A, indstail, tail(I))...)
end

# Colons get converted to slices by `uncolon`
const CI0 = Union{QuasiCartesianIndex{0}, AbstractArray{QuasiCartesianIndex{0}}}

### From abstractarray.jl: Internal multidimensional indexing definitions ###
getindex(x::Number, i::QuasiCartesianIndex{0}) = x
getindex(t::Tuple,  i::QuasiCartesianIndex{1}) = getindex(t, i.I[1])

@inline function _getindex(l::IndexStyle, A::AbstractQuasiArray, I::Union{Number, AbstractArray}...)
    @boundscheck checkbounds(A, I...)
    return _unsafe_getindex(l, _maybe_reshape(l, A, I...), I...)
end

@inline index_dimsum(::AbstractQuasiArray{Bool}, I...) = (true, index_dimsum(I...)...)
@inline function index_dimsum(::AbstractQuasiArray{<:Any,N}, I...) where N
    (ntuple(x->true, Val(N))..., index_dimsum(I...)...)
end

Slice(d::AbstractQuasiVector{<:Number}) = Inclusion(d)


_maybe_reshape(::IndexLinear, A::AbstractQuasiArray, I...) = A
_maybe_reshape(::IndexCartesian, A::AbstractQuasiVector, I...) = A
@inline _maybe_reshape(::IndexCartesian, A::AbstractQuasiArray, I...) = __maybe_reshape(A, index_ndims(I...))
@inline __maybe_reshape(A::AbstractQuasiArray{T,N}, ::NTuple{N,Any}) where {T,N} = A
@inline __maybe_reshape(A::AbstractQuasiArray, ::NTuple{N,Any}) where {N} = reshape(A, Val(N))

_unsafe_getindex(::IndexStyle, A::AbstractQuasiArray, I::Vararg{Union{Number, AbstractArray}, N}) where N =
    lazy_getindex(A, I...)
    
# Always index with the exactly indices provided.
@generated function _unsafe_getindex!(dest::Union{AbstractArray,AbstractQuasiArray}, src::AbstractQuasiArray, I::Vararg{Union{Number, AbstractArray}, N}) where N
    quote
        @_inline_meta
        D = eachindex(dest)
        Dy = iterate(D)
        @inbounds @nloops $N j d->I[d] begin
            # This condition is never hit, but at the moment
            # the optimizer is not clever enough to split the union without it
            Dy === nothing && return dest
            (idx, state) = Dy
            dest[idx] = @ncall $N getindex src j
            Dy = iterate(D, state)
        end
        return dest
    end
end


function fill!(A::AbstractQuasiArray{T}, x) where T
    xT = convert(T, x)
    for I in eachindex(A)
        @inbounds A[I] = xT
    end
    A
end
