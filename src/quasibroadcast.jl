axistype(a::AbstractQuasiVector{<:Number}, b::AbstractQuasiVector{<:Number}) = a
axistype(a::AbstractQuasiVector{<:Number}, b::AbstractVector{<:Number}) = a


"""
`Broadcast.AbstractQuasiArrayStyle{N} <: BroadcastStyle` is the abstract supertype for any style
associated with an `AbstractQuasiArray` type.
The `N` parameter is the dimensionality, which can be handy for AbstractQuasiArray types
that only support specific dimensionalities:

    struct SparseMatrixStyle <: Broadcast.AbstractQuasiArrayStyle{2} end
    Base.BroadcastStyle(::Type{<:SparseMatrixCSC}) = SparseMatrixStyle()

For `AbstractQuasiArray` types that support arbitrary dimensionality, `N` can be set to `Any`:

    struct MyQuasiArrayStyle <: Broadcast.AbstractQuasiArrayStyle{Any} end
    Base.BroadcastStyle(::Type{<:MyQuasiArray}) = MyQuasiArrayStyle()

In cases where you want to be able to mix multiple `AbstractQuasiArrayStyle`s and keep track
of dimensionality, your style needs to support a [`Val`](@ref) constructor:

    struct MyQuasiArrayStyleDim{N} <: Broadcast.AbstractQuasiArrayStyle{N} end
    (::Type{<:MyQuasiArrayStyleDim})(::Val{N}) where N = MyQuasiArrayStyleDim{N}()

Note that if two or more `AbstractQuasiArrayStyle` subtypes conflict, broadcasting machinery
will fall back to producing `QuasiArray`s. If this is undesirable, you may need to
define binary [`BroadcastStyle`](@ref) rules to control the output type.

See also [`Broadcast.DefaultQuasiArrayStyle`](@ref).
"""
abstract type AbstractQuasiArrayStyle{N} <: BroadcastStyle end

"""
`Broadcast.QuasiArrayStyle{MyQuasiArrayType}()` is a [`BroadcastStyle`](@ref) indicating that an object
behaves as an array for broadcasting. It presents a simple way to construct
[`Broadcast.AbstractQuasiArrayStyle`](@ref)s for specific `AbstractQuasiArray` container types.
Broadcast styles created this way lose track of dimensionality; if keeping track is important
for your type, you should create your own custom [`Broadcast.AbstractQuasiArrayStyle`](@ref).
"""
struct QuasiArrayStyle{A<:AbstractQuasiArray} <: AbstractQuasiArrayStyle{Any} end
QuasiArrayStyle{A}(::Val) where A = QuasiArrayStyle{A}()

"""
`Broadcast.DefaultQuasiArrayStyle{N}()` is a [`BroadcastStyle`](@ref) indicating that an object
behaves as an `N`-dimensional array for broadcasting. Specifically, `DefaultQuasiArrayStyle` is
used for any
`AbstractQuasiArray` type that hasn't defined a specialized style, and in the absence of
overrides from other `broadcast` arguments the resulting output type is `QuasiArray`.
When there are multiple inputs to `broadcast`, `DefaultQuasiArrayStyle` "loses" to any other [`Broadcast.QuasiArrayStyle`](@ref).
"""
struct DefaultQuasiArrayStyle{N} <: AbstractQuasiArrayStyle{N} end
DefaultQuasiArrayStyle(::Val{N}) where N = DefaultQuasiArrayStyle{N}()
DefaultQuasiArrayStyle{M}(::Val{N}) where {N,M} = DefaultQuasiArrayStyle{N}()
const DefaultQuasiVectorStyle = DefaultQuasiArrayStyle{1}
const DefaultQuasiMatrixStyle = DefaultQuasiArrayStyle{2}
BroadcastStyle(::Type{<:AbstractQuasiArray{T,N}}) where {T,N} = DefaultQuasiArrayStyle{N}()

# `QuasiArrayConflict` is an internal type signaling that two or more different `AbstractQuasiArrayStyle`
# objects were supplied as arguments, and that no rule was defined for resolving the
# conflict. The resulting output is `QuasiArray`. While this is the same output type
# produced by `DefaultQuasiArrayStyle`, `QuasiArrayConflict` "poisons" the BroadcastStyle so that
# 3 or more arguments still return an `QuasiArrayConflict`.
struct QuasiArrayConflict <: AbstractQuasiArrayStyle{Any} end
QuasiArrayConflict(::Val) = QuasiArrayConflict()


BroadcastStyle(a::AbstractQuasiArrayStyle{0}, b::Style{Tuple}) = b
BroadcastStyle(a::AbstractQuasiArrayStyle, ::Style{Tuple})    = a
BroadcastStyle(::A, ::A) where A<:QuasiArrayStyle             = A()
BroadcastStyle(::QuasiArrayStyle, ::QuasiArrayStyle)               = Unknown()
BroadcastStyle(::A, ::A) where A<:AbstractQuasiArrayStyle     = A()
Base.@pure function BroadcastStyle(a::A, b::B) where {A<:AbstractQuasiArrayStyle{M},B<:AbstractQuasiArrayStyle{N}} where {M,N}
    if Base.typename(A) === Base.typename(B)
        return A(Val(max(M, N)))
    end
    return Unknown()
end
# Any specific array type beats DefaultQuasiArrayStyle
BroadcastStyle(a::AbstractQuasiArrayStyle{Any}, ::DefaultQuasiArrayStyle) = a
BroadcastStyle(a::AbstractQuasiArrayStyle{N}, ::DefaultArrayStyle{0}) where N = a
BroadcastStyle(a::AbstractQuasiArrayStyle{N}, ::DefaultQuasiArrayStyle{N}) where N = a
BroadcastStyle(a::AbstractQuasiArrayStyle{M}, ::DefaultQuasiArrayStyle{N}) where {M,N} =
    typeof(a)(Val(max(M, N)))

Base.similar(bc::Broadcasted{DefaultQuasiArrayStyle{N}}, ::Type{ElType}) where {N,ElType} =
    similar(QuasiArray{ElType}, axes(bc))
# In cases of conflict we fall back on Array
Base.similar(bc::Broadcasted{QuasiArrayConflict}, ::Type{ElType}) where ElType =
    similar(QuasiArray{ElType}, axes(bc))

_axes(bc::Broadcasted{<:AbstractQuasiArrayStyle{0}}, ::Nothing) = ()

_eachindex(t::Tuple{AbstractQuasiVector{<:Number}}) = QuasiCartesianIndices(t)
_eachindex(t::NTuple{N,AbstractQuasiVector{<:Number}}) where N = QuasiCartesianIndices(t)
_eachindex(t::Tuple{AbstractQuasiVector{<:Number},Vararg{AbstractUnitRange}}) = QuasiCartesianIndices(t)
_eachindex(t::Tuple{AbstractUnitRange,AbstractQuasiVector{<:Number},Vararg{AbstractUnitRange}}) = QuasiCartesianIndices(t)

instantiate(bc::Broadcasted{<:AbstractQuasiArrayStyle{0}}) = bc

result_join(::AbstractQuasiArrayStyle, ::AbstractQuasiArrayStyle, ::Unknown, ::Unknown) =
    QuasiArrayConflict()

Base.@propagate_inbounds _newindex(ax::Tuple, I::Tuple) = (ifelse(Base.unsafe_length(ax[1])==1, first(ax[1]), I[1]), _newindex(tail(ax), tail(I))...)
Base.@propagate_inbounds _newindex(ax::Tuple{}, I::Tuple) = ()
Base.@propagate_inbounds _newindex(ax::Tuple, I::Tuple{}) = (first(ax[1]), _newindex(tail(ax), ())...)
Base.@propagate_inbounds _newindex(ax::Tuple{}, I::Tuple{}) = ()
@inline _newindex(I, keep, Idefault) =
    (ifelse(keep[1], I[1], Idefault[1]), _newindex(tail(I), tail(keep), tail(Idefault))...)
@inline _newindex(I, keep::Tuple{}, Idefault) = ()  # truncate if keep is shorter than I
# for now we assume indexing is simple
Base.@propagate_inbounds newindex(arg, I::QuasiCartesianIndex) = QuasiCartesianIndex(_newindex(axes(arg), I.I))
@inline newindex(I::QuasiCartesianIndex, keep, Idefault) = QuasiCartesianIndex(_newindex(I.I, keep, Idefault))

@inline function Base.getindex(bc::Broadcasted, I::QuasiCartesianIndex)
    @boundscheck checkbounds(bc, I)
    @inbounds _broadcast_getindex(bc, I)
end

Base.@propagate_inbounds Base.getindex(bc::Broadcasted{<:AbstractQuasiArrayStyle}, i1::Number, I::Number...) = bc[QuasiCartesianIndex((i1, I...))]
Base.@propagate_inbounds Base.getindex(bc::Broadcasted{<:AbstractQuasiArrayStyle}) = bc[QuasiCartesianIndex(())]

@inline Base.checkbounds(bc::Broadcasted{<:AbstractQuasiArrayStyle}, I::Number) =
    Base.checkbounds_indices(Bool, axes(bc), (I,)) || Base.throw_boundserror(bc, (I,))

@inline Base.checkbounds(bc::Broadcasted, I::QuasiCartesianIndex) =
    Base.checkbounds_indices(Bool, axes(bc), (I,)) || Base.throw_boundserror(bc, (I,))

Base.@propagate_inbounds _broadcast_getindex(A::AbstractQuasiArray{<:Any,0}, I) = A[] # Scalar-likes can just ignore all indices

extrude(x::AbstractQuasiArray) = Extruded(x, newindexer(x)...)

broadcastable(x::AbstractQuasiArray) = x

@inline copy(bc::Broadcasted{<:AbstractQuasiArrayStyle{0}}) = bc[CartesianIndex()]

@inline copyto!(dest::AbstractQuasiArray, bc::Broadcasted) = copyto!(dest, convert(Broadcasted{Nothing}, bc))

# Specialize this method if all you want to do is specialize on typeof(dest)
@inline function copyto!(dest::AbstractQuasiArray, bc::Broadcasted{Nothing})
    axes(dest) == axes(bc) || throwdm(axes(dest), axes(bc))
    # Performance optimization: broadcast!(identity, dest, A) is equivalent to copyto!(dest, A) if indices match
    if bc.f === identity && bc.args isa Tuple{AbstractQuasiArray} # only a single input argument to broadcast!
        A = bc.args[1]
        if axes(dest) == axes(A)
            return copyto!(dest, A)
        end
    end
    bc′ = preprocess(dest, bc)
    @simd for I in eachindex(bc′)
        @inbounds dest[I] = bc′[I]
    end
    return dest
end

## QuasiCartesianIndices
broadcasted(::typeof(+), I::QuasiCartesianIndices{N}, j::QuasiCartesianIndex{N}) where N =
    QuasiCartesianIndices(map((rng, offset)->rng .+ offset, I.indices, Tuple(j)))
broadcasted(::typeof(+), j::QuasiCartesianIndex{N}, I::QuasiCartesianIndices{N}) where N =
    I .+ j
broadcasted(::typeof(-), I::QuasiCartesianIndices{N}, j::QuasiCartesianIndex{N}) where N =
    QuasiCartesianIndices(map((rng, offset)->rng .- offset, I.indices, Tuple(j)))
function broadcasted(::typeof(-), j::QuasiCartesianIndex{N}, I::QuasiCartesianIndices{N}) where N
    diffrange(offset, rng) = range(offset-last(rng), length=length(rng))
    Iterators.reverse(QuasiCartesianIndices(map(diffrange, Tuple(j), I.indices)))
end

##
# SubQuasiArray
##
quasisubbroadcaststyle(::AbstractQuasiArrayStyle{N}, _) where N = DefaultQuasiArrayStyle{N}()
BroadcastStyle(::Type{<:SubQuasiArray{T,N,P,I}}) where {T,N,P,I} = quasisubbroadcaststyle(BroadcastStyle(P), I)
BroadcastStyle(::Type{<:SubArray{T,N,P,I}}) where {T,N,P<:AbstractQuasiArray,I} = subbroadcaststyle(BroadcastStyle(P), I)
