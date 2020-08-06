# Used for when a lazy version should be constructed on materialize
abstract type AbstractQuasiArrayApplyStyle <: ApplyStyle end
struct QuasiArrayApplyStyle <: AbstractQuasiArrayApplyStyle end


##
# LazyQuasiArray
##

"""
A `LazyQuasiArray` is an abstract type for any array that is
stored and should be manipulated lazily.
"""
abstract type LazyQuasiArray{T,N} <: AbstractQuasiArray{T,N} end

const LazyQuasiVector{T} = LazyQuasiArray{T,1}
const LazyQuasiMatrix{T} = LazyQuasiArray{T,2}


struct LazyQuasiArrayApplyStyle <: AbstractQuasiArrayApplyStyle end

abstract type AbstractQuasiLazyLayout <: AbstractLazyLayout end
struct QuasiLazyLayout <: AbstractLazyLayout end

MemoryLayout(::Type{<:LazyQuasiArray}) = QuasiLazyLayout()
lazymaterialize(M::Mul{<:Any,<:Any,<:AbstractQuasiArray,<:AbstractQuasiArray}) = copy(ApplyQuasiArray(M))
lazymaterialize(M::Mul{<:Any,<:Any,<:AbstractArray,<:AbstractQuasiArray}) = copy(ApplyQuasiArray(M))
lazymaterialize(M::Mul{<:Any,<:Any,<:AbstractQuasiArray,<:AbstractArray}) = copy(ApplyQuasiArray(M))


###
# ApplyQuasiArray
###

"""
A `ApplyQuasiArray` is a lazy realization of a function that
creates a quasi-array.
"""
struct ApplyQuasiArray{T, N, F, Args<:Tuple} <: LazyQuasiArray{T,N}
    f::F
    args::Args
end

const ApplyQuasiVector{T, F, Args<:Tuple} = ApplyQuasiArray{T, 1, F, Args}
const ApplyQuasiMatrix{T, F, Args<:Tuple} = ApplyQuasiArray{T, 2, F, Args}

QuasiLazyArray(A::Applied) = ApplyQuasiArray(A)

ApplyQuasiArray{T,N,F,Args}(M::Applied) where {T,N,F,Args} = ApplyQuasiArray{T,N,F,Args}(M.f, M.args)
ApplyQuasiArray{T,N}(M::Applied{Style,F,Args}) where {T,N,Style,F,Args} = ApplyQuasiArray{T,N,F,Args}(instantiate(M))
ApplyQuasiArray{T}(M::Applied) where {T} = ApplyQuasiArray{T,ndims(M)}(M)
ApplyQuasiArray(M::Applied) = ApplyQuasiArray{eltype(M)}(M)
ApplyQuasiVector(M::Applied) = ApplyQuasiVector{eltype(M)}(M)
ApplyQuasiMatrix(M::Applied) = ApplyQuasiMatrix{eltype(M)}(M)

ApplyQuasiArray(f, factors...) = ApplyQuasiArray(applied(f, factors...))
ApplyQuasiArray{T}(f, factors...) where T = ApplyQuasiArray{T}(applied(f, factors...))
ApplyQuasiArray{T,N}(f, factors...) where {T,N} = ApplyQuasiArray{T,N}(applied(f, factors...))

ApplyQuasiVector(f, factors...) = ApplyQuasiVector(applied(f, factors...))
ApplyQuasiMatrix(f, factors...) = ApplyQuasiMatrix(applied(f, factors...))

@inline Applied(A::AbstractQuasiArray) = Applied(call(A), arguments(A)...)
@inline ApplyQuasiArray(A::AbstractQuasiArray) = ApplyQuasiArray(call(A), arguments(A)...)

axes(A::ApplyQuasiArray) = axes(Applied(A))
size(A::ApplyQuasiArray) = map(length, axes(A))
copy(A::ApplyQuasiArray) = A # immutable arrays don't need to copy

@propagate_inbounds getindex(A::ApplyQuasiArray{T,N}, kj::Vararg{Number,N}) where {T,N} =
    Applied(A)[kj...]

@propagate_inbounds getindex(A::ApplyQuasiArray{T,N}, kj::QuasiCartesianIndex{N}) where {T,N} =
    A[kj.I...]

MemoryLayout(M::Type{ApplyQuasiArray{T,N,F,Args}}) where {T,N,F,Args} = 
    applylayout(F, tuple_type_memorylayouts(Args)...)

copy(A::Applied{LazyQuasiArrayApplyStyle}) = ApplyQuasiArray(A)
copy(A::Applied{<:AbstractQuasiArrayApplyStyle}) = QuasiArray(A)
QuasiArray(A::Applied) = QuasiArray(ApplyQuasiArray(A))


####
# broadcasting
###

struct LazyQuasiArrayStyle{N} <: AbstractQuasiArrayStyle{N} end
LazyQuasiArrayStyle(::Val{N}) where N = LazyQuasiArrayStyle{N}()
LazyQuasiArrayStyle{M}(::Val{N}) where {N,M} = LazyQuasiArrayStyle{N}()
quasisubbroadcaststyle(::LazyQuasiArrayStyle{N}, _) where N = LazyQuasiArrayStyle{N}()
subbroadcaststyle(::LazyQuasiArrayStyle{N}, _) where N = LazyArrayStyle{N}()
subbroadcaststyle(::AbstractQuasiArrayStyle{N}, _) where N = DefaultArrayStyle{N}()


struct BroadcastQuasiArray{T, N, F, Args} <: LazyQuasiArray{T, N}
    f::F
    args::Args
end

const BroadcastQuasiVector{T,F,Args} = BroadcastQuasiArray{T,1,F,Args}
const BroadcastQuasiMatrix{T,F,Args} = BroadcastQuasiArray{T,2,F,Args}

LazyQuasiArray(bc::Broadcasted) = BroadcastQuasiArray(bc)

BroadcastQuasiArray{T,N,F,Args}(bc::Broadcasted) where {T,N,F,Args} = BroadcastQuasiArray{T,N,F,Args}(bc.f,bc.args)
BroadcastQuasiArray{T,N}(bc::Broadcasted{Style,Axes,F,Args}) where {T,N,Style,Axes,F,Args} = BroadcastQuasiArray{T,N,F,Args}(bc.f,bc.args)
BroadcastQuasiArray{T}(bc::Broadcasted{<:Union{Nothing,BroadcastStyle},<:Tuple{Vararg{Any,N}},<:Any,<:Tuple}) where {T,N} =
    BroadcastQuasiArray{T,N}(bc)

_broadcast2broadcastarray(a, b...) = tuple(a, b...)
_broadcast2broadcastarray(a::Broadcasted{<:LazyQuasiArrayStyle}, b...) = tuple(BroadcastQuasiArray(a), b...)

_BroadcastQuasiArray(bc::Broadcasted) = BroadcastQuasiArray{combine_eltypes(bc.f, bc.args)}(bc)
BroadcastQuasiArray(bc::Broadcasted{S}) where S =
    _BroadcastQuasiArray(instantiate(Broadcasted{S}(bc.f, _broadcast2broadcastarray(bc.args...))))
BroadcastQuasiArray(b::BroadcastQuasiArray) = b
BroadcastQuasiArray(f, A, As...) = BroadcastQuasiArray(broadcasted(f, A, As...))

Broadcasted(A::BroadcastQuasiArray) = instantiate(broadcasted(A.f, A.args...))


axes(A::BroadcastQuasiArray) = axes(Broadcasted(A))
size(A::BroadcastQuasiArray) = map(length, axes(A))

IndexStyle(::BroadcastQuasiArray{<:Any,1}) = IndexLinear()

@propagate_inbounds getindex(A::BroadcastQuasiArray, kj::Number...) = Broadcasted(A)[kj...]
@propagate_inbounds getindex(A::BroadcastQuasiArray{T,N}, kj::QuasiCartesianIndex{N}) where {T,N} =
    A[kj.I...]


@propagate_inbounds _broadcast_getindex_range(A::Union{Ref,AbstractQuasiArray{<:Any,0},Number}, I) = A[] # Scalar-likes can just ignore all indices
# Everything else falls back to dynamically dropping broadcasted indices based upon its axes
@propagate_inbounds _broadcast_getindex_range(A, I) = A[I]

getindex(B::BroadcastQuasiArray{<:Any,1}, kr::AbstractVector{<:Number}) =
    BroadcastArray(B.f, map(a -> _broadcast_getindex_range(a,kr), B.args)...)

copy(bc::Broadcasted{<:LazyQuasiArrayStyle}) = BroadcastQuasiArray(bc)


BroadcastStyle(::Type{<:LazyQuasiArray{<:Any,N}}) where N = LazyQuasiArrayStyle{N}()

MemoryLayout(M::Type{BroadcastQuasiArray{T,N,F,Args}}) where {T,N,F,Args} = 
    broadcastlayout(F, tuple_type_memorylayouts(Args)...)

arguments(b::BroadcastLayout, V::SubQuasiArray) = LazyArrays._broadcast_sub_arguments(V)

###
# *
###


ndims(M::Applied{LazyQuasiArrayApplyStyle,typeof(*)}) = ndims(last(M.args))

call(a::AbstractQuasiArray) = call(MemoryLayout(typeof(a)), a)
call(::ApplyLayout{typeof(*)}, V::SubQuasiArray) = *

arguments(a::AbstractQuasiArray) = arguments(MemoryLayout(typeof(a)), a)
arguments(::ApplyLayout{typeof(*)}, V::SubQuasiArray{<:Any,2}) = _mat_mul_arguments(V)
arguments(::ApplyLayout{typeof(*)}, V::SubQuasiArray{<:Any,1}) = _vec_mul_arguments(V)

ApplyQuasiArray(M::Mul) = ApplyQuasiArray(*, M.A, M.B)
QuasiArray(M::Mul) = QuasiArray(ApplyQuasiArray(M))

indextype(M::Mul{<:Any,<:Any,<:AbstractQuasiMatrix,<:AbstractVector}) = Tuple{eltype(axes(M.A,1))}
indextype(M::Mul{<:Any,<:Any,<:AbstractQuasiMatrix,<:AbstractMatrix}) = Tuple{eltype(axes(M.A,1)),Int}
indextype(M::Mul{<:Any,<:Any,<:AbstractQuasiVector,<:AbstractMatrix}) = Tuple{eltype(axes(M.A,1)),Int}
indextype(M::Mul{<:Any,<:Any,<:AbstractQuasiMatrix,<:AbstractQuasiVector}) = Tuple{eltype(axes(M.A,1))}
indextype(M::Mul{<:Any,<:Any,<:AbstractQuasiMatrix,<:AbstractQuasiMatrix}) = Tuple{eltype(axes(M.A,1)),eltype(axes(M.B,2))}
indextype(M::Mul{<:Any,<:Any,<:AbstractQuasiVector,<:AbstractQuasiMatrix}) = Tuple{eltype(axes(M.A,1)),eltype(axes(M.B,2))}
indextype(M::Mul{<:Any,<:Any,<:AbstractMatrix,<:AbstractQuasiVector}) = Tuple{Int}
indextype(M::Mul{<:Any,<:Any,<:AbstractMatrix,<:AbstractQuasiMatrix}) = Tuple{Int,eltype(axes(M.B,2))}
indextype(M::Mul{<:Any,<:Any,<:AbstractVector,<:AbstractQuasiMatrix}) = Tuple{Int,eltype(axes(M.B,2))}

###
# ^
###

ndims(::Type{<:Applied{<:Any,typeof(^),<:Tuple{<:AbstractQuasiMatrix,<:Number}}}) = 2
ndims(::Applied{<:Any,typeof(^),<:Tuple{<:AbstractQuasiMatrix,<:Number}}) = 2
size(A::Applied{<:Any,typeof(^),<:Tuple{<:AbstractQuasiMatrix,<:Number}}) = size(A.args[1])
axes(A::Applied{<:Any,typeof(^),<:Tuple{<:AbstractQuasiMatrix,<:Number}}) = axes(A.args[1])
eltype(::Applied{<:Any,typeof(^),<:Tuple{<:AbstractQuasiMatrix{T},<:Integer}}) where T = T
eltype(::Applied{<:Any,typeof(^),<:Tuple{<:AbstractQuasiMatrix{T},<:Number}}) where T = complex(T)

function *(App::ApplyQuasiMatrix{<:Any,typeof(^),<:Tuple{<:AbstractQuasiMatrix{T},<:Integer}}, b::AbstractQuasiMatrix) where T
    A,p = arguments(App)
    p < 0 && return ApplyQuasiMatrix(^,inv(A),-p)*b
    p == 0 && return copy(b)
    return A*(ApplyQuasiMatrix(^,A,p-1)*b)
end