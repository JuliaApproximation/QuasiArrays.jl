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
lazymaterialize(F, args::Union{AbstractQuasiArray,AbstractArray}...) = copy(ApplyQuasiArray(F, args...))
concretize(A::AbstractQuasiArray) = convert(QuasiArray, A)


# Inclusions are left lazy. This could be refined to only be the case where the cardinality is infinite
BroadcastStyle(::Type{<:Inclusion}) = LazyQuasiArrayStyle{1}()


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

@propagate_inbounds _getindex(::Type{IND}, A::ApplyQuasiArray, I::IND) where IND =
    Applied(A)[I...]

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

## TODO: Generalise
subbroadcaststyle(::LazyQuasiArrayStyle{2}, ::Type{<:Tuple{Number,Number}}) = DefaultArrayStyle{0}()
subbroadcaststyle(::LazyQuasiArrayStyle{2}, ::Type{<:Tuple{Number,JR}}) where JR = Base.BroadcastStyle(JR)
subbroadcaststyle(::LazyQuasiArrayStyle{2}, ::Type{<:Tuple{KR,Number}}) where KR = Base.BroadcastStyle(KR)


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

_broadcast2broadcastarray(a::Broadcasted{<:LazyQuasiArrayStyle}, b...) = tuple(BroadcastQuasiArray(a), _broadcast2broadcastarray(b...)...)

_BroadcastQuasiArray(bc::Broadcasted) = BroadcastQuasiArray{combine_eltypes(bc.f, bc.args)}(bc)
BroadcastQuasiArray(bc::Broadcasted{S}) where S =
    _BroadcastQuasiArray(instantiate(Broadcasted{S}(bc.f, _broadcast2broadcastarray(bc.args...))))
BroadcastQuasiArray(b::BroadcastQuasiArray) = b
BroadcastQuasiArray(f, A, As...) = BroadcastQuasiArray(instantiate(broadcasted(f, A, As...)))
BroadcastQuasiVector(f, A, As...) = BroadcastQuasiVector{combine_eltypes(f, (A, As...))}(f, A, As...)
BroadcastQuasiMatrix(f, A, As...) = BroadcastQuasiMatrix{combine_eltypes(f, (A, As...))}(f, A, As...)
BroadcastQuasiArray{T}(f, A, As...) where {T} = BroadcastQuasiArray{T}(instantiate(broadcasted(f, A, As...)))
BroadcastQuasiArray{T,N}(f, A, As...) where {T,N} = BroadcastQuasiArray{T,N,typeof(f),typeof((A, As...))}(f, (A, As...))

@inline BroadcastQuasiArray(A::AbstractQuasiArray) = BroadcastQuasiArray(call(A), arguments(A)...)

broadcasted(A::BroadcastQuasiArray) = instantiate(broadcasted(A.f, A.args...))

axes(A::BroadcastQuasiArray) = axes(broadcasted(A))
size(A::BroadcastQuasiArray) = map(length, axes(A))

function ==(A::BroadcastQuasiArray, B::BroadcastQuasiArray)
    A.f == B.f && all(A.args .== B.args) && return true
    error("Not implemented")
end
copy(A::BroadcastQuasiArray) = A # BroadcastQuasiArray are immutable

@propagate_inbounds _getindex(::Type{IND}, bc::BroadcastQuasiArray, kj::IND) where IND = bc[QuasiCartesianIndex(kj...)]
@propagate_inbounds function getindex(bc::BroadcastQuasiArray{T,N}, kj::QuasiCartesianIndex{N}) where {T,N}
    args = Base.Broadcast._getindex(bc.args, kj)
    Base.Broadcast._broadcast_getindex_evalf(bc.f, args...)
end


copy(bc::Broadcasted{<:LazyQuasiArrayStyle}) = BroadcastQuasiArray(bc)


BroadcastStyle(::Type{<:LazyQuasiArray{<:Any,N}}) where N = LazyQuasiArrayStyle{N}()

MemoryLayout(M::Type{BroadcastQuasiArray{T,N,F,Args}}) where {T,N,F,Args} =
    broadcastlayout(F, tuple_type_memorylayouts(Args)...)

arguments(b::BroadcastLayout, V::SubQuasiArray) = LazyArrays._broadcast_sub_arguments(V)
call(b::BroadcastLayout, a::SubQuasiArray) = call(b, parent(a))


###
# show
####


summary(io::IO, A::BroadcastQuasiArray) = _broadcastarray_summary(io, A)
summary(io::IO, A::ApplyQuasiArray) = _applyarray_summary(io, A)

for op in (:+, :-, :*, :\, :/)
    @eval begin
        function summary(io::IO, A::BroadcastQuasiArray{<:Any,N,typeof($op)}) where N
            args = arguments(A)
            if length(args) == 1
                print(io, "($($op)).(")
                summary(io, first(args))
                print(io, ")")
            else
                summary(io, first(args))
                for a in tail(args)
                    print(io, " .$($op) ")
                    summary(io, a)
                end
            end
        end
    end
end

show(io::IO, ::MIME"text/plain", A::BroadcastQuasiArray) = show(io, A)

###
# *
###

# a .* (B * C) flattens to (a .* B) * C
__broadcast_mul_arguments(a, B, C...) = (a .* B, C...)
_broadcast_mul_arguments(a, B) = __broadcast_mul_arguments(a, _mul_arguments(B)...)
_mul_arguments(A::BroadcastQuasiMatrix{<:Any,typeof(*),<:Tuple{AbstractQuasiVector,AbstractQuasiMatrix}}) =
    _broadcast_mul_arguments(A.args...)

ndims(M::Applied{LazyQuasiArrayApplyStyle,typeof(*)}) = ndims(last(M.args))

call(a::AbstractQuasiArray) = call(MemoryLayout(typeof(a)), a)
call(::ApplyLayout{typeof(*)}, V::SubQuasiArray) = *

arguments(a::AbstractQuasiArray) = arguments(MemoryLayout(typeof(a)), a)
arguments(::ApplyLayout{typeof(*)}, V::SubQuasiArray{<:Any,2}) = _mat_mul_arguments(V)
arguments(::ApplyLayout{typeof(*)}, V::SubQuasiArray{<:Any,1}) = _vec_mul_arguments(V)

ApplyQuasiArray(M::Mul) = ApplyQuasiArray(*, M.A, M.B)
QuasiArray(M::Mul) = QuasiArray(ApplyQuasiArray(M))

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
