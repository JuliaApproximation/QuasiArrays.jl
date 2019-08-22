# Used for when a lazy version should be constructed on materialize
abstract type AbstractQuasiArrayApplyStyle <: ApplyStyle end
struct QuasiArrayApplyStyle <: AbstractQuasiArrayApplyStyle end



const QuasiArrayMulArray{p, q, T, V} =
    Applied{<:Any, typeof(*), <:Tuple{<:AbstractQuasiArray{T,p}, <:AbstractArray{V,q}}}

const ArrayMulQuasiArray{p, q, T, V} =
    Applied{<:Any, typeof(*), <:Tuple{<:AbstractArray{T,p}, <:AbstractQuasiArray{V,q}}}

const QuasiArrayMulQuasiArray{p, q, T, V} =
    Applied{<:Any, typeof(*), <:Tuple{<:AbstractQuasiArray{T,p}, <:AbstractQuasiArray{V,q}}}
####
# Matrix * Vector
####
const QuasiMatMulVec{T, V} = QuasiArrayMulArray{2, 1, T, V}
const QuasiMatMulMat{T, V} = QuasiArrayMulArray{2, 2, T, V}
const QuasiMatMulQuasiMat{T, V} = QuasiArrayMulQuasiArray{2, 2, T, V}


function getindex(M::Mul{<:AbstractQuasiArrayApplyStyle}, k::Real)
    A,Bs = first(M.args), tail(M.args)
    B = _mul(Bs...)
    ret = zero(eltype(M))
    for j = rowsupport(A, k) ∩ colsupport(B,1)
        ret += A[k,j] * B[j]
    end
    ret
end

function _mul_quasi_getindex(M::Mul, k::Real, j::Real)
    A,Bs = first(M.args), tail(M.args)
    B = _mul(Bs...)
    ret = zero(eltype(M))
    @inbounds for ℓ in (rowsupport(A,k) ∩ colsupport(B,j))
        ret += A[k,ℓ] * B[ℓ,j]
    end
    ret
end

getindex(M::Mul{<:AbstractQuasiArrayApplyStyle}, k::Real, j::Real) =
    _mul_quasi_getindex(M, k, j)

getindex(M::Mul{<:AbstractQuasiArrayApplyStyle}, k::Integer, j::Integer) =
    _mul_quasi_getindex(M, k, j)


function getindex(M::QuasiMatMulVec, k::AbstractArray)
    A,B = M.args
    ret = zeros(eltype(M),length(k))
    @inbounds for j in axes(A,2)
        ret .+= view(A,k,j) .* B[j]
    end
    ret
end


*(A::AbstractQuasiArray, B...) = fullmaterialize(apply(*,A,B...))
*(A::AbstractQuasiArray, B::AbstractQuasiArray, C...) = fullmaterialize(apply(*,A,B,C...))
*(A::AbstractArray, B::AbstractQuasiArray, C...) = fullmaterialize(apply(*,A,B,C...))

for op in (:pinv, :inv)
    @eval $op(A::AbstractQuasiArray) = fullmaterialize(apply($op,A))
end

axes(L::Ldiv{<:Any,<:Any,<:AbstractQuasiMatrix}) =
    (axes(L.args[1], 2),axes(L.args[2],2))
axes(L::Ldiv{<:Any,<:Any,<:AbstractQuasiVector}) =
    (axes(L.args[1], 2),)    

 \(A::AbstractQuasiArray, B::AbstractQuasiArray) = apply(\,A,B)


*(A::AbstractQuasiArray, B::Mul, C...) = apply(*,A, B.args..., C...)
*(A::Mul, B::AbstractQuasiArray, C...) = apply(*,A.args..., B, C...)

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
struct LazyLayout <: MemoryLayout end

MemoryLayout(::Type{<:LazyQuasiArray}) = LazyLayout()
ndims(M::Applied{LazyQuasiArrayApplyStyle,typeof(*)}) = ndims(last(M.args))

transposelayout(::LazyLayout) = LazyLayout()
conjlayout(::LazyLayout) = LazyLayout()
quasimulapplystyle(::LazyLayout, ::LazyLayout, lay...) = LazyQuasiArrayApplyStyle()
quasimulapplystyle(_, ::LazyLayout, lay...) = LazyQuasiArrayApplyStyle()
quasimulapplystyle(::LazyLayout, lay...) = LazyQuasiArrayApplyStyle()
quasimulapplystyle(::LazyLayout, ::LazyLayout, ::LazyLayout, lay...) = LazyQuasiArrayApplyStyle()
quasimulapplystyle(_, ::LazyLayout, ::LazyLayout, lay...) = LazyQuasiArrayApplyStyle()
quasimulapplystyle(::LazyLayout, _, ::LazyLayout, lay...) = LazyQuasiArrayApplyStyle()
quasimulapplystyle(_, _, ::LazyLayout, lay...) = LazyQuasiArrayApplyStyle()



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

function getproperty(A::ApplyQuasiArray, d::Symbol)
    if d == :applied
        applied(A.f, A.args...)
    else
        getfield(A, d)
    end
end

axes(A::ApplyQuasiArray) = axes(A.applied)
size(A::ApplyQuasiArray) = map(length, axes(A))
copy(A::ApplyQuasiArray) = copy(A.applied)

IndexStyle(::ApplyQuasiArray{<:Any,1}) = IndexLinear()

@propagate_inbounds getindex(A::ApplyQuasiArray{T,N}, kj::Vararg{Real,N}) where {T,N} =
    A.applied[kj...]

MemoryLayout(M::Type{ApplyQuasiArray{T,N,F,Args}}) where {T,N,F,Args} = ApplyLayout{F,tuple_type_memorylayouts(Args)}()

materialize(M::Applied{<:AbstractQuasiArrayApplyStyle}) = _materialize(instantiate(M), axes(M))
_materialize(A::Applied{<:AbstractQuasiArrayApplyStyle}, _) = copy(instantiate(A))


copy(A::Applied{<:AbstractQuasiArrayApplyStyle}) = QuasiArray(A)
copy(A::Applied{LazyQuasiArrayApplyStyle}) = ApplyQuasiArray(A)
QuasiArray(A::Applied) = QuasiArray(ApplyQuasiArray(A))


####
# MulQuasiArray
#####

const MulQuasiArray{T, N, Args<:Tuple} = ApplyQuasiArray{T, N, typeof(*), Args}

const MulQuasiVector{T, Args<:Tuple} = MulQuasiArray{T, 1, Args}
const MulQuasiMatrix{T, Args<:Tuple} = MulQuasiArray{T, 2, Args}

const Vec = MulQuasiVector


_ApplyArray(F, factors...) = ApplyQuasiArray(F, factors...)
_ApplyArray(F, factors::AbstractArray...) = ApplyArray(F, factors...)

most(a) = reverse(tail(reverse(a)))

MulQuasiOrArray = Union{MulArray,MulQuasiArray}

_factors(M::MulQuasiOrArray) = M.applied.args
_factors(M) = (M,)

_flatten(A::MulQuasiArray, B...) = _flatten(A.applied, B...)
flatten(A::MulQuasiArray) = ApplyQuasiArray(flatten(A.applied))


function fullmaterialize(M::Applied{<:Any,typeof(*)})
    M_mat = materialize(flatten(M))
    typeof(M_mat) <: MulQuasiOrArray || return M_mat
    typeof(M_mat.applied) == typeof(M) || return(fullmaterialize(M_mat))

    ABC = M_mat.applied.args
    length(ABC) ≤ 2 && return flatten(M_mat)

    AB = most(ABC)
    Mhead = fullmaterialize(Mul(AB...))

    typeof(_factors(Mhead)) == typeof(AB) ||
        return fullmaterialize(Mul(_factors(Mhead)..., last(ABC)))

    BC = tail(ABC)
    Mtail =  fullmaterialize(Mul(BC...))
    typeof(_factors(Mtail)) == typeof(BC) ||
        return fullmaterialize(Mul(first(ABC), _factors(Mtail)...))

    apply(*,first(ABC), Mtail.applied.args...)
end

fullmaterialize(M::ApplyQuasiArray) = flatten(fullmaterialize(M.applied))
fullmaterialize(M) = flatten(M)


adjoint(A::MulQuasiArray) = ApplyQuasiArray(*, reverse(adjoint.(A.applied.args))...)
transpose(A::MulQuasiArray) = ApplyQuasiArray(*, reverse(transpose.(A.applied.args))...)

function similar(A::MulQuasiArray)
    B,a = A.applied.args
    B*similar(a)
end

function similar(A::QuasiArrayMulArray)
    B,a = A.args
    applied(*, B, similar(a))
end

function copy(a::MulQuasiArray)
    @_propagate_inbounds_meta
    copymutable(a)
end

function copyto!(dest::MulQuasiArray, src::MulQuasiArray)
    d = last(dest.applied.args)
    s = last(src.applied.args)
    copyto!(IndexStyle(d), d, IndexStyle(s), s)
    dest
end




_mul_tail_support(j, Z) = maximum(last.(colsupport.(Ref(Z),j)))
_mul_tail_support(j, Z, Y, X...) = _mul_tail_support(OneTo(_mul_tail_support(j,Z)), Y, X...)

function _mul_getindex(k, j, A, B) 
    M = min(_mul_tail_support(j,B), maximum(last.(rowsupport.(Ref(A),k))))
    A[k,1:M]*B[1:M,j]
end

function _mul_getindex(k, j, A, B, C, D...) 
    N = _mul_tail_support(j, reverse(D)..., C, B)
    M = min(maximum(last.(rowsupport.(Ref(A),k))), N)
    _mul_getindex(OneTo(N), j, A[k,OneTo(M)]*B[OneTo(M),OneTo(N)], C, D...)
end

getindex(A::MulQuasiMatrix, k::AbstractVector{<:Real}, j::AbstractVector{<:Real}) = 
    _mul_getindex(k, j, A.args...)
 


quasimulapplystyle(_...) = QuasiArrayApplyStyle()

ApplyStyle(::typeof(*), ::Type{A}, ::Type{B}, C::Type...) where {A<:AbstractQuasiArray,B<:Union{AbstractArray,AbstractQuasiArray}} = quasimulapplystyle(MemoryLayout(A), MemoryLayout(B), MemoryLayout.(C)...)
ApplyStyle(::typeof(*), ::Type{A}, ::Type{B}, C::Type...) where {A<:AbstractArray,B<:AbstractQuasiArray} = quasimulapplystyle(MemoryLayout(A), MemoryLayout(B), MemoryLayout.(C)...)
ApplyStyle(::typeof(*), ::Type{A}, ::Type{B}, ::Type{C}, D::Type...) where {A<:AbstractQuasiArray,B<:Union{AbstractArray,AbstractQuasiArray},C<:Union{AbstractArray,AbstractQuasiArray}} = 
    quasimulapplystyle(MemoryLayout(A), MemoryLayout(B), MemoryLayout(C), MemoryLayout.(D)...)
ApplyStyle(::typeof(*), ::Type{A}, ::Type{B}, ::Type{C}, D::Type...) where {A<:AbstractArray,B<:AbstractQuasiArray,C<:Union{AbstractArray,AbstractQuasiArray}} = 
    quasimulapplystyle(MemoryLayout(A), MemoryLayout(B), MemoryLayout(C), MemoryLayout.(D)...)
ApplyStyle(::typeof(*), ::Type{A}, ::Type{B}, ::Type{C}, D::Type...) where {A<:AbstractArray,B<:AbstractArray,C<:AbstractQuasiArray} = 
    quasimulapplystyle(MemoryLayout(A), MemoryLayout(B), MemoryLayout(C), MemoryLayout.(D)...)
ApplyStyle(::typeof(*), ::Type{A}, ::Type{B}, ::Type{C}, ::Type{D}, E::Type...) where {A<:AbstractQuasiArray,B<:Union{AbstractArray,AbstractQuasiArray},C<:Union{AbstractArray,AbstractQuasiArray},D<:Union{AbstractArray,AbstractQuasiArray}} = 
    quasimulapplystyle(MemoryLayout(A), MemoryLayout(B), MemoryLayout(C), MemoryLayout(D), MemoryLayout.(E)...)
ApplyStyle(::typeof(*), ::Type{A}, ::Type{B}, ::Type{C}, ::Type{D}, E::Type...) where {A<:AbstractArray,B<:AbstractQuasiArray,C<:Union{AbstractArray,AbstractQuasiArray},D<:Union{AbstractArray,AbstractQuasiArray}} = 
    quasimulapplystyle(MemoryLayout(A), MemoryLayout(B), MemoryLayout(C), MemoryLayout(D), MemoryLayout.(E)...)
ApplyStyle(::typeof(*), ::Type{A}, ::Type{B}, ::Type{C}, ::Type{D}, E::Type...) where {A<:AbstractArray,B<:AbstractArray,C<:AbstractQuasiArray,D<:Union{AbstractArray,AbstractQuasiArray}} = 
    quasimulapplystyle(MemoryLayout(A), MemoryLayout(B), MemoryLayout(C), MemoryLayout(D), MemoryLayout.(E)...)
ApplyStyle(::typeof(*), ::Type{A}, ::Type{B}, ::Type{C}, ::Type{D}, E::Type...) where {A<:AbstractArray,B<:AbstractArray,C<:AbstractArray,D<:AbstractQuasiArray} = 
    quasimulapplystyle(MemoryLayout(A), MemoryLayout(B), MemoryLayout(C), MemoryLayout(D), MemoryLayout.(E)...)    

ApplyStyle(::typeof(\), ::Type{<:AbstractQuasiArray}, ::Type{<:AbstractQuasiArray}) = QuasiArrayApplyStyle()
ApplyStyle(::typeof(\), ::Type{<:AbstractQuasiArray}, ::Type{<:AbstractArray}) = QuasiArrayApplyStyle()
ApplyStyle(::typeof(\), ::Type{<:AbstractArray}, ::Type{<:AbstractQuasiArray}) = QuasiArrayApplyStyle()    

for op in (:pinv, :inv)
    @eval ApplyStyle(::typeof($op), args::Type{<:AbstractQuasiArray}) = QuasiArrayApplyStyle()
end
## PInvQuasiMatrix


const PInvQuasiMatrix{T, ARGS} = ApplyQuasiMatrix{T,typeof(pinv),ARGS}
const InvQuasiMatrix{T, ARGS} = ApplyQuasiMatrix{T,typeof(inv),ARGS}

axes(A::PInvQuasiMatrix) = axes(A.applied)
size(A::PInvQuasiMatrix) = map(length, axes(A))
pinv(A::PInvQuasiMatrix) = first(A.args)

@propagate_inbounds getindex(A::PInvQuasiMatrix{T}, k::Int, j::Int) where T =
    (A.applied*[Zeros(j-1); one(T); Zeros(size(A,2) - j)])[k]

*(A::PInvQuasiMatrix, B::AbstractQuasiMatrix, C...) = apply(*,A.applied, B, C...)
*(A::PInvQuasiMatrix, B::MulQuasiArray, C...) = apply(*,A.applied, B.applied, C...)


####
# Matrix * Array
####



function _lmaterialize(A::MulQuasiArray, B, C...)
    As = A.args
    flatten(_ApplyArray(*, reverse(tail(reverse(As)))..., _lmaterialize(last(As), B, C...)))
end



function _rmaterialize(Z::MulQuasiArray, Y, W...)
    Zs = Z.args
    flatten(_ApplyArray(*, _rmaterialize(first(Zs), Y, W...), tail(Zs)...))
end


####
# Lazy \ ApplyArray. This applies to first arg.
#####

struct LmaterializeApplyStyle <: ApplyStyle end

quasimulapplystyle(::ApplyLayout{typeof(inv)}, _) = LdivApplyStyle()
quasimulapplystyle(::ApplyLayout{typeof(pinv)}, _) = LdivApplyStyle()
quasimulapplystyle(::ApplyLayout{typeof(inv)}, ::LazyLayout) = LdivApplyStyle()
quasimulapplystyle(::ApplyLayout{typeof(pinv)}, ::LazyLayout) = LdivApplyStyle()
quasimulapplystyle(::ApplyLayout{typeof(inv)}, ::LazyLayout, _...) = LmaterializeApplyStyle()
quasimulapplystyle(::ApplyLayout{typeof(pinv)}, ::LazyLayout, _...) = LmaterializeApplyStyle()
quasimulapplystyle(::ApplyLayout{typeof(*)}, ::ApplyLayout{typeof(*)}, _...) = LmaterializeApplyStyle()
quasimulapplystyle(::ApplyLayout{typeof(*)}, _...) = LmaterializeApplyStyle()
quasimulapplystyle(::ApplyLayout{typeof(*)}, ::LazyLayout, _...) = LmaterializeApplyStyle()
quasimulapplystyle(::ApplyLayout{typeof(*)}, ::LazyLayout, ::LazyLayout, _...) = LmaterializeApplyStyle()


ApplyStyle(::typeof(\), ::Type{<:AbstractQuasiArray}, ::Type{<:Applied}) = LdivApplyStyle()
\(A::AbstractQuasiArray, B::Applied) = apply(\, A, B)
materialize(L::Ldiv{LazyLayout,<:ApplyLayout{typeof(*)}}) = *(L.A \  first(L.B.args),  tail(L.B.args)...)
materialize(L::Ldiv{LazyLayout,<:ApplyLayout{typeof(*)},<:Any,<:AbstractQuasiArray}) = L.A \  L.B.applied


materialize(A::Applied{LmaterializeApplyStyle,typeof(*)}) = lmaterialize(A)