const QuasiArrayMulArray{p, q, T, V} = Mul{<:Any, <:Any, <:AbstractQuasiArray{T,p}, <:AbstractArray{V,q}}
const ArrayMulQuasiArray{p, q, T, V} = Mul{<:Any, <:Any, <:AbstractArray{T,p}, <:AbstractQuasiArray{V,q}}
const QuasiArrayMulQuasiArray{p, q, T, V} = Mul{<:Any, <:Any, <:AbstractQuasiArray{T,p}, <:AbstractQuasiArray{V,q}}
####
# Matrix * Vector
####
const QuasiMatMulVec{T, V} = QuasiArrayMulArray{2, 1, T, V}
const QuasiMatMulMat{T, V} = QuasiArrayMulArray{2, 2, T, V}
const QuasiMatMulQuasiMat{T, V} = QuasiArrayMulQuasiArray{2, 2, T, V}

combine_mul_styles(::MulStyle, ::QuasiArrayApplyStyle) = QuasiArrayApplyStyle()
combine_mul_styles(::QuasiArrayApplyStyle, ::MulStyle) = QuasiArrayApplyStyle()
combine_mul_styles(::DefaultArrayApplyStyle, ::QuasiArrayApplyStyle) = QuasiArrayApplyStyle()
combine_mul_styles(::QuasiArrayApplyStyle, ::DefaultArrayApplyStyle) = QuasiArrayApplyStyle()

ApplyStyle(::typeof(*), ::Type{<:AbstractQuasiArray}) = QuasiArrayApplyStyle()
ApplyStyle(::typeof(*), ::Type{<:AbstractArray}, ::Type{<:AbstractQuasiArray}) = MulStyle()
ApplyStyle(::typeof(*), ::Type{<:AbstractQuasiArray}, ::Type{<:AbstractArray}) = MulStyle()
ApplyStyle(::typeof(*), ::Type{<:AbstractQuasiArray}, ::Type{<:AbstractQuasiArray}) = MulStyle()


function getindex(M::Applied{<:AbstractQuasiArrayApplyStyle,typeof(*)}, k::Number)
    A,Bs = first(M.args), tail(M.args)
    B = _mul(Bs...)
    Mul(A, B)[k]
end

getindex(M::Applied{<:AbstractQuasiArrayApplyStyle,typeof(*)}, k::Int) =
    Base.invoke(getindex, Tuple{Applied{<:AbstractQuasiArrayApplyStyle,typeof(*)},Number}, M, k)

function _mul_quasi_getindex(M::Applied{<:AbstractQuasiArrayApplyStyle,typeof(*)}, k::Number, j::Number)
    A,Bs = first(M.args), tail(M.args)
    B = _mul(Bs...)
    Mul(A, B)[k, j]
end

getindex(M::Applied{<:AbstractQuasiArrayApplyStyle,typeof(*)}, k::Number, j::Number) =
    _mul_quasi_getindex(M, k, j)

getindex(M::Applied{<:AbstractQuasiArrayApplyStyle,typeof(*)}, k::Integer, j::Integer) =
    _mul_quasi_getindex(M, k, j)


function getindex(M::QuasiMatMulVec, k::AbstractArray)
    A,B = M.args
    ret = zeros(eltype(M),length(k))
    @inbounds for j in axes(A,2)
        ret .+= view(A,k,j) .* B[j]
    end
    ret
end

@inline axes(M::Mul{<:Any,<:Any,<:Any,<:AbstractQuasiMatrix}) = (axes(M.A, 1),axes(M.B,2))
@inline axes(M::Mul{<:Any,<:Any,<:Any,<:AbstractQuasiVector}) = (axes(M.A, 2),)
@inline axes(M::Mul{<:Any,<:Any,<:AbstractQuasiMatrix,<:AbstractMatrix}) = (axes(M.A, 1),axes(M.B,2))
@inline axes(M::Mul{<:Any,<:Any,<:AbstractQuasiMatrix,<:AbstractVector}) = (axes(M.A, 1),)

*(A::AbstractQuasiMatrix) = A
*(A::AbstractQuasiArray, B::AbstractQuasiArray) = mul(A, B)
*(A::AbstractArray, B::AbstractQuasiArray) = mul(A, B)
*(A::AbstractQuasiArray, B::AbstractArray) = mul(A, B)

for op in (:pinv, :inv)
    @eval $op(A::AbstractQuasiArray) = fullmaterialize(apply($op,A))
end

@inline axes(L::Ldiv{<:Any,<:Any,<:Any,<:AbstractQuasiMatrix}) = (axes(L.A, 2),axes(L.B,2))
@inline axes(L::Ldiv{<:Any,<:Any,<:Any,<:AbstractQuasiVector}) = (axes(L.A, 2),)

@inline \(A::AbstractQuasiArray, B::AbstractQuasiArray) = ldiv(A,B)
@inline \(A::AbstractQuasiArray, B::AbstractArray) = ldiv(A,B)
@inline \(A::AbstractArray, B::AbstractQuasiArray) = ldiv(A,B)

@inline /(A::AbstractQuasiArray, B::AbstractQuasiArray) = rdiv(A,B)
@inline /(A::AbstractQuasiArray, B::AbstractArray) = rdiv(A,B)
@inline /(A::AbstractArray, B::AbstractQuasiArray) = rdiv(A,B)


####
# MulQuasiArray
#####

const MulQuasiArray{T, N, Args<:Tuple} = ApplyQuasiArray{T, N, typeof(*), Args}

const MulQuasiVector{T, Args<:Tuple} = MulQuasiArray{T, 1, Args}
const MulQuasiMatrix{T, Args<:Tuple} = MulQuasiArray{T, 2, Args}

const Vec = MulQuasiVector


_ApplyArray(F, factors...) = ApplyQuasiArray(F, factors...)
_ApplyArray(F, factors::AbstractArray...) = ApplyArray(F, factors...)

MulQuasiOrArray = Union{MulArray,MulQuasiArray}

_factors(M::MulQuasiOrArray) = M.args
_factors(M) = (M,)

_flatten(A::MulQuasiArray, B...) = _flatten(Applied(A), B...)
flatten(A::MulQuasiArray) = ApplyQuasiArray(flatten(Applied(A)))
flatten(A::SubQuasiArray{<:Any,2,<:MulQuasiArray}) = materialize(flatten(Applied(A)))


function fullmaterialize(M::Applied{<:Any,typeof(*)})
    M_mat = materialize(flatten(M))
    typeof(M_mat) <: MulQuasiOrArray || return M_mat
    typeof(Applied(M_mat)) == typeof(M) || return(fullmaterialize(M_mat))

    ABC = M_mat.args
    length(ABC) ≤ 2 && return flatten(M_mat)

    AB = most(ABC)
    Mhead = fullmaterialize(applied(*,AB...))

    typeof(_factors(Mhead)) == typeof(AB) ||
        return fullmaterialize(applied(*, _factors(Mhead)..., last(ABC)))

    BC = tail(ABC)
    Mtail =  fullmaterialize(applied(*, BC...))
    typeof(_factors(Mtail)) == typeof(BC) ||
        return fullmaterialize(applied(*, first(ABC), _factors(Mtail)...))

    apply(*,first(ABC), Mtail.args...)
end

fullmaterialize(M::ApplyQuasiArray) = flatten(fullmaterialize(Applied(M)))
fullmaterialize(M) = flatten(M)


adjoint(A::MulQuasiArray) = ApplyQuasiArray(*, reverse(adjoint.(A.args))...)
transpose(A::MulQuasiArray) = ApplyQuasiArray(*, reverse(transpose.(A.args))...)

function similar(A::MulQuasiArray)
    B,a = A.args
    B*similar(a)
end

function similar(A::QuasiArrayMulArray)
    B,a = A.args
    applied(*, B, similar(a))
end

function copyto!(dest::MulQuasiArray, src::MulQuasiArray)
    d = last(dest.args)
    s = last(src.args)
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

getindex(A::MulQuasiMatrix, k::AbstractVector{<:Number}, j::AbstractVector{<:Number}) =
    _mul_getindex(k, j, A.args...)


struct QuasiArrayLayout <: MemoryLayout end
MemoryLayout(::Type{<:AbstractQuasiArray}) = QuasiArrayLayout()

copy(M::Mul{QuasiArrayLayout,QuasiArrayLayout}) = QuasiArray(M)
copy(M::Mul{QuasiArrayLayout}) = QuasiArray(M)
copy(M::Mul{<:Any,QuasiArrayLayout}) = QuasiArray(M)
copy(M::Mul{<:AbstractLazyLayout,QuasiArrayLayout}) = ApplyQuasiArray(M)
copy(M::Mul{QuasiArrayLayout,<:AbstractLazyLayout}) = ApplyQuasiArray(M)

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


ApplyStyle(::typeof(\), ::Type{<:AbstractQuasiArray}, ::Type{<:Applied}) = LdivStyle()
\(A::AbstractQuasiArray, B::Applied) = apply(\, A, B)
function copy(L::Ldiv{LazyLayout,ApplyLayout{typeof(*)},<:AbstractQuasiMatrix})
    args = arguments(L.B)
    apply(*, L.A \  first(args),  tail(args)...)
end

# copy(A::Applied{QuasiArrayApplyStyle,typeof(*)}) = lmaterialize(A)

import LazyArrays: MulStyle, _αAB, scalarone
_αAB(M::Applied{<:Any,typeof(*),<:Tuple{<:AbstractQuasiArray,<:AbstractQuasiArray}}, ::Type{T}) where T = tuple(scalarone(T), M.args...)
_αAB(M::Applied{<:Any,typeof(*),<:Tuple{<:Number,<:AbstractQuasiArray,<:AbstractQuasiArray}}, ::Type{T}) where T = M.args


###
# Scalar special case, simplifies x * A and A * x
# TODO: Find an AbstractArray to latch on to by commuting
###

function *(A::MulQuasiArray, x::Number)
    args = arguments(A)
    ApplyQuasiArray(*, most(args)..., args[end] * x)
end

function *(x::Number, A::MulQuasiArray)
    args = arguments(A)
    ApplyQuasiArray(*, x * args[1], tail(args)...)
end